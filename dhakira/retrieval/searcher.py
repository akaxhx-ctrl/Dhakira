"""Hybrid search orchestrator (vector + BM25 + graph). Zero LLM calls."""

from __future__ import annotations

import asyncio
import logging

from dhakira.arabic.normalizer import ArabicNormalizer
from dhakira.config import RetrievalConfig
from dhakira.embeddings.base import BaseEmbeddings
from dhakira.models import MemoryResult, SearchResult
from dhakira.retrieval.bm25 import ArabicBM25
from dhakira.retrieval.reranker import Reranker
from dhakira.storage.base import GraphStore, VectorStore

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Hybrid search combining vector, BM25, and graph search.

    Zero LLM calls — uses embeddings + BM25 + graph traversal + local
    cross-encoder reranking. Results are fused using Reciprocal Rank
    Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: BaseEmbeddings,
        normalizer: ArabicNormalizer,
        bm25: ArabicBM25,
        graph_store: GraphStore | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.normalizer = normalizer
        self.bm25 = bm25
        self.graph_store = graph_store
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    async def search(
        self,
        query: str,
        scope: str = "user",
        scope_id: str = "",
        limit: int = 10,
    ) -> list[MemoryResult]:
        """Search memories using hybrid retrieval.

        1. Normalize query
        2. Embed query
        3. Run vector, BM25, and graph searches in parallel
        4. Fuse with RRF
        5. Rerank with cross-encoder
        6. Return top results

        Args:
            query: Search query in Arabic.
            scope: Memory scope (user/session/agent).
            scope_id: Scope identifier.
            limit: Maximum number of results.

        Returns:
            List of MemoryResult objects sorted by relevance.
        """
        # Normalize query
        normalized_query = self.normalizer.normalize_for_embedding(query)

        # Embed query
        query_embedding = await self.embeddings.embed(normalized_query)

        # Build filters
        filters = {"scope": scope, "scope_id": scope_id}
        fetch_limit = limit * 2  # Fetch more for reranking

        # Run searches in parallel
        tasks = [
            self._vector_search(query_embedding, fetch_limit, filters),
            self._bm25_search(normalized_query, fetch_limit, filters),
        ]
        if self.graph_store:
            tasks.append(self._graph_search(normalized_query, fetch_limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all results
        vector_results: list[SearchResult] = []
        bm25_results: list[SearchResult] = []
        graph_results: list[SearchResult] = []

        if not isinstance(results[0], Exception):
            vector_results = results[0]
        else:
            logger.warning("Vector search failed: %s", results[0])

        if not isinstance(results[1], Exception):
            bm25_results = results[1]
        else:
            logger.warning("BM25 search failed: %s", results[1])

        if len(results) > 2 and not isinstance(results[2], Exception):
            graph_results = results[2]
        elif len(results) > 2:
            logger.warning("Graph search failed: %s", results[2])

        # Fuse with RRF
        fused = self._rrf_fusion(vector_results, bm25_results, graph_results)

        if not fused:
            return []

        # Rerank with cross-encoder
        if self.reranker:
            fused = await self.reranker.rerank(query, fused)

        # Convert to MemoryResult
        return [
            MemoryResult(
                id=r.record.id,
                text=r.record.text_original or r.record.text,
                score=r.score,
                category=r.record.category,
                dialect=r.record.dialect,
                created_at=r.record.created_at,
                metadata=r.record.metadata,
            )
            for r in fused[:limit]
        ]

    async def _vector_search(
        self,
        embedding: list[float],
        limit: int,
        filters: dict,
    ) -> list[SearchResult]:
        return await self.vector_store.search(
            embedding=embedding,
            limit=limit,
            filters=filters,
        )

    async def _bm25_search(
        self,
        query: str,
        limit: int,
        filters: dict,
    ) -> list[SearchResult]:
        return self.bm25.search(query=query, limit=limit, filters=filters)

    async def _graph_search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Search graph for entities matching the query, then fetch related memories."""
        if not self.graph_store:
            return []

        entities = await self.graph_store.search_entities(query, limit=5)
        if not entities:
            return []

        # Get neighbors for found entities
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for entity in entities:
            subgraph = await self.graph_store.get_neighbors(entity.id, depth=2)
            for related_entity in subgraph.entities:
                if related_entity.id not in seen_ids:
                    seen_ids.add(related_entity.id)
                    # Search vector store for memories mentioning this entity
                    entity_embedding = await self.embeddings.embed(related_entity.name_normalized or related_entity.name)
                    results = await self.vector_store.search(
                        embedding=entity_embedding,
                        limit=3,
                    )
                    for r in results:
                        r.source = "graph"
                    all_results.extend(results)

        return all_results[:limit]

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        graph_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion: score = sum(weight / (k + rank_i))."""
        k = self.config.rrf_k
        scores: dict[str, float] = {}
        records: dict[str, SearchResult] = {}

        # Vector results
        for rank, result in enumerate(vector_results):
            rid = result.record.id
            scores[rid] = scores.get(rid, 0) + self.config.vector_weight / (k + rank + 1)
            records[rid] = result

        # BM25 results
        for rank, result in enumerate(bm25_results):
            rid = result.record.id
            scores[rid] = scores.get(rid, 0) + self.config.bm25_weight / (k + rank + 1)
            if rid not in records:
                records[rid] = result

        # Graph results
        for rank, result in enumerate(graph_results):
            rid = result.record.id
            scores[rid] = scores.get(rid, 0) + self.config.graph_weight / (k + rank + 1)
            if rid not in records:
                records[rid] = result

        # Build fused results
        fused = []
        for rid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            result = records[rid]
            result.score = score
            fused.append(result)

        return fused
