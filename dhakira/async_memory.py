"""AsyncMemory class — async public API for Dhakira."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from dhakira.arabic.normalizer import ArabicNormalizer
from dhakira.cache.semantic import SemanticCache
from dhakira.config import DhakiraConfig
from dhakira.consolidation.audn import AUDNCycle
from dhakira.consolidation.dedup import Deduplicator
from dhakira.embeddings.factory import create_embeddings
from dhakira.extraction.entity import EntityExtractor
from dhakira.extraction.extractor import FactExtractor
from dhakira.llm.factory import create_llm
from dhakira.models import (
    AUDNAction,
    FactCategory,
    MemoryRecord,
    MemoryResult,
    Message,
)
from dhakira.retrieval.bm25 import ArabicBM25
from dhakira.retrieval.reranker import Reranker
from dhakira.retrieval.searcher import HybridSearcher
from dhakira.storage.graph.factory import create_graph_store
from dhakira.storage.vector.factory import create_vector_store

logger = logging.getLogger(__name__)


class AsyncMemory:
    """Asynchronous Dhakira memory interface.

    Full pipeline: Arabic preprocessing → LLM extraction → AUDN consolidation
    → vector + graph storage → hybrid retrieval.

    Usage:
        memory = AsyncMemory()
        await memory.add(messages=[...], user_id="user_123")
        results = await memory.search(query="...", user_id="user_123")
    """

    def __init__(self, config: DhakiraConfig | None = None):
        self.config = config or DhakiraConfig()

        # Arabic processing
        self.normalizer = ArabicNormalizer(self.config.arabic)

        # LLM + embeddings
        self.llm = create_llm(self.config.llm)
        self.embeddings = create_embeddings(self.config.embeddings)

        # Storage
        self.vector_store = create_vector_store(
            self.config.vector_store,
            embedding_dim=self.config.embeddings.dim,
        )
        self.graph_store = create_graph_store(self.config.graph_store)

        # Extraction
        self.fact_extractor = FactExtractor(self.llm, self.normalizer)
        self.entity_extractor = EntityExtractor(self.llm, self.normalizer)

        # Consolidation
        self.audn = AUDNCycle(self.llm, self.vector_store, self.config.consolidation)
        self.dedup = Deduplicator(self.vector_store)

        # Retrieval
        self.bm25 = ArabicBM25(self.config.retrieval.bm25)
        self.reranker = Reranker(self.config.retrieval.reranker)
        self.searcher = HybridSearcher(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            normalizer=self.normalizer,
            bm25=self.bm25,
            graph_store=self.graph_store,
            reranker=self.reranker,
            config=self.config.retrieval,
        )

        # Cache
        self.cache = SemanticCache(self.config.cache)

    def _resolve_scope(
        self,
        user_id: str | None,
        session_id: str | None,
        agent_id: str | None,
    ) -> tuple[str, str]:
        """Resolve scope and scope_id from provided identifiers."""
        if agent_id:
            return "agent", agent_id
        if session_id:
            return "session", session_id
        if user_id:
            return "user", user_id
        return "user", "default"

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[str]:
        """Add memories from a conversation.

        Pipeline:
        1. Normalize Arabic content
        2. Extract facts using LLM (with cache check)
        3. Extract entities + relationships for graph
        4. For each fact: embed → dedup check → AUDN → store
        5. Store entities and relationships in graph

        Returns:
            List of created/updated memory IDs.
        """
        scope, scope_id = self._resolve_scope(user_id, session_id, agent_id)
        metadata = metadata or {}

        # Convert to Message objects
        msgs = [Message(role=m["role"], content=m["content"]) for m in messages]

        # Build content key for cache
        content_key = "\n".join(f"{m.role}: {m.content}" for m in msgs)

        # Check cache
        cached = self.cache.get(content_key)
        if cached is not None:
            logger.debug("Cache hit for extraction")
            facts_data = cached.get("facts", [])
            from dhakira.models import Fact
            facts = [Fact(**f) for f in facts_data]
        else:
            # Extract facts using LLM
            facts = await self.fact_extractor.extract(msgs)

            # Cache the result
            self.cache.put(content_key, {"facts": [f.model_dump() for f in facts]})

        if not facts:
            logger.debug("No facts extracted from conversation")
            return []

        # Extract entities and relationships for graph
        full_content = " ".join(m.content for m in msgs)
        entities, relationships = await self.entity_extractor.extract(full_content, facts)

        # Store entities and relationships in graph
        for entity in entities:
            await self.graph_store.add_entity(entity)
        for rel in relationships:
            await self.graph_store.add_relationship(rel)

        # Process each fact through the AUDN cycle
        memory_ids: list[str] = []

        for fact in facts:
            # Normalize for embedding
            normalized_text = self.normalizer.normalize_for_embedding(fact.text)
            embedding = await self.embeddings.embed(normalized_text)

            # Dedup check
            existing = await self.dedup.is_duplicate(embedding, scope, scope_id)
            if existing:
                logger.debug("Duplicate detected, skipping: %s", fact.text[:50])
                continue

            # AUDN cycle
            decision = await self.audn.process(fact, embedding, scope, scope_id)

            if decision.action == AUDNAction.ADD:
                record = MemoryRecord(
                    text=self.normalizer.normalize(fact.text),
                    text_original=fact.text,
                    embedding=embedding,
                    category=fact.category,
                    scope=scope,
                    scope_id=scope_id,
                    confidence=fact.confidence,
                    metadata=metadata,
                )
                await self.vector_store.add(record)
                self.bm25.add_document(record)
                memory_ids.append(record.id)

            elif decision.action == AUDNAction.UPDATE and decision.target_id:
                merged_text = decision.merged_text or fact.text
                merged_normalized = self.normalizer.normalize_for_embedding(merged_text)
                merged_embedding = await self.embeddings.embed(merged_normalized)

                existing_record = await self.vector_store.get(decision.target_id)
                if existing_record:
                    existing_record.text = self.normalizer.normalize(merged_text)
                    existing_record.text_original = merged_text
                    existing_record.embedding = merged_embedding
                    existing_record.updated_at = datetime.now(timezone.utc)
                    existing_record.metadata.update(metadata)
                    await self.vector_store.update(decision.target_id, existing_record)
                    self.bm25.update_document(existing_record)
                    memory_ids.append(decision.target_id)

            elif decision.action == AUDNAction.DELETE and decision.target_id:
                await self.vector_store.delete(decision.target_id, soft=True)
                self.bm25.remove_document(decision.target_id)

                # Add the new fact as a replacement
                record = MemoryRecord(
                    text=self.normalizer.normalize(fact.text),
                    text_original=fact.text,
                    embedding=embedding,
                    category=fact.category,
                    scope=scope,
                    scope_id=scope_id,
                    confidence=fact.confidence,
                    metadata=metadata,
                )
                await self.vector_store.add(record)
                self.bm25.add_document(record)
                memory_ids.append(record.id)

            # NOOP: do nothing

        # Save graph if configured
        await self.graph_store.save()

        return memory_ids

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryResult]:
        """Search memories (zero LLM calls).

        Uses hybrid retrieval: vector + BM25 + graph + reranker.
        """
        scope, scope_id = self._resolve_scope(user_id, session_id, agent_id)
        return await self.searcher.search(
            query=query,
            scope=scope,
            scope_id=scope_id,
            limit=limit,
        )

    async def get_all(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryResult]:
        """Get all memories for a scope."""
        scope, scope_id = self._resolve_scope(user_id, session_id, agent_id)

        records = await self.vector_store.get_all(
            filters={"scope": scope, "scope_id": scope_id},
        )

        return [
            MemoryResult(
                id=r.id,
                text=r.text_original or r.text,
                score=1.0,
                category=r.category,
                dialect=r.dialect,
                created_at=r.created_at,
                metadata=r.metadata,
            )
            for r in records
            if not r.is_deleted
        ]

    async def update(self, memory_id: str, text: str) -> None:
        """Update a memory's text."""
        record = await self.vector_store.get(memory_id)
        if not record:
            raise ValueError(f"Memory not found: {memory_id}")

        normalized = self.normalizer.normalize_for_embedding(text)
        embedding = await self.embeddings.embed(normalized)

        record.text = self.normalizer.normalize(text)
        record.text_original = text
        record.embedding = embedding
        record.updated_at = datetime.now(timezone.utc)

        await self.vector_store.update(memory_id, record)
        self.bm25.update_document(record)

    async def delete(self, memory_id: str) -> None:
        """Soft-delete a memory."""
        await self.vector_store.delete(memory_id, soft=True)
        self.bm25.remove_document(memory_id)
