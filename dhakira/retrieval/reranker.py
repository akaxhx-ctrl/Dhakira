"""Local cross-encoder reranking."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from dhakira.config import RerankerConfig
from dhakira.models import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Local cross-encoder reranker.

    Default: BAAI/bge-reranker-v2-m3 (~278M params, CPU-compatible).
    Reranks search results using a cross-encoder model for improved
    accuracy (+3.16 faithfulness improvement).
    """

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        if not self.config.enabled:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.config.model,
                device=self.config.device,
            )
            logger.info("Loaded reranker model: %s", self.config.model)
        except Exception as e:
            logger.warning("Failed to load reranker model: %s. Reranking disabled.", e)
            self._model = None

    def _rerank_sync(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Synchronous reranking."""
        self._load_model()

        if self._model is None or not results:
            return results

        # Prepare pairs for cross-encoder
        pairs = [(query, r.record.text) for r in results]

        scores = self._model.predict(pairs, show_progress_bar=False)

        # Update scores
        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by new scores
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:self.config.top_k]

    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Rerank search results using the cross-encoder.

        Args:
            query: The search query.
            results: Search results to rerank.

        Returns:
            Reranked results (top_k).
        """
        if not self.config.enabled or not results:
            return results

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._rerank_sync, query, results),
        )
