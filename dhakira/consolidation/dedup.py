"""Embedding-based deduplication for memories."""

from __future__ import annotations

import logging

from dhakira.models import MemoryRecord
from dhakira.storage.base import VectorStore

logger = logging.getLogger(__name__)


class Deduplicator:
    """Detect and handle duplicate memories using embedding similarity.

    Checks if a new memory is essentially a duplicate of an existing one
    by comparing embedding similarity. Used as a fast pre-check before
    the full AUDN cycle.
    """

    def __init__(self, vector_store: VectorStore, threshold: float = 0.95):
        self.vector_store = vector_store
        self.threshold = threshold

    async def is_duplicate(
        self,
        embedding: list[float],
        scope: str = "user",
        scope_id: str = "",
    ) -> MemoryRecord | None:
        """Check if a memory with this embedding already exists.

        Args:
            embedding: The embedding to check for duplicates.
            scope: Memory scope.
            scope_id: Scope identifier.

        Returns:
            The existing MemoryRecord if a near-duplicate exists, None otherwise.
        """
        filters = {"scope": scope, "scope_id": scope_id}
        results = await self.vector_store.search(
            embedding=embedding,
            limit=1,
            filters=filters,
        )

        if results and results[0].score >= self.threshold:
            logger.debug(
                "Duplicate detected (similarity=%.3f): %s",
                results[0].score,
                results[0].record.text[:50],
            )
            return results[0].record

        return None
