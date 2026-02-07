"""Qdrant vector store implementation."""

from __future__ import annotations

import logging
from datetime import timezone

from dhakira.config import VectorStoreConfig
from dhakira.models import FactCategory, MemoryRecord, SearchResult
from dhakira.storage.base import VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant vector store backend.

    Supports both in-memory mode (default, zero setup) and persistent
    storage via a path or remote Qdrant server.
    """

    def __init__(self, config: VectorStoreConfig | None = None, embedding_dim: int = 128):
        self.config = config or VectorStoreConfig()
        self.embedding_dim = embedding_dim
        self._client = None
        self._collection_ready = False

    def _get_client(self):
        if self._client is not None:
            return self._client

        from qdrant_client import QdrantClient

        if self.config.path:
            self._client = QdrantClient(path=self.config.path)
        else:
            # In-memory mode
            self._client = QdrantClient(location=":memory:")

        self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        if self._collection_ready:
            return

        from qdrant_client.models import Distance, VectorParams

        client = self._client
        collections = [c.name for c in client.get_collections().collections]

        if self.config.collection_name not in collections:
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self.config.collection_name)

        self._collection_ready = True

    def _record_to_payload(self, record: MemoryRecord) -> dict:
        return {
            "text": record.text,
            "text_original": record.text_original,
            "category": record.category.value if isinstance(record.category, FactCategory) else record.category,
            "scope": record.scope,
            "scope_id": record.scope_id,
            "dialect": record.dialect.value if record.dialect else None,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "is_deleted": record.is_deleted,
            "confidence": record.confidence,
            "source_message_id": record.source_message_id,
            "metadata": record.metadata,
        }

    def _payload_to_record(self, point_id: str, payload: dict, vector: list[float] | None = None) -> MemoryRecord:
        from datetime import datetime

        dialect = None
        if payload.get("dialect"):
            from dhakira.models import Dialect
            try:
                dialect = Dialect(payload["dialect"])
            except ValueError:
                dialect = None

        return MemoryRecord(
            id=str(point_id),
            text=payload.get("text", ""),
            text_original=payload.get("text_original", ""),
            embedding=vector or [],
            category=FactCategory(payload.get("category", "fact")),
            scope=payload.get("scope", "user"),
            scope_id=payload.get("scope_id", ""),
            dialect=dialect,
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(payload["updated_at"]) if payload.get("updated_at") else datetime.now(timezone.utc),
            is_deleted=payload.get("is_deleted", False),
            confidence=payload.get("confidence", 1.0),
            source_message_id=payload.get("source_message_id"),
            metadata=payload.get("metadata", {}),
        )

    def _build_filters(self, filters: dict | None) -> object | None:
        if not filters:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []

        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        # Always exclude soft-deleted records unless explicitly requested
        if "is_deleted" not in filters:
            conditions.append(FieldCondition(key="is_deleted", match=MatchValue(value=False)))

        return Filter(must=conditions) if conditions else None

    async def add(self, record: MemoryRecord) -> None:
        from qdrant_client.models import PointStruct

        client = self._get_client()
        point = PointStruct(
            id=record.id,
            vector=record.embedding,
            payload=self._record_to_payload(record),
        )
        client.upsert(
            collection_name=self.config.collection_name,
            points=[point],
        )

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        client = self._get_client()

        query_filter = self._build_filters(filters)

        response = client.query_points(
            collection_name=self.config.collection_name,
            query=embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
        )

        search_results = []
        for hit in response.points:
            record = self._payload_to_record(
                str(hit.id),
                hit.payload or {},
                hit.vector if isinstance(hit.vector, list) else None,
            )
            search_results.append(SearchResult(record=record, score=hit.score, source="vector"))

        return search_results

    async def update(self, id: str, record: MemoryRecord) -> None:
        from qdrant_client.models import PointStruct

        client = self._get_client()
        point = PointStruct(
            id=id,
            vector=record.embedding,
            payload=self._record_to_payload(record),
        )
        client.upsert(
            collection_name=self.config.collection_name,
            points=[point],
        )

    async def delete(self, id: str, soft: bool = True) -> None:
        client = self._get_client()

        if soft:
            # Soft delete: set is_deleted flag
            client.set_payload(
                collection_name=self.config.collection_name,
                payload={"is_deleted": True},
                points=[id],
            )
        else:
            from qdrant_client.models import PointIdsList
            client.delete(
                collection_name=self.config.collection_name,
                points_selector=PointIdsList(points=[id]),
            )

    async def get(self, id: str) -> MemoryRecord | None:
        client = self._get_client()
        try:
            results = client.retrieve(
                collection_name=self.config.collection_name,
                ids=[id],
                with_payload=True,
                with_vectors=True,
            )
            if results:
                point = results[0]
                return self._payload_to_record(
                    str(point.id),
                    point.payload or {},
                    point.vector if isinstance(point.vector, list) else None,
                )
        except Exception:
            pass
        return None

    async def get_all(self, filters: dict | None = None) -> list[MemoryRecord]:
        client = self._get_client()
        query_filter = self._build_filters(filters)

        results = client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=10000,
        )

        records = []
        for point in results[0]:
            record = self._payload_to_record(
                str(point.id),
                point.payload or {},
                point.vector if isinstance(point.vector, list) else None,
            )
            records.append(record)
        return records

    async def count(self, filters: dict | None = None) -> int:
        client = self._get_client()
        if filters:
            # Use scroll with filter to count
            records = await self.get_all(filters)
            return len(records)
        else:
            info = client.get_collection(self.config.collection_name)
            return info.points_count or 0
