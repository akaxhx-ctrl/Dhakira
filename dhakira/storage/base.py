"""Abstract VectorStore and GraphStore interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dhakira.models import Entity, MemoryRecord, Relationship, SearchResult, Subgraph


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    async def add(self, record: MemoryRecord) -> None:
        """Add a memory record to the store."""

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar records by embedding.

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results.
            filters: Optional filters (e.g., scope, scope_id).

        Returns:
            List of SearchResult objects sorted by relevance.
        """

    @abstractmethod
    async def update(self, id: str, record: MemoryRecord) -> None:
        """Update an existing record."""

    @abstractmethod
    async def delete(self, id: str, soft: bool = True) -> None:
        """Delete a record. If soft=True, mark as deleted instead of removing."""

    @abstractmethod
    async def get(self, id: str) -> MemoryRecord | None:
        """Get a record by ID."""

    @abstractmethod
    async def get_all(self, filters: dict | None = None) -> list[MemoryRecord]:
        """Get all records, optionally filtered."""

    @abstractmethod
    async def count(self, filters: dict | None = None) -> int:
        """Count records, optionally filtered."""


class GraphStore(ABC):
    """Abstract interface for graph storage backends."""

    @abstractmethod
    async def add_entity(self, entity: Entity) -> None:
        """Add an entity node to the graph."""

    @abstractmethod
    async def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship edge to the graph."""

    @abstractmethod
    async def get_neighbors(self, entity_id: str, depth: int = 1) -> Subgraph:
        """Get neighboring entities and relationships up to a given depth.

        Args:
            entity_id: The starting entity ID.
            depth: How many hops to traverse (default 1).

        Returns:
            A Subgraph containing discovered entities and relationships.
        """

    @abstractmethod
    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        """Search for entities by name or normalized name.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching Entity objects.
        """

    @abstractmethod
    async def invalidate_relationship(self, rel_id: str, reason: str) -> None:
        """Soft-invalidate a relationship (mark as invalid with reason)."""

    @abstractmethod
    async def get_all_entities(self) -> list[Entity]:
        """Get all entities in the graph."""

    @abstractmethod
    async def get_all_relationships(self) -> list[Relationship]:
        """Get all relationships in the graph."""

    @abstractmethod
    async def save(self) -> None:
        """Persist graph to storage (if applicable)."""

    @abstractmethod
    async def load(self) -> None:
        """Load graph from storage (if applicable)."""
