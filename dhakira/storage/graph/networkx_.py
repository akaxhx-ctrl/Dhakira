"""NetworkX graph store implementation (in-memory, persistence via pickle)."""

from __future__ import annotations

import logging
import os
import pickle

import networkx as nx

from dhakira.config import GraphStoreConfig
from dhakira.models import Entity, Relationship, Subgraph
from dhakira.storage.base import GraphStore

logger = logging.getLogger(__name__)


class NetworkXGraphStore(GraphStore):
    """NetworkX-based graph store.

    In-memory graph with optional pickle-based persistence.
    Zero setup required — ideal for development and small-to-medium datasets.
    """

    def __init__(self, config: GraphStoreConfig | None = None):
        self.config = config or GraphStoreConfig()
        self._graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}

        # Auto-load if persistence path exists
        if self.config.path and os.path.exists(self.config.path):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync context, will load lazily
                    self._load_sync()
                else:
                    loop.run_until_complete(self.load())
            except RuntimeError:
                self._load_sync()

    def _load_sync(self):
        """Synchronous load for initialization."""
        if self.config.path and os.path.exists(self.config.path):
            try:
                with open(self.config.path, "rb") as f:
                    data = pickle.load(f)
                self._graph = data.get("graph", nx.DiGraph())
                self._entities = data.get("entities", {})
                self._relationships = data.get("relationships", {})
                logger.info("Loaded graph from %s (%d entities, %d relationships)",
                            self.config.path, len(self._entities), len(self._relationships))
            except Exception as e:
                logger.warning("Failed to load graph from %s: %s", self.config.path, e)

    async def add_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity
        self._graph.add_node(
            entity.id,
            name=entity.name,
            name_normalized=entity.name_normalized,
            entity_type=entity.entity_type.value,
            summary=entity.summary,
        )

    async def add_relationship(self, rel: Relationship) -> None:
        self._relationships[rel.id] = rel
        self._graph.add_edge(
            rel.source_id,
            rel.target_id,
            id=rel.id,
            relation=rel.relation,
            is_valid=rel.is_valid,
            valid_from=rel.valid_from,
            valid_until=rel.valid_until,
        )

    async def get_neighbors(self, entity_id: str, depth: int = 1) -> Subgraph:
        if entity_id not in self._graph:
            return Subgraph()

        # BFS to find neighbors within depth
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()
        queue = [(entity_id, 0)]
        visited_nodes.add(entity_id)

        while queue:
            current, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue

            # Get both successors and predecessors (bidirectional)
            neighbors = set(self._graph.successors(current)) | set(self._graph.predecessors(current))

            for neighbor in neighbors:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

                # Track edges
                for edge_data in [self._graph.get_edge_data(current, neighbor),
                                  self._graph.get_edge_data(neighbor, current)]:
                    if edge_data and edge_data.get("id"):
                        visited_edges.add(edge_data["id"])

        entities = [self._entities[nid] for nid in visited_nodes if nid in self._entities]
        relationships = [
            self._relationships[rid] for rid in visited_edges
            if rid in self._relationships and self._relationships[rid].is_valid
        ]

        return Subgraph(entities=entities, relationships=relationships)

    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        query_lower = query.lower()
        results = []

        for entity in self._entities.values():
            # Match against name and normalized name
            if (query_lower in entity.name.lower() or
                    query_lower in entity.name_normalized.lower() or
                    (entity.summary and query_lower in entity.summary.lower())):
                results.append(entity)

        return results[:limit]

    async def invalidate_relationship(self, rel_id: str, reason: str) -> None:
        if rel_id in self._relationships:
            rel = self._relationships[rel_id]
            rel.is_valid = False
            rel.metadata["invalidation_reason"] = reason

            # Update edge in graph
            if self._graph.has_edge(rel.source_id, rel.target_id):
                self._graph[rel.source_id][rel.target_id]["is_valid"] = False

    async def get_all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    async def get_all_relationships(self) -> list[Relationship]:
        return list(self._relationships.values())

    async def save(self) -> None:
        if not self.config.path:
            return

        os.makedirs(os.path.dirname(self.config.path) if os.path.dirname(self.config.path) else ".", exist_ok=True)

        data = {
            "graph": self._graph,
            "entities": self._entities,
            "relationships": self._relationships,
        }
        with open(self.config.path, "wb") as f:
            pickle.dump(data, f)

        logger.info("Saved graph to %s (%d entities, %d relationships)",
                     self.config.path, len(self._entities), len(self._relationships))

    async def load(self) -> None:
        self._load_sync()
