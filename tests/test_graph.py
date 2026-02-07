"""Tests for graph store (NetworkX)."""

import pytest

from dhakira.models import Entity, EntityType, Relationship, Subgraph
from dhakira.storage.graph.networkx_ import NetworkXGraphStore


@pytest.fixture
def graph_store():
    return NetworkXGraphStore()


@pytest.fixture
def sample_entities():
    return [
        Entity(id="e1", name="أحمد", name_normalized="احمد", entity_type=EntityType.PERSON),
        Entity(id="e2", name="القاهرة", name_normalized="القاهره", entity_type=EntityType.PLACE),
        Entity(id="e3", name="شركة الاتصالات", name_normalized="شركه الاتصالات", entity_type=EntityType.ORGANIZATION),
        Entity(id="e4", name="محمد", name_normalized="محمد", entity_type=EntityType.PERSON),
    ]


@pytest.fixture
def sample_relationships():
    return [
        Relationship(id="r1", source_id="e1", target_id="e2", relation="يعيش في"),
        Relationship(id="r2", source_id="e1", target_id="e3", relation="يعمل في"),
        Relationship(id="r3", source_id="e4", target_id="e2", relation="زار"),
    ]


class TestAddEntity:
    @pytest.mark.asyncio
    async def test_add_entity(self, graph_store, sample_entities):
        await graph_store.add_entity(sample_entities[0])
        entities = await graph_store.get_all_entities()
        assert len(entities) == 1
        assert entities[0].name == "أحمد"

    @pytest.mark.asyncio
    async def test_add_multiple_entities(self, graph_store, sample_entities):
        for entity in sample_entities:
            await graph_store.add_entity(entity)
        entities = await graph_store.get_all_entities()
        assert len(entities) == 4


class TestAddRelationship:
    @pytest.mark.asyncio
    async def test_add_relationship(self, graph_store, sample_entities, sample_relationships):
        for entity in sample_entities:
            await graph_store.add_entity(entity)
        await graph_store.add_relationship(sample_relationships[0])
        rels = await graph_store.get_all_relationships()
        assert len(rels) == 1
        assert rels[0].relation == "يعيش في"


class TestGetNeighbors:
    @pytest.mark.asyncio
    async def test_direct_neighbors(self, graph_store, sample_entities, sample_relationships):
        for entity in sample_entities:
            await graph_store.add_entity(entity)
        for rel in sample_relationships:
            await graph_store.add_relationship(rel)

        subgraph = await graph_store.get_neighbors("e1", depth=1)
        entity_ids = {e.id for e in subgraph.entities}
        assert "e1" in entity_ids
        assert "e2" in entity_ids  # lives in Cairo
        assert "e3" in entity_ids  # works at company

    @pytest.mark.asyncio
    async def test_depth_2_neighbors(self, graph_store, sample_entities, sample_relationships):
        for entity in sample_entities:
            await graph_store.add_entity(entity)
        for rel in sample_relationships:
            await graph_store.add_relationship(rel)

        subgraph = await graph_store.get_neighbors("e1", depth=2)
        entity_ids = {e.id for e in subgraph.entities}
        # e4 is connected to e2 (which connects to e1), so should be found at depth 2
        assert "e4" in entity_ids

    @pytest.mark.asyncio
    async def test_nonexistent_entity(self, graph_store):
        subgraph = await graph_store.get_neighbors("nonexistent", depth=1)
        assert isinstance(subgraph, Subgraph)
        assert len(subgraph.entities) == 0


class TestSearchEntities:
    @pytest.mark.asyncio
    async def test_search_by_name(self, graph_store, sample_entities):
        for entity in sample_entities:
            await graph_store.add_entity(entity)

        results = await graph_store.search_entities("أحمد")
        assert len(results) >= 1
        assert any(e.name == "أحمد" for e in results)

    @pytest.mark.asyncio
    async def test_search_partial_match(self, graph_store, sample_entities):
        for entity in sample_entities:
            await graph_store.add_entity(entity)

        results = await graph_store.search_entities("الاتصالات")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_no_match(self, graph_store, sample_entities):
        for entity in sample_entities:
            await graph_store.add_entity(entity)

        results = await graph_store.search_entities("لندن")
        assert len(results) == 0


class TestInvalidateRelationship:
    @pytest.mark.asyncio
    async def test_invalidate(self, graph_store, sample_entities, sample_relationships):
        for entity in sample_entities:
            await graph_store.add_entity(entity)
        await graph_store.add_relationship(sample_relationships[0])

        await graph_store.invalidate_relationship("r1", "moved to a new city")

        rels = await graph_store.get_all_relationships()
        assert not rels[0].is_valid
        assert rels[0].metadata.get("invalidation_reason") == "moved to a new city"

    @pytest.mark.asyncio
    async def test_invalidated_excluded_from_neighbors(self, graph_store, sample_entities, sample_relationships):
        for entity in sample_entities:
            await graph_store.add_entity(entity)
        for rel in sample_relationships:
            await graph_store.add_relationship(rel)

        await graph_store.invalidate_relationship("r1", "moved")

        subgraph = await graph_store.get_neighbors("e1", depth=1)
        rel_ids = {r.id for r in subgraph.relationships}
        assert "r1" not in rel_ids


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path, sample_entities, sample_relationships):
        path = str(tmp_path / "test_graph.pkl")

        # Create and populate store
        from dhakira.config import GraphStoreConfig
        store1 = NetworkXGraphStore(GraphStoreConfig(path=path))
        for entity in sample_entities:
            await store1.add_entity(entity)
        for rel in sample_relationships:
            await store1.add_relationship(rel)
        await store1.save()

        # Load in new store
        store2 = NetworkXGraphStore(GraphStoreConfig(path=path))
        entities = await store2.get_all_entities()
        rels = await store2.get_all_relationships()
        assert len(entities) == 4
        assert len(rels) == 3
