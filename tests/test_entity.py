"""Tests for entity + relationship extraction (mocked LLM)."""

import pytest

from dhakira.extraction.entity import EntityExtractor
from dhakira.models import EntityType, Fact, FactCategory


class MockLLM:
    def __init__(self, response: dict | None = None):
        self.response = response or {"entities": [], "relationships": []}

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return ""

    async def generate_structured(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        return self.response


class TestEntityExtraction:
    @pytest.mark.asyncio
    async def test_basic_entity_extraction(self):
        mock_llm = MockLLM({
            "entities": [
                {"name": "أحمد", "type": "person", "summary": "شخص في المحادثة"},
                {"name": "القاهرة", "type": "place", "summary": "مدينة عربية"},
            ],
            "relationships": [
                {"source": "أحمد", "target": "القاهرة", "relation": "يعيش في"},
            ],
        })
        extractor = EntityExtractor(mock_llm)

        entities, relationships = await extractor.extract("أحمد يعيش في القاهرة")

        assert len(entities) == 2
        assert entities[0].name == "أحمد"
        assert entities[0].entity_type == EntityType.PERSON
        assert entities[1].name == "القاهرة"
        assert entities[1].entity_type == EntityType.PLACE

        assert len(relationships) == 1
        assert relationships[0].relation == "يعيش في"

    @pytest.mark.asyncio
    async def test_entity_normalization(self):
        mock_llm = MockLLM({
            "entities": [{"name": "أحمد", "type": "person"}],
            "relationships": [],
        })
        extractor = EntityExtractor(mock_llm)
        entities, _ = await extractor.extract("أحمد")
        assert entities[0].name_normalized != ""

    @pytest.mark.asyncio
    async def test_unknown_entity_type_defaults_to_concept(self):
        mock_llm = MockLLM({
            "entities": [{"name": "شيء", "type": "unknown_type"}],
            "relationships": [],
        })
        extractor = EntityExtractor(mock_llm)
        entities, _ = await extractor.extract("شيء")
        assert entities[0].entity_type == EntityType.CONCEPT

    @pytest.mark.asyncio
    async def test_relationship_creates_missing_entities(self):
        mock_llm = MockLLM({
            "entities": [],
            "relationships": [
                {"source": "خالد", "target": "مكتب", "relation": "يعمل في"},
            ],
        })
        extractor = EntityExtractor(mock_llm)
        entities, relationships = await extractor.extract("خالد يعمل في المكتب")

        # Entities should be created for source and target
        assert len(entities) == 2
        assert len(relationships) == 1

    @pytest.mark.asyncio
    async def test_with_facts_context(self):
        mock_llm = MockLLM({"entities": [], "relationships": []})
        extractor = EntityExtractor(mock_llm)

        facts = [Fact(text="يعمل في شركة", category=FactCategory.FACT)]
        await extractor.extract("نص", facts=facts)
        # Should not raise

    @pytest.mark.asyncio
    async def test_empty_text(self):
        mock_llm = MockLLM({"entities": [], "relationships": []})
        extractor = EntityExtractor(mock_llm)
        entities, rels = await extractor.extract("")
        assert entities == []
        assert rels == []

    @pytest.mark.asyncio
    async def test_llm_error(self):
        class ErrorLLM:
            async def generate_structured(self, *a, **kw):
                raise Exception("fail")

        extractor = EntityExtractor(ErrorLLM())
        entities, rels = await extractor.extract("نص")
        assert entities == []
        assert rels == []
