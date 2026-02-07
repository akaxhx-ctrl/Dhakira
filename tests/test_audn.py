"""Tests for AUDN consolidation cycle (mocked LLM + vector store)."""

import pytest

from dhakira.consolidation.audn import AUDNCycle
from dhakira.config import ConsolidationConfig
from dhakira.models import AUDNAction, Fact, FactCategory, MemoryRecord, SearchResult


class MockLLM:
    def __init__(self, response: dict | None = None):
        self.response = response or {"action": "ADD", "reason": "default"}
        self.calls = []

    async def generate_structured(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        self.calls.append(prompt)
        return self.response


class MockVectorStore:
    def __init__(self, results: list[SearchResult] | None = None):
        self.results = results or []

    async def search(self, embedding, limit=10, filters=None):
        return self.results


@pytest.fixture
def sample_fact():
    return Fact(text="يحب القهوة العربية", category=FactCategory.PREFERENCE, confidence=0.9)


@pytest.fixture
def sample_embedding():
    return [0.1] * 128


class TestAUDNThresholdSkip:
    @pytest.mark.asyncio
    async def test_add_when_no_similar_memories(self, sample_fact, sample_embedding):
        llm = MockLLM()
        store = MockVectorStore([])
        audn = AUDNCycle(llm, store)

        decision = await audn.process(sample_fact, sample_embedding)
        assert decision.action == AUDNAction.ADD
        assert len(llm.calls) == 0  # No LLM call needed

    @pytest.mark.asyncio
    async def test_add_when_similarity_below_threshold(self, sample_fact, sample_embedding):
        record = MemoryRecord(text="يحب الشاي", embedding=[0.2] * 128)
        results = [SearchResult(record=record, score=0.3, source="vector")]

        llm = MockLLM()
        store = MockVectorStore(results)
        audn = AUDNCycle(llm, store, ConsolidationConfig(similarity_threshold=0.5))

        decision = await audn.process(sample_fact, sample_embedding)
        assert decision.action == AUDNAction.ADD
        assert len(llm.calls) == 0  # Skipped LLM call


class TestAUDNWithLLM:
    @pytest.mark.asyncio
    async def test_update_decision(self, sample_fact, sample_embedding):
        record = MemoryRecord(id="mem_1", text="يحب القهوة", embedding=[0.2] * 128)
        results = [SearchResult(record=record, score=0.8, source="vector")]

        llm = MockLLM({
            "action": "UPDATE",
            "target_id": "mem_1",
            "merged_text": "يحب القهوة العربية بدون سكر",
            "reason": "more specific",
        })
        store = MockVectorStore(results)
        audn = AUDNCycle(llm, store)

        decision = await audn.process(sample_fact, sample_embedding)
        assert decision.action == AUDNAction.UPDATE
        assert decision.target_id == "mem_1"
        assert decision.merged_text == "يحب القهوة العربية بدون سكر"
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_delete_decision(self, sample_fact, sample_embedding):
        record = MemoryRecord(id="mem_2", text="يكره القهوة", embedding=[0.2] * 128)
        results = [SearchResult(record=record, score=0.7, source="vector")]

        llm = MockLLM({
            "action": "DELETE",
            "target_id": "mem_2",
            "reason": "contradicts new info",
        })
        store = MockVectorStore(results)
        audn = AUDNCycle(llm, store)

        decision = await audn.process(sample_fact, sample_embedding)
        assert decision.action == AUDNAction.DELETE
        assert decision.target_id == "mem_2"

    @pytest.mark.asyncio
    async def test_noop_decision(self, sample_fact, sample_embedding):
        record = MemoryRecord(id="mem_3", text="يحب القهوة العربية", embedding=[0.2] * 128)
        results = [SearchResult(record=record, score=0.95, source="vector")]

        llm = MockLLM({"action": "NOOP", "reason": "already known"})
        store = MockVectorStore(results)
        audn = AUDNCycle(llm, store)

        decision = await audn.process(sample_fact, sample_embedding)
        assert decision.action == AUDNAction.NOOP


class TestAUDNErrorHandling:
    @pytest.mark.asyncio
    async def test_llm_error_defaults_to_add(self, sample_fact, sample_embedding):
        class ErrorLLM:
            async def generate_structured(self, *a, **kw):
                raise Exception("API Error")

        record = MemoryRecord(text="نص", embedding=[0.2] * 128)
        results = [SearchResult(record=record, score=0.8, source="vector")]

        store = MockVectorStore(results)
        audn = AUDNCycle(ErrorLLM(), store)

        decision = await audn.process(sample_fact, sample_embedding)
        assert decision.action == AUDNAction.ADD

    @pytest.mark.asyncio
    async def test_invalid_action_defaults_to_add(self, sample_fact, sample_embedding):
        record = MemoryRecord(text="نص", embedding=[0.2] * 128)
        results = [SearchResult(record=record, score=0.8, source="vector")]

        llm = MockLLM({"action": "INVALID", "reason": "bad"})
        store = MockVectorStore(results)
        audn = AUDNCycle(llm, store)

        decision = await audn.process(sample_fact, sample_embedding)
        assert decision.action == AUDNAction.ADD
