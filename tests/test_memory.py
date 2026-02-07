"""End-to-end memory tests with mocked LLM and embeddings."""

import pytest

from dhakira.async_memory import AsyncMemory
from dhakira.config import (
    CacheConfig,
    DhakiraConfig,
    EmbeddingsConfig,
    LLMConfig,
    RerankerConfig,
    RetrievalConfig,
)
from dhakira.embeddings.base import BaseEmbeddings
from dhakira.llm.base import BaseLLM
from dhakira.models import FactCategory


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self):
        self.call_count = 0

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.call_count += 1
        return ""

    async def generate_structured(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        self.call_count += 1
        system_lower = (system or "").lower()
        # Return simple extraction results
        if "memory extraction" in system_lower or "extract key facts" in system_lower:
            return {
                "facts": [
                    {"text": "اسمه أحمد", "category": "fact", "confidence": 0.95},
                    {"text": "يحب القهوة العربية", "category": "preference", "confidence": 0.9},
                ]
            }
        elif "entity" in system_lower and "relationship" in system_lower:
            return {
                "entities": [
                    {"name": "أحمد", "type": "person", "summary": "المستخدم"},
                ],
                "relationships": [],
            }
        elif "memory consolidation" in system_lower:
            return {"action": "ADD", "reason": "new fact"}
        return {}


class MockEmbeddings(BaseEmbeddings):
    """Mock embeddings that return deterministic vectors."""

    def __init__(self, dim: int = 128):
        self._dim = dim
        self.call_count = 0

    async def embed(self, text: str) -> list[float]:
        self.call_count += 1
        # Generate a deterministic vector based on text hash
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        base = [int(c, 16) / 15.0 for c in h]
        # Repeat to fill dimension
        vec = (base * (self._dim // len(base) + 1))[:self._dim]
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    def get_dimension(self) -> int:
        return self._dim


@pytest.fixture
def config():
    return DhakiraConfig(
        retrieval=RetrievalConfig(
            reranker=RerankerConfig(enabled=False),  # Disable reranker for tests
        ),
        cache=CacheConfig(enabled=False),
    )


@pytest.fixture
def async_memory(config):
    memory = AsyncMemory(config)
    # Replace with mocks
    mock_llm = MockLLM()
    memory.llm = mock_llm
    memory.fact_extractor.llm = mock_llm
    memory.entity_extractor.llm = mock_llm
    memory.audn.llm = mock_llm
    mock_embed = MockEmbeddings()
    memory.embeddings = mock_embed
    memory.searcher.embeddings = mock_embed
    # Set dedup threshold very high so it doesn't interfere with tests
    memory.dedup.threshold = 0.999
    return memory


class TestAddMemories:
    @pytest.mark.asyncio
    async def test_add_basic_conversation(self, async_memory):
        ids = await async_memory.add(
            messages=[
                {"role": "user", "content": "اسمي أحمد وأحب القهوة العربية"},
                {"role": "assistant", "content": "أهلا أحمد!"},
            ],
            user_id="user_123",
        )
        assert len(ids) > 0

    @pytest.mark.asyncio
    async def test_add_with_agent_scope(self, async_memory):
        ids = await async_memory.add(
            messages=[
                {"role": "assistant", "content": "تم تحليل البيانات بنجاح"},
            ],
            agent_id="agent_1",
        )
        assert len(ids) > 0

    @pytest.mark.asyncio
    async def test_add_empty_extraction(self, async_memory):
        # Override LLM to return no facts
        async_memory.llm = type(
            "EmptyLLM", (), {
                "generate_structured": lambda self, *a, **kw: self._ret(),
                "_ret": staticmethod(lambda: {}),
            },
        )()

        class EmptyLLM:
            async def generate_structured(self, *a, **kw):
                return {"facts": []}

        async_memory.llm = EmptyLLM()
        async_memory.fact_extractor.llm = async_memory.llm
        async_memory.entity_extractor.llm = async_memory.llm

        ids = await async_memory.add(
            messages=[{"role": "user", "content": "مرحبا"}],
            user_id="user_123",
        )
        assert ids == []


class TestSearchMemories:
    @pytest.mark.asyncio
    async def test_search_after_add(self, async_memory):
        await async_memory.add(
            messages=[
                {"role": "user", "content": "اسمي أحمد وأحب القهوة العربية"},
            ],
            user_id="user_123",
        )

        results = await async_memory.search(
            query="ما هي المشروبات المفضلة؟",
            user_id="user_123",
        )
        # Should find something (vector search)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_different_scope(self, async_memory):
        await async_memory.add(
            messages=[{"role": "user", "content": "اسمي أحمد"}],
            user_id="user_123",
        )

        # Search with different user should return no results
        results = await async_memory.search(
            query="ما اسمه؟",
            user_id="user_999",
        )
        assert len(results) == 0


class TestGetAllMemories:
    @pytest.mark.asyncio
    async def test_get_all(self, async_memory):
        await async_memory.add(
            messages=[{"role": "user", "content": "اسمي أحمد"}],
            user_id="user_123",
        )

        all_memories = await async_memory.get_all(user_id="user_123")
        assert len(all_memories) > 0
        assert all_memories[0].text  # Has text content


class TestUpdateMemory:
    @pytest.mark.asyncio
    async def test_update(self, async_memory):
        ids = await async_memory.add(
            messages=[{"role": "user", "content": "اسمي أحمد"}],
            user_id="user_123",
        )
        assert len(ids) > 0

        await async_memory.update(
            memory_id=ids[0],
            text="يفضل القهوة التركية بدلا من العربية",
        )

        record = await async_memory.vector_store.get(ids[0])
        assert record is not None
        assert "التركيه" in record.text or "التركية" in record.text

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self, async_memory):
        with pytest.raises(ValueError):
            await async_memory.update(memory_id="nonexistent", text="نص")


class TestDeleteMemory:
    @pytest.mark.asyncio
    async def test_soft_delete(self, async_memory):
        ids = await async_memory.add(
            messages=[{"role": "user", "content": "اسمي أحمد"}],
            user_id="user_123",
        )
        assert len(ids) > 0

        await async_memory.delete(memory_id=ids[0])

        # Should not appear in get_all
        all_memories = await async_memory.get_all(user_id="user_123")
        deleted_ids = {m.id for m in all_memories}
        assert ids[0] not in deleted_ids


class TestScopeResolution:
    @pytest.mark.asyncio
    async def test_user_scope(self, async_memory):
        scope, scope_id = async_memory._resolve_scope("u1", None, None)
        assert scope == "user"
        assert scope_id == "u1"

    @pytest.mark.asyncio
    async def test_agent_scope_takes_priority(self, async_memory):
        scope, scope_id = async_memory._resolve_scope("u1", "s1", "a1")
        assert scope == "agent"
        assert scope_id == "a1"

    @pytest.mark.asyncio
    async def test_session_scope(self, async_memory):
        scope, scope_id = async_memory._resolve_scope(None, "s1", None)
        assert scope == "session"
        assert scope_id == "s1"

    @pytest.mark.asyncio
    async def test_default_scope(self, async_memory):
        scope, scope_id = async_memory._resolve_scope(None, None, None)
        assert scope == "user"
        assert scope_id == "default"
