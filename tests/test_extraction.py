"""Tests for fact extraction (mocked LLM)."""

import pytest

from dhakira.extraction.extractor import FactExtractor
from dhakira.models import Fact, FactCategory, Message


class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self, response: dict | None = None):
        self.response = response or {"facts": []}
        self.calls = []

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(("generate", prompt, system))
        return ""

    async def generate_structured(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        self.calls.append(("generate_structured", prompt, system))
        return self.response


class TestFactExtraction:
    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        mock_llm = MockLLM({
            "facts": [
                {"text": "اسمه أحمد", "category": "fact", "confidence": 0.95},
                {"text": "يحب القهوة العربية", "category": "preference", "confidence": 0.9},
            ]
        })
        extractor = FactExtractor(mock_llm)

        messages = [
            Message(role="user", content="اسمي أحمد وأحب القهوة العربية"),
            Message(role="assistant", content="أهلا أحمد!"),
        ]

        facts = await extractor.extract(messages)
        assert len(facts) == 2
        assert facts[0].text == "اسمه أحمد"
        assert facts[0].category == FactCategory.FACT
        assert facts[1].category == FactCategory.PREFERENCE

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        mock_llm = MockLLM()
        extractor = FactExtractor(mock_llm)
        facts = await extractor.extract([])
        assert facts == []

    @pytest.mark.asyncio
    async def test_with_context(self):
        mock_llm = MockLLM({"facts": [{"text": "يعمل مهندسا", "category": "fact", "confidence": 0.8}]})
        extractor = FactExtractor(mock_llm)

        messages = [Message(role="user", content="أنا مهندس برمجيات")]

        facts = await extractor.extract(messages, context="سياق سابق")
        assert len(facts) == 1
        # Verify context was included in the prompt
        call_prompt = mock_llm.calls[0][1]
        assert "سياق سابق" in call_prompt

    @pytest.mark.asyncio
    async def test_invalid_category_defaults_to_fact(self):
        mock_llm = MockLLM({
            "facts": [{"text": "نص", "category": "invalid_category", "confidence": 0.8}]
        })
        extractor = FactExtractor(mock_llm)
        messages = [Message(role="user", content="نص")]
        facts = await extractor.extract(messages)
        assert facts[0].category == FactCategory.FACT

    @pytest.mark.asyncio
    async def test_confidence_clamping(self):
        mock_llm = MockLLM({
            "facts": [
                {"text": "نص1", "category": "fact", "confidence": 1.5},
                {"text": "نص2", "category": "fact", "confidence": -0.5},
            ]
        })
        extractor = FactExtractor(mock_llm)
        messages = [Message(role="user", content="نص")]
        facts = await extractor.extract(messages)
        assert facts[0].confidence == 1.0
        assert facts[1].confidence == 0.0

    @pytest.mark.asyncio
    async def test_empty_text_facts_skipped(self):
        mock_llm = MockLLM({
            "facts": [
                {"text": "", "category": "fact", "confidence": 0.8},
                {"text": "نص حقيقي", "category": "fact", "confidence": 0.8},
            ]
        })
        extractor = FactExtractor(mock_llm)
        messages = [Message(role="user", content="نص")]
        facts = await extractor.extract(messages)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self):
        class ErrorLLM:
            async def generate(self, *args, **kwargs):
                raise Exception("API Error")

            async def generate_structured(self, *args, **kwargs):
                raise Exception("API Error")

        extractor = FactExtractor(ErrorLLM())
        messages = [Message(role="user", content="نص")]
        facts = await extractor.extract(messages)
        assert facts == []

    @pytest.mark.asyncio
    async def test_bilingual_prompt_structure(self):
        mock_llm = MockLLM({"facts": []})
        extractor = FactExtractor(mock_llm)
        messages = [Message(role="user", content="مرحبا")]
        await extractor.extract(messages)

        # System prompt should be in English (cheaper tokens)
        system_prompt = mock_llm.calls[0][2]
        assert "Extract" in system_prompt
        # User prompt should contain Arabic content
        user_prompt = mock_llm.calls[0][1]
        assert "مرحبا" in user_prompt
