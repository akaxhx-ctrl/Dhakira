"""Fact extraction from conversations using bilingual prompts."""

from __future__ import annotations

import logging

from dhakira.arabic.normalizer import ArabicNormalizer
from dhakira.extraction.prompts import FACT_EXTRACTION_PROMPT, FACT_EXTRACTION_SYSTEM
from dhakira.llm.base import BaseLLM
from dhakira.models import Fact, FactCategory, Message

logger = logging.getLogger(__name__)


class FactExtractor:
    """Extract memorable facts from conversations using a nano LLM.

    Uses bilingual prompts (English instructions + Arabic content) to
    minimize token costs while extracting structured facts.
    """

    def __init__(self, llm: BaseLLM, normalizer: ArabicNormalizer | None = None):
        self.llm = llm
        self.normalizer = normalizer or ArabicNormalizer()

    async def extract(
        self,
        messages: list[Message],
        context: str | None = None,
    ) -> list[Fact]:
        """Extract facts from a conversation.

        Args:
            messages: Conversation messages.
            context: Optional additional context.

        Returns:
            List of extracted Fact objects.
        """
        if not messages:
            return []

        # Build conversation content with normalized Arabic
        content_parts = []
        for msg in messages:
            normalized = self.normalizer.normalize(msg.content)
            content_parts.append(f"{msg.role}: {normalized}")

        content = "\n".join(content_parts)
        if context:
            content = f"Context: {context}\n\n{content}"

        prompt = FACT_EXTRACTION_PROMPT.format(content=content)

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema={"type": "object", "properties": {"facts": {"type": "array"}}},
                system=FACT_EXTRACTION_SYSTEM,
            )
        except Exception as e:
            logger.error("Fact extraction failed: %s", e)
            return []

        return self._parse_facts(result, content)

    def _parse_facts(self, result: dict, source_text: str) -> list[Fact]:
        """Parse LLM response into Fact objects."""
        facts = []
        raw_facts = result.get("facts", [])

        for raw in raw_facts:
            if not isinstance(raw, dict):
                continue

            text = raw.get("text", "").strip()
            if not text:
                continue

            category_str = raw.get("category", "fact")
            try:
                category = FactCategory(category_str)
            except ValueError:
                category = FactCategory.FACT

            confidence = raw.get("confidence", 0.8)
            if not isinstance(confidence, (int, float)):
                confidence = 0.8
            confidence = max(0.0, min(1.0, float(confidence)))

            facts.append(Fact(
                text=text,
                category=category,
                confidence=confidence,
                source_text=source_text[:500],
            ))

        return facts
