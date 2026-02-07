"""Anthropic LLM provider."""

from __future__ import annotations

import json
import logging

from dhakira.config import LLMConfig
from dhakira.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            kwargs = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url

            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()

        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        self._track_usage(response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text if response.content else ""

    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system: str | None = None,
    ) -> dict:
        json_prompt = (
            f"{prompt}\n\n"
            "Respond with valid JSON only, no markdown formatting or code blocks."
        )

        result = await self.generate(json_prompt, system=system)

        # Strip potential markdown code blocks
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:])
            if result.endswith("```"):
                result = result[:-3]

        try:
            return json.loads(result.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured Anthropic response: %s", result[:200])
            return {}
