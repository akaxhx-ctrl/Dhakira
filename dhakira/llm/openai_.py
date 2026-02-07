"""OpenAI LLM provider."""

from __future__ import annotations

import json
import logging

from dhakira.config import LLMConfig
from dhakira.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI-compatible LLM provider.

    Default: gpt-4.1-nano ($0.10/M input tokens — cheapest capable model).
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if response.usage:
            self._track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)

        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system: str | None = None,
    ) -> dict:
        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )

        if response.usage:
            self._track_usage(response.usage.prompt_tokens, response.usage.completion_tokens)

        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured LLM response: %s", content[:200])
            return {}
