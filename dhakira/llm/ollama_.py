"""Ollama local LLM provider."""

from __future__ import annotations

import json
import logging

from dhakira.config import LLMConfig
from dhakira.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama local LLM provider for fully local operation."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(provider="ollama", model="llama3.2")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from ollama import AsyncClient

            kwargs = {}
            if self.config.base_url:
                kwargs["host"] = self.config.base_url

            self._client = AsyncClient(**kwargs)
        return self._client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat(
            model=self.config.model,
            messages=messages,
            options={"temperature": self.config.temperature},
        )

        return response["message"]["content"]

    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system: str | None = None,
    ) -> dict:
        json_prompt = (
            f"{prompt}\n\n"
            "Respond with valid JSON only, no additional text or formatting."
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
            logger.warning("Failed to parse structured Ollama response: %s", result[:200])
            return {}
