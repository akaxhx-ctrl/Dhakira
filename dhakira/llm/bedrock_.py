"""AWS Bedrock LLM provider."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dhakira.config import LLMConfig
from dhakira.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class BedrockLLM(BaseLLM):
    """AWS Bedrock LLM provider using the Converse API."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(
            provider="bedrock",
            model=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        )
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _get_client(self):
        if self._client is None:
            import boto3

            region = self.config.base_url or os.getenv("AWS_REGION", "us-east-1")
            self._client = boto3.client("bedrock-runtime", region_name=region)
        return self._client

    async def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._get_client()

        kwargs = {
            "modelId": self.config.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = await asyncio.to_thread(client.converse, **kwargs)

        usage = response.get("usage", {})
        self._track_usage(
            usage.get("inputTokens", 0),
            usage.get("outputTokens", 0),
        )

        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        if content and "text" in content[0]:
            return content[0]["text"]
        return ""

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
            logger.warning("Failed to parse structured Bedrock response: %s", result[:200])
            return {}
