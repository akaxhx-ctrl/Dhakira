"""Dhakira benchmark runner.

Tracks token usage (with normalization savings) and retrieval quality.
In token-counting mode, uses a mock LLM that counts tokens without calling APIs.
Dhakira uses local embeddings (free) — so embedding cost is $0.
"""

from __future__ import annotations

import json
import logging

from benchmarks.config import BenchmarkConfig, BenchmarkMode
from benchmarks.dataset import Conversation
from benchmarks.metrics import (
    BenchmarkResult,
    CostMetrics,
    LatencyMetrics,
    NormalizationMetrics,
    Timer,
)
from benchmarks.runners.base import BaseRunner
from benchmarks.token_counter import count_tokens, estimate_cost

logger = logging.getLogger(__name__)


class TokenCountingLLM:
    """Mock LLM that counts tokens without making API calls.

    Simulates Dhakira's LLM usage: fact extraction + entity extraction + AUDN decisions.
    Tracks input/output tokens for cost estimation.
    Extracts realistic facts from conversation content for meaningful quality evaluation.
    """

    def __init__(self, model: str, cost_metrics: CostMetrics):
        self.model = model
        self.cost_metrics = cost_metrics

    def _extract_user_lines(self, prompt: str) -> list[str]:
        """Extract user message content from the prompt for realistic mock facts."""
        lines = []
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("user:"):
                content = stripped[len("user:"):].strip()
                if content and len(content) > 20:
                    lines.append(content)
        return lines

    async def generate(self, prompt: str, system: str | None = None) -> str:
        input_tokens = count_tokens(prompt, self.model)
        if system:
            input_tokens += count_tokens(system, self.model)
        output_tokens = max(50, input_tokens // 3)
        self.cost_metrics.add_llm_call(input_tokens, output_tokens)
        return ""

    async def generate_structured(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        input_tokens = count_tokens(prompt, self.model)
        if system:
            input_tokens += count_tokens(system, self.model)

        system_lower = (system or "").lower()

        if "memory extraction" in system_lower or "extract key facts" in system_lower:
            # Extract real user content from the prompt for meaningful facts
            user_lines = self._extract_user_lines(prompt)
            facts = []
            for line in user_lines[:3]:
                # Use each user message as a fact (truncated to ~100 chars)
                fact_text = line[:150] if len(line) > 150 else line
                facts.append({"text": fact_text, "category": "fact", "confidence": 0.9})
            if not facts:
                facts = [{"text": "حقيقة مستخرجة", "category": "fact", "confidence": 0.8}]
            response = {"facts": facts}
        elif "entity" in system_lower and "relationship" in system_lower:
            response = {
                "entities": [{"name": "شخص", "type": "person", "summary": "مستخدم"}],
                "relationships": [],
            }
        elif "memory consolidation" in system_lower:
            response = {"action": "ADD", "reason": "new fact"}
        else:
            response = {}

        output_tokens = count_tokens(json.dumps(response, ensure_ascii=False), self.model)
        self.cost_metrics.add_llm_call(input_tokens, output_tokens)
        return response


class MockEmbeddings:
    """Mock embeddings that return deterministic vectors without loading a model.

    Used in token-counting mode to avoid downloading HuggingFace models.
    Returns MD5-based deterministic vectors (same approach as test suite).
    """

    def __init__(self, dim: int = 128):
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        base = [int(c, 16) / 15.0 for c in h]
        vec = (base * (self._dim // len(base) + 1))[:self._dim]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    def get_dimension(self) -> int:
        return self._dim


class DhakiraRunner(BaseRunner):
    """Benchmark runner for Dhakira."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.cost = CostMetrics()
        self.normalization = NormalizationMetrics()
        self.latency = LatencyMetrics()
        self.memory = None

    async def setup(self) -> None:
        from dhakira.async_memory import AsyncMemory
        from dhakira.config import (
            CacheConfig,
            DhakiraConfig,
            LLMConfig,
            RerankerConfig,
            RetrievalConfig,
        )

        llm_kwargs = {
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
        }
        if self.config.llm_provider == "bedrock":
            import os
            llm_kwargs["base_url"] = os.getenv("AWS_REGION", "us-east-1")
            llm_kwargs["max_tokens"] = 4096  # Claude entity extraction needs more space

        dhakira_config = DhakiraConfig(
            llm=LLMConfig(**llm_kwargs),
            retrieval=RetrievalConfig(
                reranker=RerankerConfig(enabled=False),  # Fair comparison — Mem0 has no reranker
            ),
            cache=CacheConfig(enabled=False),  # Disable cache — measure raw pipeline cost
        )

        self.memory = AsyncMemory(dhakira_config)

        if self.config.mode == BenchmarkMode.TOKEN_COUNTING:
            # Replace LLM with token-counting mock
            mock_llm = TokenCountingLLM(self.config.llm_model, self.cost)
            self.memory.llm = mock_llm
            self.memory.fact_extractor.llm = mock_llm
            self.memory.entity_extractor.llm = mock_llm
            self.memory.audn.llm = mock_llm

            # Replace embeddings with mock (avoid downloading HuggingFace model)
            mock_embed = MockEmbeddings(dim=dhakira_config.embeddings.dim)
            self.memory.embeddings = mock_embed
            self.memory.searcher.embeddings = mock_embed

        # Set dedup threshold very high so it doesn't interfere
        self.memory.dedup.threshold = 0.999

    async def add_conversations(self, conversations: list[Conversation]) -> None:
        for conv in conversations:
            # Track normalization savings
            for msg in conv.messages:
                original_text = msg["content"]
                normalized_text = self.memory.normalizer.normalize_for_embedding(original_text)
                original_tokens = count_tokens(original_text, self.config.llm_model)
                normalized_tokens = count_tokens(normalized_text, self.config.llm_model)
                self.normalization.original_tokens += original_tokens
                self.normalization.normalized_tokens += normalized_tokens

            with Timer() as t:
                await self.memory.add(
                    messages=conv.messages,
                    user_id=self.config.user_id,
                )
            self.latency.add_times.append(t.elapsed)

        # In real-api mode, pull actual token usage from the LLM provider
        if self.config.mode == BenchmarkMode.REAL_API:
            llm = self.memory.llm
            self.cost.total_input_tokens = llm.total_input_tokens
            self.cost.total_output_tokens = llm.total_output_tokens
            self.cost.llm_calls = llm.call_count

        # Local embeddings are free — $0 cost.
        # Embedding calls are tracked but not charged.
        logger.info(
            "Dhakira: %d LLM calls, %d input tokens, %d output tokens, %d embedding calls (local, $0)",
            self.cost.llm_calls,
            self.cost.total_input_tokens,
            self.cost.total_output_tokens,
            self.cost.embedding_calls,
        )

    async def search(self, query: str, user_id: str) -> list[str]:
        with Timer() as t:
            results = await self.memory.search(query=query, user_id=user_id, limit=10)
        self.latency.search_times.append(t.elapsed)
        return [r.text for r in results]

    def get_result(self) -> BenchmarkResult:
        # Compute cost: only LLM calls cost money (local embeddings are free)
        self.cost.estimated_cost_usd = estimate_cost(
            input_tokens=self.cost.total_input_tokens,
            output_tokens=self.cost.total_output_tokens,
            model=self.config.llm_model,
        )
        return BenchmarkResult(
            system_name="Dhakira",
            cost=self.cost,
            quality=None,  # Filled in by run_benchmark
            normalization=self.normalization,
            latency=self.latency,
        )

    async def teardown(self) -> None:
        self.memory = None
