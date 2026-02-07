"""Mem0 benchmark runner.

Simulates Mem0's memory pipeline for fair comparison.
In token-counting mode: intercepts OpenAI client calls to count tokens without
making real API calls. No Arabic preprocessing is applied (this is what we're
comparing against).
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
    Timer,
)
from benchmarks.runners.base import BaseRunner
from benchmarks.token_counter import count_tokens, estimate_cost

logger = logging.getLogger(__name__)


class Mem0TokenCountingRunner(BaseRunner):
    """Simulates Mem0's pipeline by counting tokens without API calls.

    Mem0's add pipeline (per conversation):
      1. LLM call to extract facts (using the full conversation text, no Arabic normalization)
      2. For each extracted fact: embed via OpenAI text-embedding-3-small
      3. LLM call to decide update/add/delete (similar to AUDN but always calls LLM)

    Mem0's search pipeline:
      1. Embed query via OpenAI
      2. Vector search
      (No BM25, no graph, no reranker)
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.cost = CostMetrics()
        self.latency = LatencyMetrics()
        # Store facts for search simulation
        self._stored_facts: list[str] = []

    async def setup(self) -> None:
        pass

    async def add_conversations(self, conversations: list[Conversation]) -> None:
        for conv in conversations:
            with Timer() as t:
                await self._simulate_mem0_add(conv)
            self.latency.add_times.append(t.elapsed)

    async def _simulate_mem0_add(self, conv: Conversation) -> None:
        """Simulate Mem0's add pipeline with token counting.

        Mem0 sends the full conversation to the LLM for extraction,
        then for each fact it calls the LLM again for the update decision,
        and embeds each fact via OpenAI embeddings.

        Mem0 uses verbose prompts since it has no Arabic-specific optimizations.
        The raw Arabic text (with diacritics, tatweel, etc.) is sent as-is.
        """
        # Step 1: Build conversation text (NO Arabic normalization — this is the key difference)
        content = "\n".join(f"{m['role']}: {m['content']}" for m in conv.messages)

        # Step 2: Fact extraction LLM call
        # Mem0 uses a detailed system prompt + full conversation
        extraction_system = (
            "You are a Personal Memory Manager for AI assistants. Your task is to extract "
            "and manage important information from conversations between a human user and an "
            "AI assistant. You will be provided with the conversation and any existing memories. "
            "Your job is to:\n"
            "1. Identify new information worth remembering\n"
            "2. Update existing memories if new information modifies them\n"
            "3. Flag any contradictions between new information and existing memories\n\n"
            "Guidelines:\n"
            "- Extract only meaningful, personal information (facts, preferences, habits, etc.)\n"
            "- Be concise but complete in your extractions\n"
            "- Do not extract trivial information or chitchat\n"
            "- Return valid JSON with the specified format"
        )
        extraction_prompt = (
            "Analyze the following conversation and extract important information.\n\n"
            f"Conversation:\n{content}\n\n"
            "Existing memories: None\n\n"
            'Return JSON format: {"facts": [{"text": "...", "category": "fact|preference|event"}]}'
        )

        input_tokens = count_tokens(extraction_system + extraction_prompt, self.config.llm_model)
        # Simulate extraction output (~2-3 facts per conversation, using actual user content)
        simulated_facts = [
            {"text": m["content"][:150], "category": "fact"}
            for m in conv.messages
            if m["role"] == "user"
        ][:3]
        output_text = json.dumps({"facts": simulated_facts}, ensure_ascii=False)
        output_tokens = count_tokens(output_text, self.config.llm_model)
        self.cost.add_llm_call(input_tokens, output_tokens)

        # Step 3: For each fact — embed + LLM update decision
        # Mem0 always calls LLM for update decision (no similarity threshold skip like Dhakira)
        for fact in simulated_facts:
            fact_text = fact["text"]

            # Embedding call (OpenAI API — costs money, unlike Dhakira's local embeddings)
            emb_tokens = count_tokens(fact_text, self.config.embedding_model)
            self.cost.add_embedding_call(emb_tokens)

            # LLM call for update/add/delete decision
            update_system = (
                "You are a memory management system. Decide what action to take with a new piece "
                "of information given existing memories. Actions: ADD (new information), UPDATE "
                "(modifies existing), DELETE (contradicts existing), NOOP (already known).\n"
                "Return valid JSON with the action and reasoning."
            )
            update_prompt = (
                f"New information: {fact_text}\n\n"
                "Existing memories:\n(none)\n\n"
                'Return JSON: {"action": "ADD|UPDATE|DELETE|NOOP", "reason": "..."}'
            )
            update_input = count_tokens(update_system + update_prompt, self.config.llm_model)
            update_output = count_tokens('{"action": "ADD", "reason": "new information"}', self.config.llm_model)
            self.cost.add_llm_call(update_input, update_output)

            self._stored_facts.append(fact_text)

    async def search(self, query: str, user_id: str) -> list[str]:
        with Timer() as t:
            results = await self._simulate_mem0_search(query)
        self.latency.search_times.append(t.elapsed)
        return results

    async def _simulate_mem0_search(self, query: str) -> list[str]:
        """Simulate Mem0's search pipeline.

        Mem0 search:
        1. Embed query via OpenAI
        2. Vector similarity search (no BM25, no graph)
        """
        # Embedding call for query
        emb_tokens = count_tokens(query, self.config.embedding_model)
        self.cost.add_embedding_call(emb_tokens)

        # Return stored facts (in real mode, this would be vector search)
        # For token-counting mode, we return all stored facts to simulate results
        return self._stored_facts[:10]

    def get_result(self) -> BenchmarkResult:
        # Compute total cost: LLM + embedding calls
        llm_cost = estimate_cost(
            input_tokens=self.cost.total_input_tokens,
            output_tokens=self.cost.total_output_tokens,
            model=self.config.llm_model,
        )
        embedding_cost = estimate_cost(
            input_tokens=self.cost.total_embedding_tokens,
            model=self.config.embedding_model,
        )
        self.cost.estimated_cost_usd = llm_cost + embedding_cost

        return BenchmarkResult(
            system_name="Mem0",
            cost=self.cost,
            quality=None,  # Filled in by run_benchmark
            normalization=None,
            latency=self.latency,
        )

    async def teardown(self) -> None:
        self._stored_facts.clear()


class Mem0RealAPIRunner(BaseRunner):
    """Mem0 runner that makes real API calls.

    Requires: OPENAI_API_KEY environment variable and mem0ai package.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.cost = CostMetrics()
        self.latency = LatencyMetrics()
        self.mem0 = None

    async def setup(self) -> None:
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError("mem0ai package required for real-api mode. Install: pip install mem0ai")

        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {"model": self.config.llm_model, "temperature": 0.0},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": self.config.embedding_model},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"collection_name": "mem0_benchmark", "path": ":memory:"},
            },
        }
        self.mem0 = Memory.from_config(mem0_config)

    async def add_conversations(self, conversations: list[Conversation]) -> None:
        for conv in conversations:
            content = "\n".join(f"{m['role']}: {m['content']}" for m in conv.messages)
            with Timer() as t:
                self.mem0.add(content, user_id=self.config.user_id)
            self.latency.add_times.append(t.elapsed)

    async def search(self, query: str, user_id: str) -> list[str]:
        with Timer() as t:
            results = self.mem0.search(query, user_id=user_id)
        self.latency.search_times.append(t.elapsed)
        return [r.get("memory", r.get("text", "")) for r in results.get("results", [])]

    def get_result(self) -> BenchmarkResult:
        # In real API mode, we can't precisely track tokens without OpenAI's response headers
        # The cost here is approximate
        return BenchmarkResult(
            system_name="Mem0",
            cost=self.cost,
            quality=None,
            normalization=None,
            latency=self.latency,
        )

    async def teardown(self) -> None:
        self.mem0 = None
