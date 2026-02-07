"""CLI entry point for the Dhakira benchmark suite.

Usage:
    python benchmarks/run_benchmark.py --mode token-counting
    python benchmarks/run_benchmark.py --mode real-api --provider bedrock
    python benchmarks/run_benchmark.py --mode real-api  # requires OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so benchmarks can import dhakira
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env file from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from benchmarks.config import BenchmarkMode, parse_args
from benchmarks.dataset import ALL_CONVERSATIONS, ALL_QUERIES, CONVERSATIONS_BY_ID
from benchmarks.metrics import BenchmarkResult, compute_quality
from benchmarks.report import generate_report
from benchmarks.runners.base import BaseRunner
from benchmarks.runners.dhakira_runner import DhakiraRunner
from benchmarks.runners.mem0_runner import Mem0RealAPIRunner, Mem0TokenCountingRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark")


def _build_conversation_texts() -> dict[str, str]:
    """Build mapping of conversation_id -> full concatenated text."""
    texts = {}
    for conv in ALL_CONVERSATIONS:
        texts[conv.id] = " ".join(m["content"] for m in conv.messages)
    return texts


def _build_ground_truth() -> dict[str, list[str]]:
    """Build mapping of query_id -> ground truth conversation IDs."""
    return {q.id: q.ground_truth_ids for q in ALL_QUERIES}


async def run_single(runner: BaseRunner, config) -> BenchmarkResult:
    """Run a single benchmark runner end-to-end."""
    name = runner.__class__.__name__
    logger.info("Setting up %s...", name)
    await runner.setup()

    logger.info("Adding %d conversations with %s...", len(ALL_CONVERSATIONS), name)
    await runner.add_conversations(ALL_CONVERSATIONS)

    logger.info("Running %d search queries with %s...", len(ALL_QUERIES), name)
    search_results: dict[str, list[str]] = {}
    for query in ALL_QUERIES:
        results = await runner.search(query.text, user_id=config.user_id)
        search_results[query.id] = results

    # Compute quality metrics
    conversation_texts = _build_conversation_texts()
    ground_truth = _build_ground_truth()
    quality = compute_quality(search_results, ground_truth, conversation_texts)

    result = runner.get_result()
    result.quality = quality

    logger.info("Tearing down %s...", name)
    await runner.teardown()

    return result


async def main() -> None:
    config = parse_args()
    logger.info("Benchmark mode: %s", config.mode.value)

    runners: list[BaseRunner] = []

    # Always include Dhakira
    runners.append(DhakiraRunner(config))

    # Mem0 runner depends on mode
    if config.mode == BenchmarkMode.TOKEN_COUNTING:
        runners.append(Mem0TokenCountingRunner(config))
    else:
        import os
        if os.getenv("OPENAI_API_KEY"):
            runners.append(Mem0RealAPIRunner(config))
        else:
            logger.warning("OPENAI_API_KEY not set — skipping Mem0 real-api runner (Mem0 requires OpenAI)")


    # Run benchmarks
    results: list[BenchmarkResult] = []
    for runner in runners:
        result = await run_single(runner, config)
        results.append(result)

    # Generate report
    report = generate_report(results, config)

    # Print to stdout
    print()
    print(report)

    # Save to file
    output_path = config.output_dir / "benchmark_report.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", output_path)

    # Print quick summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r.system_name}:")
        print(f"  LLM calls:      {r.cost.llm_calls}")
        print(f"  Input tokens:    {r.cost.total_input_tokens:,}")
        print(f"  Output tokens:   {r.cost.total_output_tokens:,}")
        print(f"  Embed tokens:    {r.cost.total_embedding_tokens:,}")
        print(f"  Estimated cost:  ${r.cost.estimated_cost_usd:.6f}")
        if r.quality:
            print(f"  Precision:       {r.quality.precision:.3f}")
            print(f"  Recall:          {r.quality.recall:.3f}")
            print(f"  F1:              {r.quality.f1:.3f}")
            print(f"  MRR:             {r.quality.mrr:.3f}")
        if r.normalization:
            print(f"  Normalization:   {r.normalization.savings_pct:.1f}% token savings")


if __name__ == "__main__":
    asyncio.run(main())
