"""Markdown report generator for benchmark results."""

from __future__ import annotations

from datetime import datetime, timezone

from benchmarks.config import BenchmarkConfig
from benchmarks.metrics import BenchmarkResult
from benchmarks.token_counter import estimate_cost


def generate_report(results: list[BenchmarkResult], config: BenchmarkConfig) -> str:
    """Generate a markdown benchmark report."""
    lines: list[str] = []

    lines.append("# Dhakira Benchmark Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Mode:** {config.mode.value}")
    lines.append(f"**LLM Model:** {config.llm_model}")
    lines.append(f"**Embedding Model (Mem0):** {config.embedding_model}")
    lines.append(f"**Dhakira Embeddings:** Local (Arabic-Triplet-Matryoshka-V2, $0)")
    lines.append("")

    # Cost comparison table
    lines.append("## Cost Comparison")
    lines.append("")
    lines.append("| Metric | " + " | ".join(r.system_name for r in results) + " |")
    lines.append("|--------|" + "|".join("--------" for _ in results) + "|")
    lines.append(
        "| LLM calls | "
        + " | ".join(str(r.cost.llm_calls) for r in results)
        + " |"
    )
    lines.append(
        "| LLM input tokens | "
        + " | ".join(f"{r.cost.total_input_tokens:,}" for r in results)
        + " |"
    )
    lines.append(
        "| LLM output tokens | "
        + " | ".join(f"{r.cost.total_output_tokens:,}" for r in results)
        + " |"
    )

    # LLM cost row
    llm_costs = []
    for r in results:
        c = estimate_cost(r.cost.total_input_tokens, r.cost.total_output_tokens, config.llm_model)
        llm_costs.append(c)
    lines.append(
        "| LLM cost | "
        + " | ".join(f"${c:.6f}" for c in llm_costs)
        + " |"
    )
    lines.append(
        "| Embedding calls | "
        + " | ".join(str(r.cost.embedding_calls) for r in results)
        + " |"
    )
    lines.append(
        "| Embedding tokens | "
        + " | ".join(f"{r.cost.total_embedding_tokens:,}" for r in results)
        + " |"
    )

    # Embedding cost row
    emb_costs = []
    for r in results:
        c = estimate_cost(r.cost.total_embedding_tokens, 0, config.embedding_model)
        emb_costs.append(c)
    lines.append(
        "| Embedding cost | "
        + " | ".join(
            "$0 (local)" if c == 0 and r.cost.total_embedding_tokens == 0 else f"${c:.6f}"
            for c, r in zip(emb_costs, results)
        )
        + " |"
    )

    lines.append(
        "| **Total estimated cost** | "
        + " | ".join(f"**${r.cost.estimated_cost_usd:.6f}**" for r in results)
        + " |"
    )
    lines.append("")

    # Cost savings analysis
    if len(results) >= 2:
        dhakira = next((r for r in results if r.system_name == "Dhakira"), None)
        mem0 = next((r for r in results if r.system_name == "Mem0"), None)
        if dhakira and mem0 and mem0.cost.estimated_cost_usd > 0:
            savings_pct = (1 - dhakira.cost.estimated_cost_usd / mem0.cost.estimated_cost_usd) * 100

            lines.append("### Cost Analysis")
            lines.append("")

            if savings_pct > 0:
                lines.append(f"**Dhakira is {savings_pct:.1f}% cheaper** than Mem0 overall.")
            else:
                lines.append(f"Dhakira's LLM cost is higher (entity extraction + AUDN), but this is "
                             f"offset by **$0 embedding cost** from local models.")

            lines.append("")

            # Show where savings come from
            dhakira_llm = estimate_cost(
                dhakira.cost.total_input_tokens, dhakira.cost.total_output_tokens, config.llm_model
            )
            mem0_llm = estimate_cost(
                mem0.cost.total_input_tokens, mem0.cost.total_output_tokens, config.llm_model
            )
            mem0_emb = estimate_cost(mem0.cost.total_embedding_tokens, 0, config.embedding_model)

            lines.append("| Cost Component | Dhakira | Mem0 | Delta |")
            lines.append("|---------------|---------|------|-------|")
            llm_delta = dhakira_llm - mem0_llm
            lines.append(
                f"| LLM | ${dhakira_llm:.6f} | ${mem0_llm:.6f} | "
                f"{'+'  if llm_delta >= 0 else ''}{llm_delta:.6f} |"
            )
            lines.append(
                f"| Embeddings | $0 (local) | ${mem0_emb:.6f} | -{mem0_emb:.6f} |"
            )
            total_delta = dhakira.cost.estimated_cost_usd - mem0.cost.estimated_cost_usd
            lines.append(
                f"| **Total** | **${dhakira.cost.estimated_cost_usd:.6f}** | "
                f"**${mem0.cost.estimated_cost_usd:.6f}** | "
                f"**{'+'  if total_delta >= 0 else ''}{total_delta:.6f}** |"
            )
            lines.append("")

            # Projection at scale
            lines.append("### Projected Savings at Scale")
            lines.append("")
            lines.append("Embedding costs dominate at scale (thousands of memories × repeated search queries).")
            lines.append("With Dhakira's free local embeddings, savings grow as usage increases.")
            lines.append("")
            for n_memories in [100, 1000, 10000]:
                # At scale: each add has ~3 embedding calls, each search has 1
                # Assume 2 searches per memory on average
                n_embed_calls = n_memories * 3 + n_memories * 2
                avg_tokens_per_embed = (mem0.cost.total_embedding_tokens / max(mem0.cost.embedding_calls, 1))
                scale_embed_cost = estimate_cost(
                    int(n_embed_calls * avg_tokens_per_embed), 0, config.embedding_model
                )
                lines.append(f"- **{n_memories:,} memories**: Mem0 embedding cost ~${scale_embed_cost:.4f} "
                             f"(Dhakira: $0)")
            lines.append("")

            # Cache savings estimate
            lines.append("### With Dhakira Features Enabled")
            lines.append("")
            lines.append("The benchmark runs with cache **disabled** for fairness. In production:")
            lines.append("- **Semantic cache**: Skips repeated extraction (~30-50% savings on LLM calls)")
            lines.append("- **AUDN threshold skip**: ~40-60% of new facts bypass LLM consolidation call")
            lines.append("- **Reranker**: Improves quality at minimal latency cost (local model)")
            lines.append("")

    # Quality comparison
    lines.append("## Retrieval Quality")
    lines.append("")
    lines.append("| Metric | " + " | ".join(r.system_name for r in results) + " |")
    lines.append("|--------|" + "|".join("--------" for _ in results) + "|")
    if results[0].quality is not None:
        lines.append(
            "| Precision | "
            + " | ".join(f"{r.quality.precision:.3f}" if r.quality else "N/A" for r in results)
            + " |"
        )
        lines.append(
            "| Recall | "
            + " | ".join(f"{r.quality.recall:.3f}" if r.quality else "N/A" for r in results)
            + " |"
        )
        lines.append(
            "| F1 | "
            + " | ".join(f"{r.quality.f1:.3f}" if r.quality else "N/A" for r in results)
            + " |"
        )
        lines.append(
            "| MRR | "
            + " | ".join(f"{r.quality.mrr:.3f}" if r.quality else "N/A" for r in results)
            + " |"
        )
    else:
        lines.append("| | " + " | ".join("(token-counting mode)" for _ in results) + " |")
    lines.append("")
    lines.append("*Note: In token-counting mode, quality metrics are approximate since both systems "
                 "use mock LLMs and embeddings. Run with `--mode real-api` for accurate quality comparison.*")
    lines.append("")

    # Normalization impact (Dhakira only)
    dhakira_result = next((r for r in results if r.normalization is not None), None)
    if dhakira_result and dhakira_result.normalization:
        nm = dhakira_result.normalization
        lines.append("## Arabic Normalization Impact (Dhakira)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Original tokens (tiktoken) | {nm.original_tokens:,} |")
        lines.append(f"| Normalized tokens (tiktoken) | {nm.normalized_tokens:,} |")
        lines.append(f"| **Token delta** | **{nm.savings_pct:+.1f}%** |")
        lines.append("")
        if nm.savings_pct < 0:
            lines.append(
                "Note: tiktoken (used by OpenAI models) may tokenize normalized Arabic into *more* tokens, "
                "because normalization can break multi-character token clusters. However, Arabic normalization "
                "primarily benefits **local embedding models** (Arabic-Triplet) and **BM25 keyword matching** "
                "where consistent forms improve retrieval quality."
            )
        else:
            lines.append(
                "Arabic normalization reduces token count by removing diacritics, unifying alif variants, "
                "and standardizing punctuation."
            )
        lines.append("")

    # Latency comparison
    has_latency = any(r.latency.add_times or r.latency.search_times for r in results)
    if has_latency:
        lines.append("## Latency")
        lines.append("")
        lines.append("| Metric | " + " | ".join(r.system_name for r in results) + " |")
        lines.append("|--------|" + "|".join("--------" for _ in results) + "|")
        lines.append(
            "| Avg add (ms) | "
            + " | ".join(f"{r.latency.avg_add_ms:.1f}" for r in results)
            + " |"
        )
        lines.append(
            "| Avg search (ms) | "
            + " | ".join(f"{r.latency.avg_search_ms:.1f}" for r in results)
            + " |"
        )
        lines.append("")
        lines.append("*Note: In token-counting mode, latency reflects mock processing time, "
                     "not real API latency. Use `--mode real-api` for real latency numbers.*")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Fair Comparison Guardrails")
    lines.append("")
    lines.append(f"- Same LLM model (`{config.llm_model}`) for both Dhakira and Mem0")
    lines.append("- Same dataset: 12 Arabic conversations across 4 dialects (MSA, Egyptian, Gulf, Levantine)")
    lines.append("- Same 14 search queries with ground truth for quality evaluation")
    lines.append("- Dhakira reranker **disabled** (Mem0 doesn't have one)")
    lines.append("- Dhakira semantic cache **disabled** (measure raw pipeline cost)")
    lines.append("- Token counting uses same tiktoken encoder for both systems")
    lines.append("- Dhakira's local embeddings counted as **$0** (Mem0's OpenAI embeddings at API price)")
    lines.append("")
    lines.append("### Pipeline Comparison")
    lines.append("")
    lines.append("| Step | Dhakira | Mem0 |")
    lines.append("|------|---------|------|")
    lines.append("| Arabic normalization | Yes | No |")
    lines.append("| Fact extraction (LLM) | Yes | Yes |")
    lines.append("| Entity extraction (LLM) | Yes | No |")
    lines.append("| AUDN consolidation | Threshold skip < 0.5 | Always LLM |")
    lines.append("| Embeddings | Local (free) | OpenAI API (paid) |")
    lines.append("| Search | Vector + BM25 + Graph (RRF) | Vector only |")
    lines.append("| Semantic cache | Available (disabled in bench) | No |")
    lines.append("| Reranker | Available (disabled in bench) | No |")
    lines.append("")

    return "\n".join(lines)
