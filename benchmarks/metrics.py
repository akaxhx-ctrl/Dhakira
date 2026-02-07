"""Metrics dataclasses and quality computation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CostMetrics:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_embedding_tokens: int = 0
    llm_calls: int = 0
    embedding_calls: int = 0
    estimated_cost_usd: float = 0.0

    def add_llm_call(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.llm_calls += 1

    def add_embedding_call(self, tokens: int) -> None:
        self.total_embedding_tokens += tokens
        self.embedding_calls += 1


@dataclass
class QualityMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank


@dataclass
class NormalizationMetrics:
    original_tokens: int = 0
    normalized_tokens: int = 0

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.normalized_tokens / self.original_tokens) * 100


@dataclass
class LatencyMetrics:
    add_times: list[float] = field(default_factory=list)
    search_times: list[float] = field(default_factory=list)

    @property
    def avg_add_ms(self) -> float:
        return (sum(self.add_times) / len(self.add_times) * 1000) if self.add_times else 0.0

    @property
    def avg_search_ms(self) -> float:
        return (sum(self.search_times) / len(self.search_times) * 1000) if self.search_times else 0.0


@dataclass
class BenchmarkResult:
    system_name: str
    cost: CostMetrics = field(default_factory=CostMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    normalization: NormalizationMetrics | None = None
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def _word_overlap(text_a: str, text_b: str) -> float:
    """Compute word overlap ratio between two texts."""
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / min(len(words_a), len(words_b))


def _is_match(result_text: str, conversation_text: str, threshold: float = 0.6) -> bool:
    """Check if a search result matches a conversation via substring or word overlap.

    A result is considered a match if:
    - The result text is a substring of the conversation text, OR
    - The conversation text is a substring of the result text, OR
    - Word overlap >= threshold (default 60%)
    """
    result_text = result_text.strip()
    conversation_text = conversation_text.strip()

    # Substring containment
    if result_text in conversation_text or conversation_text in result_text:
        return True

    # Word overlap
    return _word_overlap(result_text, conversation_text) >= threshold


def compute_quality(
    search_results: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    all_conversation_texts: dict[str, str],
) -> QualityMetrics:
    """Compute precision, recall, F1, and MRR.

    Args:
        search_results: Mapping query_id -> list of result texts from the system.
        ground_truth: Mapping query_id -> list of conversation IDs that should match.
        all_conversation_texts: Mapping conversation_id -> full concatenated text.

    Returns:
        QualityMetrics with averaged scores.
    """
    precisions = []
    recalls = []
    reciprocal_ranks = []

    for query_id, result_texts in search_results.items():
        gt_conv_ids = ground_truth.get(query_id, [])
        if not gt_conv_ids:
            continue

        # Build ground truth text set
        gt_texts = [all_conversation_texts[cid] for cid in gt_conv_ids if cid in all_conversation_texts]

        # For each result, check if it matches any ground truth conversation
        matched_gt = set()
        first_relevant_rank = 0

        for rank, result_text in enumerate(result_texts, 1):
            is_relevant = False
            for gt_cid, gt_text in zip(gt_conv_ids, gt_texts):
                if _is_match(result_text, gt_text):
                    matched_gt.add(gt_cid)
                    is_relevant = True

            if is_relevant and first_relevant_rank == 0:
                first_relevant_rank = rank

        # Precision: how many returned results are relevant
        n_relevant_returned = len(matched_gt)
        n_returned = len(result_texts)
        precision = n_relevant_returned / n_returned if n_returned > 0 else 0.0

        # Recall: how many ground truth items were found
        recall = n_relevant_returned / len(gt_conv_ids) if gt_conv_ids else 0.0

        # Reciprocal rank
        rr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

    n = len(precisions)
    if n == 0:
        return QualityMetrics()

    avg_p = sum(precisions) / n
    avg_r = sum(recalls) / n
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
    mrr = sum(reciprocal_ranks) / n

    return QualityMetrics(precision=avg_p, recall=avg_r, f1=f1, mrr=mrr)
