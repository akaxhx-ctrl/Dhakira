"""Token counting and cost estimation using tiktoken."""

from __future__ import annotations

import tiktoken


# Price per 1M tokens (USD) — as of 2025
PRICING = {
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "text-embedding-3-small": {"input": 0.02},
    "text-embedding-3-large": {"input": 0.13},
    # Claude via Bedrock / Anthropic API
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
}

# Cache encoders to avoid repeated initialization
_encoder_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoder(model: str) -> tiktoken.Encoding:
    """Get or create a tiktoken encoder for the given model."""
    if model not in _encoder_cache:
        try:
            _encoder_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            _encoder_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoder_cache[model]


def count_tokens(text: str, model: str = "gpt-4.1-nano") -> int:
    """Count tokens for a given text and model."""
    encoder = _get_encoder(model)
    return len(encoder.encode(text))


def count_embedding_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens for an embedding API call."""
    return count_tokens(text, model)


def estimate_cost(
    input_tokens: int,
    output_tokens: int = 0,
    model: str = "gpt-4.1-nano",
) -> float:
    """Estimate cost in USD for given token counts.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens (0 for embeddings).
        model: Model name for pricing lookup.

    Returns:
        Estimated cost in USD.
    """
    pricing = PRICING.get(model)
    if not pricing:
        return 0.0

    cost = (input_tokens / 1_000_000) * pricing["input"]
    if output_tokens > 0 and "output" in pricing:
        cost += (output_tokens / 1_000_000) * pricing["output"]
    return cost
