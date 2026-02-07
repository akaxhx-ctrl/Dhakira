"""Embeddings provider factory."""

from __future__ import annotations

from dhakira.config import EmbeddingsConfig
from dhakira.embeddings.base import BaseEmbeddings


def create_embeddings(config: EmbeddingsConfig | None = None) -> BaseEmbeddings:
    """Create an embeddings provider instance based on configuration.

    Args:
        config: Embeddings configuration. Defaults to HuggingFace GATE model.

    Returns:
        A BaseEmbeddings implementation.

    Raises:
        ValueError: If the provider is not supported.
    """
    config = config or EmbeddingsConfig()

    if config.provider == "huggingface":
        from dhakira.embeddings.huggingface_ import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(config)
    else:
        raise ValueError(
            f"Unsupported embeddings provider: {config.provider}. Supported: huggingface"
        )
