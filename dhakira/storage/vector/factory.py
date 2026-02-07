"""VectorStore factory."""

from __future__ import annotations

from dhakira.config import VectorStoreConfig
from dhakira.storage.base import VectorStore


def create_vector_store(
    config: VectorStoreConfig | None = None,
    embedding_dim: int = 128,
) -> VectorStore:
    """Create a vector store instance based on configuration.

    Args:
        config: Vector store configuration. Defaults to Qdrant in-memory.
        embedding_dim: Embedding vector dimension.

    Returns:
        A VectorStore implementation.

    Raises:
        ValueError: If the provider is not supported.
    """
    config = config or VectorStoreConfig()

    if config.provider == "qdrant":
        from dhakira.storage.vector.qdrant import QdrantVectorStore
        return QdrantVectorStore(config, embedding_dim=embedding_dim)
    else:
        raise ValueError(
            f"Unsupported vector store provider: {config.provider}. Supported: qdrant"
        )
