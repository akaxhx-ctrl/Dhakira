"""Embedding model abstraction."""

from dhakira.embeddings.base import BaseEmbeddings
from dhakira.embeddings.factory import create_embeddings

__all__ = ["BaseEmbeddings", "create_embeddings"]
