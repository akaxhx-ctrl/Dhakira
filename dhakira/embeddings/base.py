"""Abstract embeddings interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
