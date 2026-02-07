"""HuggingFace embeddings provider (GATE, Swan, BGE-M3)."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from dhakira.config import EmbeddingsConfig
from dhakira.embeddings.base import BaseEmbeddings

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddings(BaseEmbeddings):
    """Local HuggingFace embeddings using sentence-transformers.

    Default: Omartificial/Arabic-Triplet-Matryoshka-V2 (135M params, CPU-friendly).
    Supports Matryoshka truncation for reduced dimension embeddings.
    """

    def __init__(self, config: EmbeddingsConfig | None = None):
        self.config = config or EmbeddingsConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.config.model,
            device=self.config.device,
            truncate_dim=self.config.dim,
        )
        logger.info(
            "Loaded embedding model: %s (dim=%d, device=%s)",
            self.config.model,
            self.config.dim,
            self.config.device,
        )

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, partial(self._encode_sync, [text]))
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._encode_sync, texts))

    def get_dimension(self) -> int:
        return self.config.dim
