"""Dialect detection using a fine-tuned MARBERT/CAMeLBERT classifier."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dhakira.config import ArabicConfig
from dhakira.models import Dialect, DialectResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Mapping from model labels to Dialect enum
_LABEL_MAP: dict[str, Dialect] = {
    "MSA": Dialect.MSA,
    "modern standard arabic": Dialect.MSA,
    "Gulf": Dialect.GULF,
    "gulf": Dialect.GULF,
    "Egyptian": Dialect.EGYPTIAN,
    "egypt": Dialect.EGYPTIAN,
    "Levantine": Dialect.LEVANTINE,
    "levantine": Dialect.LEVANTINE,
    "Maghrebi": Dialect.MAGHREBI,
    "maghrebi": Dialect.MAGHREBI,
}


class DialectDetector:
    """Detects Arabic dialect using a fine-tuned transformer classifier.

    Default model: CAMeL-Lab/bert-base-arabic-camelbert-da (~160M params, CPU-friendly).
    """

    DIALECTS = list(Dialect)

    def __init__(self, config: ArabicConfig | None = None):
        self.config = config or ArabicConfig()
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the classification pipeline."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.config.dialect_model,
                device=-1,  # CPU
                truncation=True,
                max_length=512,
            )
            logger.info("Loaded dialect detection model: %s", self.config.dialect_model)
        except Exception as e:
            logger.warning("Failed to load dialect model: %s. Falling back to MSA.", e)
            self._pipeline = None

    def detect(self, text: str) -> DialectResult:
        """Detect the dialect of the given Arabic text.

        Args:
            text: Arabic text to classify.

        Returns:
            DialectResult with dialect label and confidence score.
        """
        if not self.config.detect_dialect:
            return DialectResult(dialect=Dialect.MSA, confidence=1.0)

        self._load_pipeline()

        if self._pipeline is None:
            return DialectResult(dialect=Dialect.MSA, confidence=0.0)

        try:
            result = self._pipeline(text[:512])[0]
            label = result["label"]
            score = result["score"]

            dialect = _LABEL_MAP.get(label, Dialect.UNKNOWN)
            return DialectResult(dialect=dialect, confidence=score)
        except Exception as e:
            logger.warning("Dialect detection failed: %s", e)
            return DialectResult(dialect=Dialect.MSA, confidence=0.0)

    def detect_batch(self, texts: list[str]) -> list[DialectResult]:
        """Batch dialect detection for efficiency.

        Args:
            texts: List of Arabic texts to classify.

        Returns:
            List of DialectResult objects.
        """
        if not self.config.detect_dialect:
            return [DialectResult(dialect=Dialect.MSA, confidence=1.0) for _ in texts]

        self._load_pipeline()

        if self._pipeline is None:
            return [DialectResult(dialect=Dialect.MSA, confidence=0.0) for _ in texts]

        try:
            truncated = [t[:512] for t in texts]
            results = self._pipeline(truncated)
            dialect_results = []
            for result in results:
                label = result["label"]
                score = result["score"]
                dialect = _LABEL_MAP.get(label, Dialect.UNKNOWN)
                dialect_results.append(DialectResult(dialect=dialect, confidence=score))
            return dialect_results
        except Exception as e:
            logger.warning("Batch dialect detection failed: %s", e)
            return [DialectResult(dialect=Dialect.MSA, confidence=0.0) for _ in texts]
