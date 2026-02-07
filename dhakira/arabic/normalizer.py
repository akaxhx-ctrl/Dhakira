"""Dialect-aware Arabic normalization pipeline."""

from __future__ import annotations

from dhakira.arabic.utils import (
    normalize_alif,
    normalize_numerals,
    normalize_punctuation,
    normalize_taa_marbuta,
    normalize_whitespace,
    normalize_yaa,
    remove_diacritics,
    remove_tatweel,
    unicode_normalize,
)
from dhakira.config import ArabicConfig


class ArabicNormalizer:
    """Dialect-aware Arabic text normalization pipeline.

    Applies a series of normalization steps that can reduce token count
    by ~14-18% while preserving semantic meaning. Dialect-specific rules
    are applied when a dialect is provided.
    """

    def __init__(self, config: ArabicConfig | None = None):
        self.config = config or ArabicConfig()

    def normalize(self, text: str, dialect: str | None = None) -> str:
        """Apply the full normalization pipeline.

        Args:
            text: Input Arabic text.
            dialect: Optional dialect label for dialect-aware normalization.

        Returns:
            Normalized text.
        """
        if not text:
            return text

        # Step 1: NFKC Unicode normalization (always)
        text = unicode_normalize(text)

        # Step 2: Alif unification
        text = normalize_alif(text, preserve_variants=self.config.preserve_alif_variants)

        # Step 3: Taa marbuta handling (skip for Egyptian where distinction matters)
        if self.config.normalize_taa_marbuta and dialect != "Egyptian":
            text = normalize_taa_marbuta(text)

        # Step 4: Yaa normalization (skip for Maghrebi where alif maksura is common)
        if self.config.normalize_yaa and dialect != "Maghrebi":
            text = normalize_yaa(text)

        # Step 5: Arabic-Indic numerals → Western
        if self.config.normalize_numerals:
            text = normalize_numerals(text)

        # Step 6: Punctuation standardization
        if self.config.normalize_punctuation:
            text = normalize_punctuation(text)

        # Step 7: Tatweel/kashida removal
        if self.config.remove_tatweel:
            text = remove_tatweel(text)

        # Step 8: Diacritics handling
        if self.config.remove_diacritics:
            text = remove_diacritics(text)

        # Step 9: Whitespace normalization (always)
        text = normalize_whitespace(text)

        return text

    def normalize_for_embedding(self, text: str, dialect: str | None = None) -> str:
        """Aggressive normalization for embedding (max token compression).

        Applies all normalization steps regardless of config for maximum
        token reduction before embedding.
        """
        if not text:
            return text

        text = unicode_normalize(text)
        text = normalize_alif(text, preserve_variants=False)
        text = normalize_taa_marbuta(text)
        text = normalize_yaa(text)
        text = normalize_numerals(text)
        text = normalize_punctuation(text)
        text = remove_tatweel(text)
        text = remove_diacritics(text)
        text = normalize_whitespace(text)

        return text

    def normalize_for_storage(self, text: str, dialect: str | None = None) -> str:
        """Light normalization for stored text (preserve readability).

        Only applies non-destructive normalizations that don't change
        the visual appearance significantly.
        """
        if not text:
            return text

        text = unicode_normalize(text)
        text = remove_tatweel(text)
        text = normalize_numerals(text)
        text = normalize_whitespace(text)

        return text
