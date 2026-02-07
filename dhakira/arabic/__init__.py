"""Arabic language processing for Dhakira."""

from dhakira.arabic.chunker import ArabicChunker
from dhakira.arabic.dialect import DialectDetector
from dhakira.arabic.normalizer import ArabicNormalizer

__all__ = ["ArabicNormalizer", "DialectDetector", "ArabicChunker"]
