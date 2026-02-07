"""Low-level Arabic character helpers."""

from __future__ import annotations

import re
import unicodedata

# Arabic Unicode ranges
ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

TATWEEL = "\u0640"

# Alif variants
ALIF_MADDA = "\u0622"  # آ
ALIF_HAMZA_ABOVE = "\u0623"  # أ
ALIF_HAMZA_BELOW = "\u0625"  # إ
ALIF_WASLA = "\u0671"  # ٱ
ALIF = "\u0627"  # ا

# Taa marbuta and Haa
TAA_MARBUTA = "\u0629"  # ة
HAA = "\u0647"  # ه

# Yaa variants
ALIF_MAKSURA = "\u0649"  # ى
YAA = "\u064A"  # ي

# Arabic-Indic numerals → Western Arabic numerals
ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EXTENDED_ARABIC_INDIC_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Arabic punctuation → standard equivalents
PUNCTUATION_MAP = str.maketrans({
    "\u060C": ",",   # Arabic comma
    "\u061B": ";",   # Arabic semicolon
    "\u061F": "?",   # Arabic question mark
    "\u066B": ".",   # Arabic decimal point
    "\u066C": ",",   # Arabic thousands separator
})

# Arabic sentence-ending punctuation
SENTENCE_ENDINGS = re.compile(r"[.!?؟\u06D4]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟\u06D4])\s+")


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (tashkeel)."""
    return ARABIC_DIACRITICS.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove tatweel/kashida characters."""
    return text.replace(TATWEEL, "")


def normalize_alif(text: str, preserve_variants: bool = False) -> str:
    """Unify alif variants (أ إ آ ٱ → ا)."""
    if preserve_variants:
        return text
    text = text.replace(ALIF_MADDA, ALIF)
    text = text.replace(ALIF_HAMZA_ABOVE, ALIF)
    text = text.replace(ALIF_HAMZA_BELOW, ALIF)
    text = text.replace(ALIF_WASLA, ALIF)
    return text


def normalize_taa_marbuta(text: str) -> str:
    """Normalize taa marbuta to haa (ة → ه)."""
    return text.replace(TAA_MARBUTA, HAA)


def normalize_yaa(text: str) -> str:
    """Normalize alif maksura to yaa (ى → ي)."""
    return text.replace(ALIF_MAKSURA, YAA)


def normalize_numerals(text: str) -> str:
    """Convert Arabic-Indic numerals to Western Arabic numerals."""
    text = text.translate(ARABIC_INDIC_MAP)
    text = text.translate(EXTENDED_ARABIC_INDIC_MAP)
    return text


def normalize_punctuation(text: str) -> str:
    """Standardize Arabic punctuation to ASCII equivalents."""
    return text.translate(PUNCTUATION_MAP)


def unicode_normalize(text: str) -> str:
    """Apply NFKC Unicode normalization."""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))


def arabic_token_count(text: str) -> int:
    """Approximate token count for Arabic text (rough heuristic: ~1.5 tokens per word)."""
    words = text.split()
    arabic_words = sum(1 for w in words if is_arabic(w))
    non_arabic_words = len(words) - arabic_words
    return int(arabic_words * 1.5 + non_arabic_words)
