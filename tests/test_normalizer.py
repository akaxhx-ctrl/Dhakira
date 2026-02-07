"""Tests for Arabic normalization pipeline."""

import pytest

from dhakira.arabic.normalizer import ArabicNormalizer
from dhakira.config import ArabicConfig


@pytest.fixture
def normalizer():
    return ArabicNormalizer()


@pytest.fixture
def normalizer_preserve_alif():
    return ArabicNormalizer(ArabicConfig(preserve_alif_variants=True))


@pytest.fixture
def normalizer_no_diacritics_removal():
    return ArabicNormalizer(ArabicConfig(remove_diacritics=False))


class TestAlifNormalization:
    def test_alif_hamza_above(self, normalizer):
        assert normalizer.normalize("أحمد") == "احمد"

    def test_alif_hamza_below(self, normalizer):
        assert normalizer.normalize("إسلام") == "اسلام"

    def test_alif_madda(self, normalizer):
        assert normalizer.normalize("آمنة") == "امنه"

    def test_alif_wasla(self, normalizer):
        assert normalizer.normalize("ٱلكتاب") == "الكتاب"

    def test_preserve_alif_variants(self, normalizer_preserve_alif):
        assert normalizer_preserve_alif.normalize("أحمد") == "أحمد"


class TestDiacritics:
    def test_remove_fatha(self, normalizer):
        assert normalizer.normalize("كَتَبَ") == "كتب"

    def test_remove_shadda(self, normalizer):
        assert normalizer.normalize("محمَّد") == "محمد"

    def test_remove_tanwin(self, normalizer):
        assert normalizer.normalize("كتابًا") == "كتابا"

    def test_preserve_diacritics_when_configured(self, normalizer_no_diacritics_removal):
        result = normalizer_no_diacritics_removal.normalize("كَتَبَ")
        assert "َ" in result


class TestTatweel:
    def test_remove_tatweel(self, normalizer):
        assert normalizer.normalize("العـــربية") == "العربيه"

    def test_multiple_tatweels(self, normalizer):
        assert normalizer.normalize("مـرحـبـا") == "مرحبا"


class TestTaaMarbuta:
    def test_taa_marbuta_to_haa(self, normalizer):
        assert normalizer.normalize("مدرسة") == "مدرسه"

    def test_skip_for_egyptian(self, normalizer):
        result = normalizer.normalize("مدرسة", dialect="Egyptian")
        assert "ة" in result

    def test_no_skip_for_msa(self, normalizer):
        result = normalizer.normalize("مدرسة", dialect="MSA")
        assert "ه" in result


class TestYaaNormalization:
    def test_alif_maksura_to_yaa(self, normalizer):
        assert normalizer.normalize("على") == "علي"

    def test_skip_for_maghrebi(self, normalizer):
        result = normalizer.normalize("على", dialect="Maghrebi")
        assert "ى" in result


class TestNumerals:
    def test_arabic_indic_numerals(self, normalizer):
        assert normalizer.normalize("١٢٣٤٥") == "12345"

    def test_extended_arabic_indic(self, normalizer):
        assert normalizer.normalize("۰۱۲۳") == "0123"

    def test_mixed_numerals(self, normalizer):
        assert "123" in normalizer.normalize("العدد ١٢٣")


class TestPunctuation:
    def test_arabic_comma(self, normalizer):
        assert normalizer.normalize("أحمد، علي") == "احمد, علي"

    def test_arabic_question_mark(self, normalizer):
        assert normalizer.normalize("كيف حالك؟") == "كيف حالك?"

    def test_arabic_semicolon(self, normalizer):
        assert normalizer.normalize("قال؛ ذهب") == "قال; ذهب"


class TestWhitespace:
    def test_multiple_spaces(self, normalizer):
        assert normalizer.normalize("كلمة   كلمة") == "كلمه كلمه"

    def test_leading_trailing(self, normalizer):
        result = normalizer.normalize("  مرحبا  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestEmptyInput:
    def test_empty_string(self, normalizer):
        assert normalizer.normalize("") == ""

    def test_none_like(self, normalizer):
        assert normalizer.normalize("") == ""


class TestMixedContent:
    def test_arabic_english_mixed(self, normalizer):
        result = normalizer.normalize("Hello أحمد World")
        assert "Hello" in result
        assert "احمد" in result

    def test_urls_preserved(self, normalizer):
        result = normalizer.normalize("زر الموقع http://example.com")
        assert "http://example.com" in result

    def test_numbers_in_text(self, normalizer):
        result = normalizer.normalize("عمره ٢٥ سنة")
        assert "25" in result


class TestNormalizeForEmbedding:
    def test_aggressive_normalization(self):
        normalizer = ArabicNormalizer(ArabicConfig(preserve_alif_variants=True))
        result = normalizer.normalize_for_embedding("أحمد في المدرسة")
        # Should normalize alif despite config saying preserve
        assert "ا" in result
        assert "أ" not in result


class TestNormalizeForStorage:
    def test_light_normalization(self):
        normalizer = ArabicNormalizer()
        result = normalizer.normalize_for_storage("أحمد في المدرسة")
        # Should preserve alif variants
        assert "أ" in result
        # But still remove tatweel
        assert "ـ" not in normalizer.normalize_for_storage("العـربية")
