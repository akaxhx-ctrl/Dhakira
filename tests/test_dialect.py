"""Tests for dialect detection."""

import pytest

from dhakira.arabic.dialect import DialectDetector
from dhakira.config import ArabicConfig
from dhakira.models import Dialect


class TestDialectDetectorDisabled:
    def test_returns_msa_when_disabled(self):
        detector = DialectDetector(ArabicConfig(detect_dialect=False))
        result = detector.detect("أي نص عربي")
        assert result.dialect == Dialect.MSA
        assert result.confidence == 1.0

    def test_batch_returns_msa_when_disabled(self):
        detector = DialectDetector(ArabicConfig(detect_dialect=False))
        results = detector.detect_batch(["نص أول", "نص ثاني"])
        assert len(results) == 2
        assert all(r.dialect == Dialect.MSA for r in results)


class TestDialectDetectorFallback:
    """Test fallback behavior when model is not available."""

    def test_graceful_fallback_on_bad_model(self):
        config = ArabicConfig(detect_dialect=True, dialect_model="nonexistent/model")
        detector = DialectDetector(config)
        result = detector.detect("مرحبا")
        assert result.dialect == Dialect.MSA
        assert result.confidence == 0.0

    def test_batch_graceful_fallback(self):
        config = ArabicConfig(detect_dialect=True, dialect_model="nonexistent/model")
        detector = DialectDetector(config)
        results = detector.detect_batch(["نص أول", "نص ثاني"])
        assert len(results) == 2
        assert all(r.confidence == 0.0 for r in results)


class TestDialectResult:
    def test_dialect_result_fields(self):
        from dhakira.models import DialectResult
        result = DialectResult(dialect=Dialect.EGYPTIAN, confidence=0.95)
        assert result.dialect == Dialect.EGYPTIAN
        assert result.confidence == 0.95

    def test_confidence_bounds(self):
        from dhakira.models import DialectResult
        with pytest.raises(Exception):
            DialectResult(dialect=Dialect.MSA, confidence=1.5)
        with pytest.raises(Exception):
            DialectResult(dialect=Dialect.MSA, confidence=-0.1)
