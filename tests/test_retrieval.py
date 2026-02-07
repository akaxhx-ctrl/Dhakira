"""Tests for hybrid search and retrieval components."""

import pytest

from dhakira.models import MemoryRecord, SearchResult
from dhakira.retrieval.bm25 import ArabicBM25, arabic_tokenize


class TestArabicTokenizer:
    def test_basic_tokenization(self):
        tokens = arabic_tokenize("أحمد يحب القهوة العربية")
        assert "أحمد" in tokens
        assert "يحب" in tokens
        assert "القهوة" in tokens

    def test_mixed_arabic_english(self):
        tokens = arabic_tokenize("أحمد likes Python")
        assert "أحمد" in tokens
        assert "likes" in tokens
        assert "python" in tokens  # lowercase

    def test_filters_single_chars(self):
        tokens = arabic_tokenize("أ ب ت كلمة")
        assert "كلمة" in tokens
        assert "أ" not in tokens

    def test_empty_string(self):
        assert arabic_tokenize("") == []


class TestBM25Search:
    @pytest.fixture
    def bm25_with_docs(self):
        bm25 = ArabicBM25()
        docs = [
            MemoryRecord(id="1", text="أحمد يحب القهوة العربية", scope="user", scope_id="u1"),
            MemoryRecord(id="2", text="محمد يعمل في القاهرة", scope="user", scope_id="u1"),
            MemoryRecord(id="3", text="سارة تدرس الطب في الجامعة", scope="user", scope_id="u1"),
            MemoryRecord(id="4", text="يفضل الشاي الأخضر مع النعناع", scope="user", scope_id="u2"),
        ]
        for doc in docs:
            bm25.add_document(doc)
        return bm25

    def test_basic_search(self, bm25_with_docs):
        results = bm25_with_docs.search("القهوة")
        assert len(results) >= 1
        assert results[0].record.id == "1"
        assert results[0].source == "bm25"

    def test_search_with_filters(self, bm25_with_docs):
        results = bm25_with_docs.search("يحب", filters={"scope": "user", "scope_id": "u1"})
        # Only results from u1
        assert all(r.record.scope_id == "u1" for r in results)

    def test_search_no_results(self, bm25_with_docs):
        results = bm25_with_docs.search("سيارة")
        assert len(results) == 0

    def test_empty_index(self):
        bm25 = ArabicBM25()
        results = bm25.search("أي شيء")
        assert results == []

    def test_remove_document(self, bm25_with_docs):
        bm25_with_docs.remove_document("1")
        results = bm25_with_docs.search("القهوة العربية")
        assert all(r.record.id != "1" for r in results)

    def test_deleted_records_excluded(self):
        bm25 = ArabicBM25()
        doc = MemoryRecord(id="1", text="كلمة مفتاحية مهمة", is_deleted=True)
        bm25.add_document(doc)
        results = bm25.search("مفتاحية")
        assert len(results) == 0

    def test_bulk_load(self):
        bm25 = ArabicBM25()
        records = [
            MemoryRecord(id="1", text="القهوة العربية مشروب تقليدي"),
            MemoryRecord(id="2", text="الشاي الأخضر مفيد للصحة"),
        ]
        bm25.load_documents(records)
        results = bm25.search("القهوة")
        assert len(results) >= 1
        assert results[0].record.id == "1"


class TestRRFFusion:
    def test_rrf_basic(self):
        from dhakira.retrieval.searcher import HybridSearcher

        # We'll test the _rrf_fusion method directly
        class DummyStore:
            pass

        class DummyEmbed:
            pass

        class DummyNorm:
            pass

        searcher = HybridSearcher.__new__(HybridSearcher)
        from dhakira.config import RetrievalConfig
        searcher.config = RetrievalConfig(rrf_k=60)

        r1 = SearchResult(record=MemoryRecord(id="a", text="text a"), score=0.9, source="vector")
        r2 = SearchResult(record=MemoryRecord(id="b", text="text b"), score=0.8, source="vector")
        r3 = SearchResult(record=MemoryRecord(id="a", text="text a"), score=5.0, source="bm25")
        r4 = SearchResult(record=MemoryRecord(id="c", text="text c"), score=4.0, source="bm25")

        fused = searcher._rrf_fusion([r1, r2], [r3, r4], [])

        # "a" appears in both, should have highest combined score
        assert fused[0].record.id == "a"
        assert fused[0].score > fused[1].score
