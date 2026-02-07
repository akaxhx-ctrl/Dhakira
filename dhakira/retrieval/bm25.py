"""Arabic-aware BM25 keyword search."""

from __future__ import annotations

import logging
import re

from dhakira.config import BM25Config
from dhakira.models import MemoryRecord, SearchResult

logger = logging.getLogger(__name__)

# Simple Arabic tokenizer: split on whitespace and punctuation, remove short tokens
_TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")


def arabic_tokenize(text: str) -> list[str]:
    """Tokenize Arabic text for BM25.

    Simple whitespace + punctuation tokenizer that preserves Arabic words.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if len(t) > 1]  # Filter single-char tokens


class ArabicBM25:
    """Arabic-aware BM25 keyword search.

    Maintains an in-memory BM25 index of memory records for fast
    keyword-based retrieval. Complements vector search for cases
    where exact term matching is important.
    """

    def __init__(self, config: BM25Config | None = None):
        self.config = config or BM25Config()
        self._documents: list[MemoryRecord] = []
        self._tokenized_docs: list[list[str]] = []
        self._bm25 = None
        self._dirty = True

    def add_document(self, record: MemoryRecord) -> None:
        """Add a document to the BM25 index."""
        self._documents.append(record)
        self._tokenized_docs.append(arabic_tokenize(record.text))
        self._dirty = True

    def remove_document(self, record_id: str) -> None:
        """Remove a document from the BM25 index."""
        for i, doc in enumerate(self._documents):
            if doc.id == record_id:
                self._documents.pop(i)
                self._tokenized_docs.pop(i)
                self._dirty = True
                break

    def update_document(self, record: MemoryRecord) -> None:
        """Update a document in the BM25 index."""
        self.remove_document(record.id)
        self.add_document(record)

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from current documents."""
        if not self._dirty:
            return

        if not self._tokenized_docs:
            self._bm25 = None
            self._dirty = False
            return

        from rank_bm25 import BM25Plus

        self._bm25 = BM25Plus(
            self._tokenized_docs,
            k1=self.config.k1,
            b=self.config.b,
        )
        self._dirty = False

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search the BM25 index.

        Args:
            query: Search query (will be tokenized).
            limit: Maximum number of results.
            filters: Optional filters for scope/scope_id.

        Returns:
            List of SearchResult objects sorted by BM25 score.
        """
        if not self._documents:
            return []

        self._rebuild_index()

        if self._bm25 is None:
            return []

        query_tokens = arabic_tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with documents and filter
        scored_docs = list(zip(scores, self._documents))

        # Apply filters
        if filters:
            scored_docs = [
                (score, doc) for score, doc in scored_docs
                if self._matches_filters(doc, filters)
            ]

        # Filter out deleted records
        scored_docs = [(s, d) for s, d in scored_docs if not d.is_deleted]

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Filter out zero scores and limit
        results = []
        for score, doc in scored_docs[:limit]:
            if score > 0:
                results.append(SearchResult(record=doc, score=float(score), source="bm25"))

        return results

    def _matches_filters(self, doc: MemoryRecord, filters: dict) -> bool:
        """Check if a document matches the given filters."""
        for key, value in filters.items():
            if key == "scope" and doc.scope != value:
                return False
            if key == "scope_id" and doc.scope_id != value:
                return False
        return True

    def load_documents(self, records: list[MemoryRecord]) -> None:
        """Bulk load documents into the index."""
        self._documents = list(records)
        self._tokenized_docs = [arabic_tokenize(r.text) for r in records]
        self._dirty = True
