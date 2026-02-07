"""Sentence-aware Arabic text chunking."""

from __future__ import annotations

import re

from dhakira.arabic.utils import arabic_token_count
from dhakira.config import ChunkerConfig
from dhakira.models import Chunk

# Arabic sentence boundaries: period, exclamation, question mark (Arabic and Latin), Urdu full stop
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟\u06D4])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class ArabicChunker:
    """Sentence-aware chunker for Arabic text.

    Splits on Arabic sentence boundaries, merges short sentences,
    splits long ones, and adds overlap between chunks. Sentence-aware
    chunking scores ~74.78 vs 69.41 for fixed-size on Arabic RAG tasks.
    """

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into sentence-aware chunks with overlap.

        1. Split on paragraph breaks first
        2. Split on Arabic sentence boundaries
        3. Merge short sentences until reaching min_tokens
        4. Split long sentences if they exceed max_tokens
        5. Add overlap between chunks
        """
        if not text or not text.strip():
            return []

        # Split into paragraphs, then sentences
        sentences = []
        paragraphs = _PARAGRAPH_SPLIT.split(text)
        for para in paragraphs:
            para_sentences = _SENTENCE_SPLIT.split(para.strip())
            sentences.extend(s.strip() for s in para_sentences if s.strip())

        if not sentences:
            return [Chunk(text=text.strip(), start_char=0, end_char=len(text), token_count=arabic_token_count(text))]

        # Merge short sentences and split long ones
        merged = self._merge_and_split(sentences)

        # Build chunks with character offsets
        chunks = []
        search_start = 0
        for chunk_text in merged:
            start = text.find(chunk_text[:20], search_start)
            if start == -1:
                start = search_start
            end = start + len(chunk_text)
            search_start = start + 1

            chunks.append(Chunk(
                text=chunk_text,
                start_char=start,
                end_char=end,
                token_count=arabic_token_count(chunk_text),
            ))

        # Add overlap
        if self.config.overlap_ratio > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks, text)

        return chunks

    def _merge_and_split(self, sentences: list[str]) -> list[str]:
        """Merge short sentences and split long ones."""
        result: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = arabic_token_count(sent)

            # If single sentence exceeds max, split it by words
            if sent_tokens > self.config.max_tokens:
                # Flush current buffer first
                if current_parts:
                    result.append(" ".join(current_parts))
                    current_parts = []
                    current_tokens = 0

                # Split long sentence
                result.extend(self._split_long_sentence(sent))
                continue

            # If adding this sentence would exceed max, flush and start new chunk
            if current_tokens + sent_tokens > self.config.max_tokens and current_parts:
                result.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(sent)
            current_tokens += sent_tokens

        # Flush remaining
        if current_parts:
            result.append(" ".join(current_parts))

        return result

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Split a long sentence into smaller chunks by words."""
        words = sentence.split()
        result: list[str] = []
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = arabic_token_count(word)
            if current_tokens + word_tokens > self.config.max_tokens and current_words:
                result.append(" ".join(current_words))
                current_words = []
                current_tokens = 0

            current_words.append(word)
            current_tokens += word_tokens

        if current_words:
            result.append(" ".join(current_words))

        return result

    def _add_overlap(self, chunks: list[Chunk], original_text: str) -> list[Chunk]:
        """Add overlap between adjacent chunks."""
        if len(chunks) <= 1:
            return chunks

        overlap_tokens = int(self.config.max_tokens * self.config.overlap_ratio)
        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            prev_words = prev_text.split()

            # Get last N words from previous chunk as overlap
            overlap_words: list[str] = []
            overlap_count = 0
            for word in reversed(prev_words):
                word_tok = arabic_token_count(word)
                if overlap_count + word_tok > overlap_tokens:
                    break
                overlap_words.insert(0, word)
                overlap_count += word_tok

            if overlap_words:
                overlap_prefix = " ".join(overlap_words)
                new_text = overlap_prefix + " " + chunks[i].text
            else:
                new_text = chunks[i].text

            result.append(Chunk(
                text=new_text,
                start_char=chunks[i].start_char,
                end_char=chunks[i].end_char,
                token_count=arabic_token_count(new_text),
            ))

        return result
