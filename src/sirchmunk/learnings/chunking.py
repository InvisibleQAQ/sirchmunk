# Copyright (c) ModelScope Contributors. All rights reserved.
"""Structure-aware document chunking for evidence extraction.

Splits documents along natural boundaries (paragraphs, headings, code
fences, JSON lines) instead of fixed-size sliding windows, producing
semantically coherent chunks for downstream scoring and evaluation.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sentence boundary detection patterns
# ---------------------------------------------------------------------------
# Matches sentence-ending punctuation followed by whitespace
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?。！？])\s+'
)


def find_sentence_boundary(text: str, near_pos: int, window: int = 50) -> int:
    """Find the nearest sentence boundary to a position within a window.

    Searches for sentence-ending punctuation (.!?。！？) followed by whitespace
    in a window around ``near_pos``, preferring boundaries before the position.

    Args:
        text: Full document text.
        near_pos: Target position to search near.
        window: Search window size on each side (default 50 chars).

    Returns:
        Best boundary position, or ``near_pos`` if no boundary found.
    """
    if near_pos <= 0:
        return 0
    if near_pos >= len(text):
        return len(text)

    start = max(0, near_pos - window)
    end = min(len(text), near_pos + window)
    search_region = text[start:end]

    # Find all sentence boundaries in the region
    boundaries = []
    for m in _SENTENCE_SPLIT_RE.finditer(search_region):
        # Position after the punctuation and whitespace
        abs_pos = start + m.end()
        distance = abs(abs_pos - near_pos)
        # Prefer boundaries before the target position (penalty for after)
        prefer_before = 1 if abs_pos <= near_pos else 2
        boundaries.append((distance, prefer_before, abs_pos))

    if boundaries:
        # Sort by distance, then prefer boundaries before target
        boundaries.sort(key=lambda x: (x[0], x[1]))
        return boundaries[0][2]

    return near_pos


@dataclass
class Chunk:
    """A semantically coherent document segment."""

    start: int
    end: int
    content: str
    chunk_type: str = "text"  # text | code | json_line | heading

    @property
    def length(self) -> int:
        return self.end - self.start


class DocumentChunker:
    """Split documents along natural boundaries.

    Detects document structure (paragraphs, code blocks, JSON lines,
    markdown headings) and produces chunks whose sizes are balanced
    around *target_size* by merging undersize and splitting oversize
    segments.
    """

    _CODE_FENCE = re.compile(r"^```", re.MULTILINE)
    _HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)
    _BLANK_LINE = re.compile(r"\n\s*\n")

    def __init__(
        self,
        target_size: int = 800,
        max_size: int = 2000,
        min_size: int = 100,
    ):
        self.target_size = target_size
        self.max_size = max_size
        self.min_size = min_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: str) -> List[Chunk]:
        """Split *doc* into balanced, boundary-respecting chunks."""
        if not doc or not doc.strip():
            return []

        doc_type = self._detect_type(doc)

        if doc_type == "jsonl":
            raw = self._split_jsonl(doc)
        elif doc_type == "markdown":
            raw = self._split_markdown(doc)
        elif doc_type == "code":
            raw = self._split_code(doc)
        else:
            raw = self._split_paragraphs(doc)

        return self._balance(raw, doc)

    # ------------------------------------------------------------------
    # Heuristic document-type detection
    # ------------------------------------------------------------------

    def _detect_type(self, doc: str) -> str:
        head = doc.lstrip()[:2000]
        if head.startswith("{") and "\n{" in head:
            return "jsonl"
        fence_count = len(self._CODE_FENCE.findall(head))
        if fence_count > 4:
            return "code"
        if self._HEADING.search(head):
            return "markdown"
        return "text"

    # ------------------------------------------------------------------
    # Type-specific splitters
    # ------------------------------------------------------------------

    def _split_jsonl(self, doc: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        pos = 0
        for line in doc.split("\n"):
            end = pos + len(line)
            stripped = line.strip()
            if stripped:
                chunks.append(
                    Chunk(start=pos, end=end, content=stripped, chunk_type="json_line")
                )
            pos = end + 1
        return chunks

    def _split_markdown(self, doc: str) -> List[Chunk]:
        positions = [m.start() for m in self._HEADING.finditer(doc)]
        if not positions:
            return self._split_paragraphs(doc)

        boundaries = sorted(set([0] + positions + [len(doc)]))
        sections: List[Chunk] = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            text = doc[s:e].strip()
            if text:
                ctype = "heading" if s in positions else "text"
                sections.append(Chunk(start=s, end=e, content=text, chunk_type=ctype))
        return sections

    def _split_code(self, doc: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        in_code = False
        code_start = 0
        text_start = 0
        pos = 0

        for line in doc.split("\n"):
            line_end = pos + len(line)
            if line.strip().startswith("```"):
                if not in_code:
                    preceding = doc[text_start:pos].strip()
                    if preceding:
                        chunks.append(
                            Chunk(start=text_start, end=pos, content=preceding, chunk_type="text")
                        )
                    code_start = pos
                    in_code = True
                else:
                    code_content = doc[code_start:line_end].strip()
                    if code_content:
                        chunks.append(
                            Chunk(start=code_start, end=line_end, content=code_content, chunk_type="code")
                        )
                    in_code = False
                    text_start = line_end + 1
            pos = line_end + 1

        remaining_start = code_start if in_code else text_start
        remaining = doc[remaining_start:].strip()
        if remaining:
            rtype = "code" if in_code else "text"
            chunks.append(
                Chunk(start=remaining_start, end=len(doc), content=remaining, chunk_type=rtype)
            )
        return chunks if chunks else self._split_paragraphs(doc)

    def _split_paragraphs(self, doc: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        parts = self._BLANK_LINE.split(doc)
        pos = 0
        for part in parts:
            idx = doc.find(part, pos)
            if idx == -1:
                idx = pos
            stripped = part.strip()
            if stripped:
                chunks.append(
                    Chunk(start=idx, end=idx + len(part), content=stripped, chunk_type="text")
                )
            pos = idx + len(part)
        return chunks if chunks else [Chunk(start=0, end=len(doc), content=doc, chunk_type="text")]

    # ------------------------------------------------------------------
    # Post-processing: merge / split to target size
    # ------------------------------------------------------------------

    def _balance(self, raw_chunks: List[Chunk], doc: str) -> List[Chunk]:
        if not raw_chunks:
            return [Chunk(start=0, end=len(doc), content=doc, chunk_type="text")]

        balanced: List[Chunk] = []
        buf: Optional[Chunk] = None

        for chunk in raw_chunks:
            if chunk.length > self.max_size:
                if buf:
                    balanced.append(buf)
                    buf = None
                balanced.extend(self._split_fixed(chunk))
            elif chunk.length < self.min_size and buf is not None:
                buf = Chunk(
                    start=buf.start,
                    end=chunk.end,
                    content=doc[buf.start : chunk.end],
                    chunk_type=buf.chunk_type,
                )
                if buf.length >= self.target_size:
                    balanced.append(buf)
                    buf = None
            else:
                if buf:
                    balanced.append(buf)
                buf = chunk

        if buf:
            balanced.append(buf)

        return balanced if balanced else [Chunk(start=0, end=len(doc), content=doc, chunk_type="text")]

    def _split_fixed(self, chunk: Chunk) -> List[Chunk]:
        """Split oversized chunks at natural boundaries (sentence > paragraph > newline).

        Uses sentence boundary detection to avoid cutting mid-sentence,
        which improves evidence evaluation quality.
        """
        parts: List[Chunk] = []
        text = chunk.content
        offset = chunk.start
        step = self.target_size

        i = 0
        while i < len(text):
            end = min(i + step, len(text))
            if end < len(text):
                # Strategy 1: Try sentence boundary first (highest quality)
                sent_boundary = find_sentence_boundary(text, end, window=step // 4)
                if sent_boundary > i and sent_boundary < len(text):
                    end = sent_boundary
                else:
                    # Strategy 2: Fall back to paragraph/newline/punctuation
                    search_start = max(i, end - step // 5)
                    for sep in ("\n\n", "\n", ". ", "。", "；", "; "):
                        idx = text.rfind(sep, search_start, end)
                        if idx > i:
                            end = idx + len(sep)
                            break
            segment = text[i:end]
            if segment.strip():
                parts.append(
                    Chunk(
                        start=offset + i,
                        end=offset + end,
                        content=segment,
                        chunk_type=chunk.chunk_type,
                    )
                )
            i = end

        return parts
