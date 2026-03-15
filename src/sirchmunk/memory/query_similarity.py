# Copyright (c) ModelScope Contributors. All rights reserved.
"""QuerySimilarityIndex — semantic query similarity with optional embeddings.

Maintains a compact index of historical queries together with their
feedback summaries (mode, confidence, useful keywords, useful files).
At lookup time, the current query is compared to the index:

1. **Embedding path** (preferred): cosine similarity via ``EmbeddingUtil``.
2. **BM25 fallback**: when embeddings are unavailable, uses
   ``BM25Scorer`` from ``sirchmunk.utils.bm25_util`` for lexical
   similarity.

The index is persisted as a small JSON file for portability and
human-readability.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .schemas import SimilarQueryHint


class QuerySimilarityIndex:
    """Lightweight query similarity store with optional embedding support.

    Parameters
    ----------
    index_file : Path
        JSON file for persistent storage.
    embedding_util : optional
        An ``EmbeddingUtil`` instance.  When ``None``, falls back to BM25.
    tokenizer : optional
        A ``TokenizerUtil`` instance forwarded to ``BM25Scorer``.
    max_entries : int
        Maximum number of queries retained (FIFO eviction).
    """

    _SAVE_INTERVAL = 5.0  # seconds between writes

    def __init__(
        self,
        index_file: Path,
        *,
        embedding_util: Any = None,
        tokenizer: Any = None,
        max_entries: int = 2000,
    ):
        self._file = index_file
        self._embedding_util = embedding_util
        self._tokenizer = tokenizer
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: List[Dict[str, Any]] = []
        self._dirty = False
        self._last_save: float = 0.0
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._entries = raw
        except Exception as exc:
            logger.debug(f"QuerySimilarityIndex: load failed: {exc}")

    def _save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            tmp = self._file.with_suffix(".tmp")
            try:
                tmp.write_text(
                    json.dumps(self._entries, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                os.replace(str(tmp), str(self._file))
                self._dirty = False
                self._last_save = time.monotonic()
            except Exception as exc:
                logger.debug(f"QuerySimilarityIndex: save failed: {exc}")
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def _maybe_save(self) -> None:
        if time.monotonic() - self._last_save >= self._SAVE_INTERVAL:
            self._save()

    def close(self) -> None:
        self._save()

    # ── Recording ─────────────────────────────────────────────────────

    def record(
        self,
        query: str,
        *,
        confidence: float,
        mode: str = "DEEP",
        keywords: Optional[List[str]] = None,
        useful_files: Optional[List[str]] = None,
        answer_snippet: str = "",
    ) -> None:
        """Record a query and its outcome summary."""
        embedding = self._encode(query)
        entry: Dict[str, Any] = {
            "query": query,
            "confidence": round(confidence, 4),
            "mode": mode,
            "keywords": (keywords or [])[:15],
            "useful_files": (useful_files or [])[:20],
            "answer_snippet": answer_snippet[:300],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if embedding is not None:
            entry["embedding"] = embedding

        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
            self._dirty = True
        self._maybe_save()

    # ── Lookup ────────────────────────────────────────────────────────

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.55,
    ) -> List[SimilarQueryHint]:
        """Return up to *top_k* similar historical queries."""
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return []

        embedding = self._encode(query)
        if embedding is not None:
            return self._search_by_embedding(
                query, embedding, entries, top_k, min_similarity,
            )
        return self._search_by_bm25(query, entries, top_k, min_similarity)

    # ── Embedding path ────────────────────────────────────────────────

    def _encode(self, text: str) -> Optional[List[float]]:
        if self._embedding_util is None:
            return None
        try:
            vec = self._embedding_util.encode(text)
            if vec is not None:
                return vec if isinstance(vec, list) else vec.tolist()
        except Exception:
            pass
        return None

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _search_by_embedding(
        self,
        query: str,
        query_emb: List[float],
        entries: List[Dict[str, Any]],
        top_k: int,
        min_sim: float,
    ) -> List[SimilarQueryHint]:
        scored: List[tuple] = []
        for e in entries:
            emb = e.get("embedding")
            if emb is None:
                continue
            sim = self._cosine(query_emb, emb)
            if sim >= min_sim and e.get("query", "") != query:
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SimilarQueryHint(
                query=e["query"],
                similarity=round(sim, 4),
                confidence=e.get("confidence", 0.0),
                mode=e.get("mode"),
                keywords=e.get("keywords", []),
                useful_files=e.get("useful_files", []),
            )
            for sim, e in scored[:top_k]
        ]

    # ── BM25 fallback ─────────────────────────────────────────────────

    def _search_by_bm25(
        self,
        query: str,
        entries: List[Dict[str, Any]],
        top_k: int,
        min_sim: float,
    ) -> List[SimilarQueryHint]:
        try:
            from sirchmunk.utils.bm25_util import BM25Scorer

            scorer = BM25Scorer(tokenizer=self._tokenizer)
            docs = [e.get("query", "") for e in entries]
            scores = scorer.score(query, docs)
            if not scores:
                return []
            max_s = max(scores) if scores else 1.0
            if max_s <= 0:
                return []
            normed = [s / max_s for s in scores]
            indexed = sorted(
                enumerate(normed), key=lambda x: x[1], reverse=True,
            )
            results: List[SimilarQueryHint] = []
            for idx, nscore in indexed[:top_k]:
                if nscore < min_sim:
                    break
                e = entries[idx]
                if e.get("query", "") == query:
                    continue
                results.append(SimilarQueryHint(
                    query=e["query"],
                    similarity=round(nscore, 4),
                    confidence=e.get("confidence", 0.0),
                    mode=e.get("mode"),
                    keywords=e.get("keywords", []),
                    useful_files=e.get("useful_files", []),
                ))
            return results
        except Exception:
            return []

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._entries)
            has_emb = sum(1 for e in self._entries if e.get("embedding"))
        return {
            "total_queries": n,
            "with_embeddings": has_emb,
            "embedding_available": self._embedding_util is not None,
        }
