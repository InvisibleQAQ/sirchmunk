"""AgentMemory — self-evolving episodic and semantic memory.

Inspired by Claude Code's CLAUDE.md / memory mechanism, this module
records pipeline behaviour, user search habits, and file-location
patterns.  The accumulated knowledge makes the agent progressively
smarter: future searches benefit from learned shortcuts, optimised
parameters, and path-routing heuristics.

Supports both **text** and **vision** search pipelines — the
``query_type`` field on :class:`SearchEpisode` distinguishes the two.

Architecture (all tables share one DuckDB file):

    ┌──────────────────────────────────────────────────────────────┐
    │  search_episodes    — full search lifecycle log              │
    │  path_patterns      — file-type → directory co-occurrence    │
    │  query_clusters     — semantic query grouping & statistics   │
    │  pipeline_stats     — per-phase latency & yield statistics   │
    │  user_preferences   — learned user-level settings / habits   │
    └──────────────────────────────────────────────────────────────┘

Read-side API:  ``advise()``  — provides hints before a search.
Write-side API: ``record_episode()`` — captures observations after a search.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import duckdb

# ------------------------------------------------------------------ #
# DDL
# ------------------------------------------------------------------ #

_EPISODES_DDL = """\
CREATE TABLE IF NOT EXISTS search_episodes (
    episode_id   VARCHAR PRIMARY KEY,
    query        VARCHAR,
    query_type   VARCHAR,
    paths        VARCHAR,
    top_k        INTEGER,
    result_count INTEGER,
    hit_paths    VARCHAR,
    timings      VARCHAR,
    phase_stats  VARCHAR,
    created_at   VARCHAR
)
"""

_PATH_PATTERNS_DDL = """\
CREATE TABLE IF NOT EXISTS path_patterns (
    directory      VARCHAR,
    file_type      VARCHAR,
    hit_count      INTEGER DEFAULT 0,
    miss_count     INTEGER DEFAULT 0,
    last_hit_query VARCHAR,
    updated_at     VARCHAR,
    PRIMARY KEY (directory, file_type)
)
"""

_QUERY_CLUSTERS_DDL = """\
CREATE TABLE IF NOT EXISTS query_clusters (
    cluster_key        VARCHAR PRIMARY KEY,
    sample_queries     VARCHAR,
    total_searches     INTEGER DEFAULT 0,
    total_hits         INTEGER DEFAULT 0,
    avg_latency_ms     DOUBLE  DEFAULT 0.0,
    best_phase1_cap    INTEGER DEFAULT 0,
    best_scout_top_k   INTEGER DEFAULT 0,
    preferred_paths    VARCHAR,
    updated_at         VARCHAR
)
"""

_PIPELINE_STATS_DDL = """\
CREATE TABLE IF NOT EXISTS pipeline_stats (
    stat_key    VARCHAR PRIMARY KEY,
    stat_value  DOUBLE,
    sample_count INTEGER DEFAULT 0,
    updated_at  VARCHAR
)
"""

_USER_PREFS_DDL = """\
CREATE TABLE IF NOT EXISTS user_preferences (
    pref_key   VARCHAR PRIMARY KEY,
    pref_value VARCHAR,
    updated_at VARCHAR
)
"""


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class SearchEpisode:
    """Complete record of a single search invocation.

    Works for both text search (``query_type='text'``) and vision
    search (``query_type='vision'``, ``'image'``, ``'hybrid'``).
    """

    episode_id: str
    query: str
    query_type: str = "text"          # text | vision | image | hybrid
    paths: List[str] = field(default_factory=list)
    top_k: int = 10
    result_count: int = 0
    hit_paths: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    phase_stats: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(),
    )


@dataclass
class SearchAdvice:
    """Hints provided to the pipeline before a search begins."""

    suggested_paths: List[str] = field(default_factory=list)
    suggested_grep_cap: Optional[int] = None
    suggested_scout_top_k: Optional[int] = None
    path_confidence: float = 0.0
    cluster_key: str = ""
    memo: str = ""


# ------------------------------------------------------------------ #
# Core Memory class
# ------------------------------------------------------------------ #

class AgentMemory:
    """DuckDB-backed long-term memory shared across all search modes.

    One instance per ``work_path``; the DB file lives alongside other
    persistent state so a single cache directory is self-contained.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self._conn.execute(_EPISODES_DDL)
        self._conn.execute(_PATH_PATTERNS_DDL)
        self._conn.execute(_QUERY_CLUSTERS_DDL)
        self._conn.execute(_PIPELINE_STATS_DDL)
        self._conn.execute(_USER_PREFS_DDL)

    def close(self) -> None:
        self._conn.close()

    # ================================================================ #
    #  WRITE  — MemoryRecorder
    # ================================================================ #

    def record_episode(self, ep: SearchEpisode) -> None:
        """Persist a complete search episode and update derived tables."""
        self._conn.execute(
            "INSERT OR REPLACE INTO search_episodes "
            "(episode_id, query, query_type, paths, top_k, "
            " result_count, hit_paths, timings, phase_stats, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ep.episode_id,
                ep.query,
                ep.query_type,
                json.dumps(ep.paths, ensure_ascii=False),
                ep.top_k,
                ep.result_count,
                json.dumps(ep.hit_paths, ensure_ascii=False),
                json.dumps(ep.timings),
                json.dumps(ep.phase_stats, ensure_ascii=False),
                ep.created_at,
            ],
        )
        self._update_path_patterns(ep)
        self._update_query_cluster(ep)
        self._update_pipeline_stats(ep)

    # -- Path patterns ------------------------------------------------

    def _update_path_patterns(self, ep: SearchEpisode) -> None:
        """Track which directories yield hits for which file types."""
        if not ep.hit_paths:
            for p in ep.paths:
                self._increment_path_pattern(
                    directory=p, file_type="*", hit=False, query=ep.query,
                )
            return

        for hp in ep.hit_paths:
            directory = os.path.dirname(hp)
            ext = os.path.splitext(hp)[1].lower() or "*"
            self._increment_path_pattern(
                directory=directory, file_type=ext,
                hit=True, query=ep.query,
            )

    def _increment_path_pattern(
        self,
        directory: str,
        file_type: str,
        hit: bool,
        query: str,
    ) -> None:
        now = datetime.now().isoformat()
        existing = self._conn.execute(
            "SELECT hit_count, miss_count FROM path_patterns "
            "WHERE directory = ? AND file_type = ?",
            [directory, file_type],
        ).fetchall()
        if existing:
            h, m = existing[0]
            if hit:
                h += 1
            else:
                m += 1
            self._conn.execute(
                "UPDATE path_patterns SET hit_count=?, miss_count=?, "
                "last_hit_query=?, updated_at=? "
                "WHERE directory=? AND file_type=?",
                [h, m, query if hit else "", now, directory, file_type],
            )
        else:
            self._conn.execute(
                "INSERT INTO path_patterns "
                "(directory, file_type, hit_count, miss_count, "
                " last_hit_query, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [directory, file_type,
                 1 if hit else 0, 0 if hit else 1,
                 query if hit else "", now],
            )

    # -- Query clusters -----------------------------------------------

    def _update_query_cluster(self, ep: SearchEpisode) -> None:
        """Group semantically similar queries and maintain aggregate stats."""
        key = _cluster_key(ep.query)
        if not key:
            return

        now = datetime.now().isoformat()
        total_latency = sum(ep.timings.values()) * 1000  # ms

        existing = self._conn.execute(
            "SELECT sample_queries, total_searches, total_hits, "
            "avg_latency_ms, best_phase1_cap, best_scout_top_k, "
            "preferred_paths FROM query_clusters "
            "WHERE cluster_key = ?",
            [key],
        ).fetchall()

        if existing:
            row = existing[0]
            samples = _json_loads(row[0], [])
            searches = row[1] + 1
            hits = row[2] + ep.result_count
            avg_lat = row[3] * 0.7 + total_latency * 0.3
            phase1_cap = row[4]
            scout_k = row[5]
            pref_paths = _json_loads(row[6], [])

            if ep.query not in samples:
                samples.append(ep.query)
            samples = samples[-10:]

            p1_count = ep.phase_stats.get("phase1_candidates", 0)
            if ep.result_count > 0 and p1_count > 0:
                phase1_cap = _ema(phase1_cap, p1_count, 0.3)

            scout_k_used = ep.phase_stats.get("scout_top_k", 0)
            if ep.result_count > 0 and scout_k_used > 0:
                scout_k = _ema(scout_k, scout_k_used, 0.3)

            if ep.hit_paths:
                dirs = list({os.path.dirname(p) for p in ep.hit_paths})
                for d in dirs:
                    if d not in pref_paths:
                        pref_paths.append(d)
                pref_paths = pref_paths[-20:]

            self._conn.execute(
                "UPDATE query_clusters SET "
                "sample_queries=?, total_searches=?, total_hits=?, "
                "avg_latency_ms=?, best_phase1_cap=?, "
                "best_scout_top_k=?, preferred_paths=?, updated_at=? "
                "WHERE cluster_key=?",
                [
                    json.dumps(samples, ensure_ascii=False),
                    searches, hits, avg_lat,
                    int(phase1_cap), int(scout_k),
                    json.dumps(pref_paths, ensure_ascii=False),
                    now, key,
                ],
            )
        else:
            pref_paths = list({os.path.dirname(p) for p in ep.hit_paths})
            self._conn.execute(
                "INSERT INTO query_clusters "
                "(cluster_key, sample_queries, total_searches, total_hits, "
                " avg_latency_ms, best_phase1_cap, best_scout_top_k, "
                " preferred_paths, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    key,
                    json.dumps([ep.query], ensure_ascii=False),
                    1,
                    ep.result_count,
                    total_latency,
                    ep.phase_stats.get("phase1_candidates", 0),
                    ep.phase_stats.get("scout_top_k", 0),
                    json.dumps(pref_paths, ensure_ascii=False),
                    now,
                ],
            )

    # -- Pipeline statistics ------------------------------------------

    def _update_pipeline_stats(self, ep: SearchEpisode) -> None:
        """Maintain running averages of per-phase latencies."""
        now = datetime.now().isoformat()
        for phase_name, elapsed in ep.timings.items():
            key = f"latency:{ep.query_type}:{phase_name}"
            existing = self._conn.execute(
                "SELECT stat_value, sample_count FROM pipeline_stats "
                "WHERE stat_key = ?",
                [key],
            ).fetchall()
            if existing:
                old_val, n = existing[0]
                new_val = _ema(old_val, elapsed, min(0.3, 1.0 / (n + 1)))
                self._conn.execute(
                    "UPDATE pipeline_stats SET stat_value=?, "
                    "sample_count=?, updated_at=? WHERE stat_key=?",
                    [new_val, n + 1, now, key],
                )
            else:
                self._conn.execute(
                    "INSERT INTO pipeline_stats "
                    "(stat_key, stat_value, sample_count, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    [key, elapsed, 1, now],
                )

        hr_key = f"hit_rate:{ep.query_type}"
        hit = 1.0 if ep.result_count > 0 else 0.0
        existing = self._conn.execute(
            "SELECT stat_value, sample_count FROM pipeline_stats "
            "WHERE stat_key = ?",
            [hr_key],
        ).fetchall()
        if existing:
            old_val, n = existing[0]
            new_val = _ema(old_val, hit, min(0.3, 1.0 / (n + 1)))
            self._conn.execute(
                "UPDATE pipeline_stats SET stat_value=?, "
                "sample_count=?, updated_at=? WHERE stat_key=?",
                [new_val, n + 1, now, hr_key],
            )
        else:
            self._conn.execute(
                "INSERT INTO pipeline_stats "
                "(stat_key, stat_value, sample_count, updated_at) "
                "VALUES (?, ?, ?, ?)",
                [hr_key, hit, 1, now],
            )

    # ================================================================ #
    #  READ  — MemoryAdvisor
    # ================================================================ #

    def advise(self, query: str, paths: List[str]) -> SearchAdvice:
        """Synthesise advice from accumulated memory before a search starts."""
        advice = SearchAdvice()
        key = _cluster_key(query)
        if not key:
            return advice
        advice.cluster_key = key

        row = self._conn.execute(
            "SELECT total_searches, total_hits, avg_latency_ms, "
            "best_phase1_cap, best_scout_top_k, preferred_paths "
            "FROM query_clusters WHERE cluster_key = ?",
            [key],
        ).fetchall()

        if row:
            (searches, hits, avg_lat, p1_cap,
             scout_k, pref_paths_json) = row[0]
            pref_paths = _json_loads(pref_paths_json, [])

            if p1_cap > 0:
                advice.suggested_grep_cap = p1_cap
            if scout_k > 0:
                advice.suggested_scout_top_k = scout_k

            if pref_paths:
                advice.suggested_paths = pref_paths
                if searches > 0:
                    advice.path_confidence = min(1.0, hits / searches)

            hit_rate = hits / searches if searches > 0 else 0.0
            advice.memo = (
                f"cluster='{key}': {searches} searches, "
                f"hit_rate={hit_rate:.0%}, avg_latency={avg_lat:.0f}ms"
            )

        if not advice.suggested_paths:
            top_dirs = self._get_top_directories(n=5)
            if top_dirs:
                advice.suggested_paths = top_dirs
                advice.path_confidence = 0.3

        return advice

    def get_search_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent search episodes for inspection / debugging."""
        rows = self._conn.execute(
            "SELECT episode_id, query, query_type, result_count, "
            "timings, created_at "
            "FROM search_episodes ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [
            {
                "episode_id": r[0],
                "query": r[1],
                "query_type": r[2],
                "result_count": r[3],
                "timings": _json_loads(r[4], {}),
                "created_at": r[5],
            }
            for r in rows
        ]

    def get_cluster_summary(self) -> List[Dict[str, Any]]:
        """Return all query clusters with aggregate statistics."""
        rows = self._conn.execute(
            "SELECT cluster_key, sample_queries, total_searches, "
            "total_hits, avg_latency_ms, preferred_paths "
            "FROM query_clusters ORDER BY total_searches DESC",
        ).fetchall()
        return [
            {
                "cluster_key": r[0],
                "sample_queries": _json_loads(r[1], []),
                "total_searches": r[2],
                "total_hits": r[3],
                "avg_latency_ms": r[4],
                "preferred_paths": _json_loads(r[5], []),
            }
            for r in rows
        ]

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Return aggregate pipeline performance statistics."""
        rows = self._conn.execute(
            "SELECT stat_key, stat_value, sample_count "
            "FROM pipeline_stats ORDER BY stat_key",
        ).fetchall()
        return {r[0]: {"value": r[1], "n": r[2]} for r in rows}

    def get_path_hotspots(
        self,
        min_hits: int = 2,
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return directories with highest hit rates."""
        rows = self._conn.execute(
            "SELECT directory, file_type, hit_count, miss_count, "
            "last_hit_query FROM path_patterns "
            "WHERE hit_count >= ? "
            "ORDER BY hit_count DESC LIMIT ?",
            [min_hits, top_n],
        ).fetchall()
        return [
            {
                "directory": r[0],
                "file_type": r[1],
                "hit_count": r[2],
                "miss_count": r[3],
                "hit_rate": r[2] / max(r[2] + r[3], 1),
                "last_hit_query": r[4],
            }
            for r in rows
        ]

    def get_user_pref(self, key: str) -> Optional[str]:
        """Retrieve a learned user preference."""
        rows = self._conn.execute(
            "SELECT pref_value FROM user_preferences WHERE pref_key = ?",
            [key],
        ).fetchall()
        return rows[0][0] if rows else None

    def set_user_pref(self, key: str, value: str) -> None:
        """Store a user preference."""
        self._conn.execute(
            "INSERT OR REPLACE INTO user_preferences "
            "(pref_key, pref_value, updated_at) VALUES (?, ?, ?)",
            [key, value, datetime.now().isoformat()],
        )

    def get_episode_count(self) -> int:
        """Total number of recorded search episodes."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM search_episodes",
        ).fetchone()[0]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_top_directories(self, n: int = 5) -> List[str]:
        """Return directories with the highest hit counts."""
        rows = self._conn.execute(
            "SELECT directory, SUM(hit_count) as total_hits "
            "FROM path_patterns "
            "GROUP BY directory "
            "HAVING total_hits > 0 "
            "ORDER BY total_hits DESC LIMIT ?",
            [n],
        ).fetchall()
        return [r[0] for r in rows]


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #

_CLUSTER_STOP_EN = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "is", "it",
    "and", "or", "not", "with", "from", "by", "as", "be", "this", "that",
    "all", "any", "my", "me", "we", "our", "you", "your",
    "find", "show", "get", "give", "look", "help",
    "photo", "image", "picture", "photos", "images", "pictures",
})

_CJK_STOP_WORDS = frozenset(
    "的了是在我有和就不人都帮找到出所全图片照张这那给来个一"
    "请把要为吗呢吧会被让用着过上下中大小它他她们从对于很"
    "也还可以能得里面去看说做没什么怎样哪些多少几"
)

_CJK_RANGE = re.compile(r"[\u4e00-\u9fff]")

_tokenizer: Any = None
_tokenizer_init_attempted = False


def _get_tokenizer() -> Any:
    """Lazy-load :class:`sirchmunk.utils.tokenizer_util.TokenizerUtil`."""
    global _tokenizer, _tokenizer_init_attempted
    if _tokenizer_init_attempted:
        return _tokenizer
    _tokenizer_init_attempted = True
    try:
        from sirchmunk.utils.tokenizer_util import TokenizerUtil
        _tokenizer = TokenizerUtil()
    except Exception:
        _tokenizer = None
    return _tokenizer


def _stem_en(word: str) -> str:
    """Minimal English plural normalizer for clustering."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if len(word) > 4 and re.search(r"(s|x|z|ch|sh)es$", word):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _segment_cjk(text: str) -> List[str]:
    """Segment CJK text into meaningful tokens.

    Uses the main project's :class:`TokenizerUtil` for proper segmentation.
    Falls back to CJK character extraction when unavailable.
    """
    tok = _get_tokenizer()
    if tok is not None:
        tokens = tok.segment(text)
        cjk_tokens: List[str] = []
        for t in tokens:
            t = t.strip()
            if not t or len(t) < 2:
                continue
            if _CJK_RANGE.search(t) and t not in _CJK_STOP_WORDS:
                cjk_tokens.append(t)
        return cjk_tokens

    chars = [c for c in text if _CJK_RANGE.match(c) and c not in _CJK_STOP_WORDS]
    return list(dict.fromkeys(chars))


def _cluster_key(query: str) -> str:
    """Extract a coarse semantic key from a query for cluster grouping.

    Strategy:
      - English tokens: whole words, stopword-filtered, stemmed.
      - CJK tokens: segmented via :class:`TokenizerUtil`, stopword-filtered.
      - Sorted, deduplicated, top 4 → deterministic key.
    """
    q = query.lower()

    en_tokens = re.findall(r"[a-z]{2,}", q)
    en_content = [
        _stem_en(t) for t in en_tokens if t not in _CLUSTER_STOP_EN
    ]

    cjk_content = _segment_cjk(q)

    content = en_content + cjk_content
    if not content:
        return ""
    content = sorted(set(content))[:4]
    return "|".join(content)


def _ema(old: float, new: float, alpha: float) -> float:
    """Exponential moving average."""
    if old == 0.0:
        return new
    return old * (1 - alpha) + new * alpha


def _json_loads(text: Optional[str], default: Any = None) -> Any:
    if not text:
        return default if default is not None else []
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []
