# Copyright (c) ModelScope Contributors. All rights reserved.
"""PatternMemory — query pattern → strategy mapping + reasoning chain templates.

Backed by JSON files for human-readability and easy schema evolution.
Thread-safe via a reentrant lock.

Storage layout::

    {base_dir}/query_patterns.json
    {base_dir}/reasoning_chains.json
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .base import MemoryStore
from .schemas import (
    QueryPattern,
    ReasoningChain,
    StrategyHint,
    compute_pattern_id,
    compute_pattern_id_at_level,
)

# Heuristic word sets for lightweight query classification (no LLM needed)
_COMPARISON_WORDS = frozenset({
    "which", "both", "difference", "compare", "versus", "vs",
    "better", "more", "less", "rather", "prefer", "between",
})
_FACTUAL_WORDS = frozenset({
    "when", "where", "who", "what", "how many", "how much",
})
_DEFINITION_WORDS = frozenset({
    "what is", "what are", "define", "definition", "meaning",
})
_PROCEDURAL_WORDS = frozenset({
    "how to", "how do", "steps", "process", "procedure",
})

# Chinese query classification patterns
_ZH_COMPARISON_RE = re.compile(r"哪个更|对比|区别|比较|和.+哪个|还是.+好")
_ZH_FACTUAL_RE = re.compile(r"谁|什么时候|哪里|多少|几个|在哪")
_ZH_DEFINITION_RE = re.compile(r"什么是|是什么|定义|含义|意思是")
_ZH_PROCEDURAL_RE = re.compile(r"如何|怎么|怎样|步骤|流程|方法是")
_ZH_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Pre-compiled regex for classify_query entity extraction
_EN_NAMED_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_YEAR_RE = re.compile(r"\b\d{4}\b")
_LOCATION_RE = re.compile(r"\b(?:city|country|state|river|mountain)\b")
_CJK_LOC_RE = re.compile(r"[\u4e00-\u9fff]{2,}(?:市|省|县|国|河|山|湖)")
_CJK_TITLE_RE = re.compile(r"《[^》]+》")

# HMRPL: minimum samples required per resolution level for confident predictions
_MIN_SAMPLES_BY_LEVEL = {4: 3, 3: 3, 2: 2, 1: 2, 0: 1}


class PatternMemory(MemoryStore):
    """Query pattern → strategy mapping and reasoning chain templates.

    Uses heuristic query classification at lookup time (zero-LLM-cost)
    and learns optimal parameters from feedback signals.
    """

    _MIN_SAMPLES_FOR_CONFIDENT = 3
    _MIN_SUCCESS_RATE = 0.4
    _EMA_ALPHA = 0.3
    _SAVE_MIN_INTERVAL = 5.0  # seconds between disk writes

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._patterns_file = base_dir / "query_patterns.json"
        self._chains_file = base_dir / "reasoning_chains.json"
        self._patterns: Dict[str, QueryPattern] = {}
        self._chains: Dict[str, ReasoningChain] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._last_save_time: float = 0.0

    # ── MemoryStore protocol ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "PatternMemory"

    def initialize(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def decay(self, now: Optional[datetime] = None) -> int:
        if not now:
            now = datetime.now(timezone.utc)
        count = 0
        with self._lock:
            for p in self._patterns.values():
                try:
                    updated = datetime.fromisoformat(p.updated_at)
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    if (now - updated).days > 30 and p.success_rate > 0.1:
                        p.success_rate *= 0.95
                        count += 1
                except (ValueError, TypeError):
                    continue
        if count:
            self._mark_dirty()
        return count

    def cleanup(self, max_entries: Optional[int] = None) -> int:
        max_entries = max_entries or 500
        removed = 0
        with self._lock:
            if len(self._patterns) <= max_entries:
                return 0
            ranked = sorted(
                self._patterns.items(),
                key=lambda x: (x[1].success_rate, x[1].sample_count),
            )
            while len(self._patterns) > max_entries and ranked:
                pid, _ = ranked.pop(0)
                del self._patterns[pid]
                self._chains.pop(f"{pid}_chain", None)
                removed += 1
        if removed:
            self._mark_dirty()
        return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._patterns)
            return {
                "name": self.name,
                "patterns_count": total,
                "chains_count": len(self._chains),
                "avg_success_rate": (
                    sum(p.success_rate for p in self._patterns.values())
                    / max(total, 1)
                ),
            }

    def close(self) -> None:
        self._flush()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        with self._lock:
            self._patterns = self._load_json(
                self._patterns_file, QueryPattern.from_dict,
            )
            self._chains = self._load_json(
                self._chains_file, ReasoningChain.from_dict,
            )
            self._dirty = False

    @staticmethod
    def _load_json(path: Path, factory):
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {k: factory(v) for k, v in raw.items()}
        except Exception as exc:
            logger.warning(f"PatternMemory: failed to load {path.name}: {exc}")
            return {}

    def _mark_dirty(self) -> None:
        """Mark data as changed and flush if enough time has elapsed."""
        self._dirty = True
        now = time.monotonic()
        if now - self._last_save_time >= self._SAVE_MIN_INTERVAL:
            self._flush()

    def _flush(self) -> None:
        """Unconditionally write dirty data to disk."""
        with self._lock:
            if not self._dirty:
                return
            self._atomic_write(
                self._patterns_file,
                {k: v.to_dict() for k, v in self._patterns.items()},
            )
            self._atomic_write(
                self._chains_file,
                {k: v.to_dict() for k, v in self._chains.items()},
            )
            self._dirty = False
            self._last_save_time = time.monotonic()

    @staticmethod
    def _atomic_write(path: Path, data: Any) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(path))
        except Exception as exc:
            logger.warning(f"PatternMemory: write failed for {path.name}: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    # ── Query classification (heuristic, zero-LLM) ───────────────────

    @staticmethod
    def classify_query(query: str) -> Dict[str, Any]:
        """Extract lightweight feature vector from a query string.

        Supports both English and Chinese queries.

        Returns dict with ``query_type``, ``complexity``, ``entity_types``,
        ``entity_count``, and ``hop_hint``.
        """
        q_lower = query.lower().strip()
        words = q_lower.split()
        has_cjk = bool(_ZH_CJK_RE.search(query))

        # --- Query type ---
        query_type = "factual"
        if has_cjk:
            if _ZH_DEFINITION_RE.search(query):
                query_type = "definition"
            elif _ZH_PROCEDURAL_RE.search(query):
                query_type = "procedural"
            elif _ZH_COMPARISON_RE.search(query):
                query_type = "comparison"
            elif _ZH_FACTUAL_RE.search(query):
                query_type = "factual"
            else:
                query_type = "bridge"
        else:
            if any(w in q_lower for w in _DEFINITION_WORDS):
                query_type = "definition"
            elif any(w in q_lower for w in _PROCEDURAL_WORDS):
                query_type = "procedural"
            elif any(w in words for w in _COMPARISON_WORDS):
                query_type = "comparison"
            elif any(w in q_lower for w in _FACTUAL_WORDS):
                query_type = "factual"
            else:
                query_type = "bridge"

        # --- Complexity ---
        if has_cjk:
            char_count = len(query.strip())
            if char_count <= 10:
                complexity = "simple"
            elif char_count <= 30:
                complexity = "moderate"
            else:
                complexity = "complex"
        else:
            if len(words) <= 6:
                complexity = "simple"
            elif len(words) <= 15:
                complexity = "moderate"
            else:
                complexity = "complex"

        # --- Entity types + entity count ---
        entity_types: List[str] = []
        entity_count = 0

        en_named = _EN_NAMED_RE.findall(query)
        if en_named:
            entity_types.append("named_entity")
            entity_count += len(en_named)

        year_matches = _YEAR_RE.findall(query)
        if year_matches:
            entity_types.append("date")
            entity_count += len(year_matches)

        if _LOCATION_RE.search(q_lower):
            entity_types.append("location")

        if has_cjk:
            cjk_entities = _CJK_LOC_RE.findall(query)
            if cjk_entities:
                if "location" not in entity_types:
                    entity_types.append("location")
                entity_count += len(cjk_entities)
            cjk_names = _CJK_TITLE_RE.findall(query)
            if cjk_names:
                entity_types.append("title")
                entity_count += len(cjk_names)

        # Bucket entity_count for stable hashing
        ec_bucket = min(entity_count, 3)

        # --- Hop hint ---
        hop_hint = "single"
        if query_type == "comparison" or entity_count >= 2:
            hop_hint = "multi"
        elif query_type == "bridge":
            hop_hint = "multi" if entity_count >= 1 else "single"

        return {
            "query_type": query_type,
            "complexity": complexity,
            "entity_types": entity_types,
            "entity_count": ec_bucket,
            "hop_hint": hop_hint,
        }

    # ── Public API ────────────────────────────────────────────────────

    def suggest_strategy(self, query: str) -> Optional[StrategyHint]:
        """Return a strategy hint via hierarchical Thompson Sampling (HMRPL).

        Performs fine-to-coarse fallback (L4 → L0), returning the first level
        that has enough samples and passes the Thompson Sampling threshold.
        When no level matches, falls back to soft-match or warm transfer.
        """
        features = self.classify_query(query)

        # Extract feature components for compute_pattern_id_at_level
        query_type = features["query_type"]
        complexity = features["complexity"]
        entity_types = features["entity_types"]
        entity_count = features.get("entity_count", 0)
        hop_hint = features.get("hop_hint", "single")

        with self._lock:
            # HMRPL: fine-to-coarse hierarchical lookup (L4 → L0)
            for level in range(4, -1, -1):
                pid = compute_pattern_id_at_level(
                    query_type, complexity, entity_types,
                    entity_count, hop_hint, level=level,
                )
                pattern = self._patterns.get(pid)
                if not pattern:
                    continue

                min_samples = _MIN_SAMPLES_BY_LEVEL.get(level, 3)
                if pattern.sample_count < min_samples:
                    continue

                # Thompson Sampling: draw from Beta posterior
                sampled = random.betavariate(
                    max(pattern.alpha, 0.01), max(pattern.beta_param, 0.01),
                )
                if sampled >= self._MIN_SUCCESS_RATE:
                    token_budget = self._estimate_token_budget(pattern, features)
                    return StrategyHint(
                        mode=pattern.optimal_mode,
                        top_k_files=pattern.optimal_params.get("top_k_files"),
                        max_loops=pattern.optimal_params.get("max_loops"),
                        enable_dir_scan=pattern.optimal_params.get("enable_dir_scan"),
                        keyword_strategy=pattern.optimal_params.get("keyword_strategy"),
                        confidence=sampled,
                        source_pattern_id=pattern.pattern_id,
                        resolution_level=level,
                        token_budget=token_budget,
                    )

            # Hierarchical lookup found nothing; try legacy soft-match fallback
            pattern = self._soft_match(features)

        if not pattern:
            # CTS-WT warm transfer for cold-start
            warm_hint = self._suggest_warm_transfer(features)
            if warm_hint:
                return warm_hint
            return None
        if pattern.sample_count < self._MIN_SAMPLES_FOR_CONFIDENT:
            return None

        # Thompson Sampling on soft-matched pattern
        sampled = random.betavariate(
            max(pattern.alpha, 0.01), max(pattern.beta_param, 0.01),
        )
        if sampled < self._MIN_SUCCESS_RATE:
            return None

        token_budget = self._estimate_token_budget(pattern, features)
        return StrategyHint(
            mode=pattern.optimal_mode,
            top_k_files=pattern.optimal_params.get("top_k_files"),
            max_loops=pattern.optimal_params.get("max_loops"),
            enable_dir_scan=pattern.optimal_params.get("enable_dir_scan"),
            keyword_strategy=pattern.optimal_params.get("keyword_strategy"),
            confidence=sampled,
            source_pattern_id=pattern.pattern_id,
            resolution_level=pattern.resolution_level,
            token_budget=token_budget,
        )

    def _soft_match(self, features: Dict[str, Any]) -> Optional[QueryPattern]:
        """Find the nearest pattern when exact pid is absent.

        Uses feature-level overlap (Jaccard on entity_types + exact match
        on query_type and complexity).  Returns the best match only when
        it has sufficient samples.
        """
        target_et = set(features.get("entity_types", []))
        best: Optional[Tuple[float, QueryPattern]] = None
        for p in self._patterns.values():
            if p.sample_count < self._MIN_SAMPLES_FOR_CONFIDENT:
                continue
            score = 0.0
            if p.query_type == features["query_type"]:
                score += 0.4
            if p.complexity == features["complexity"]:
                score += 0.2
            if p.hop_hint == features.get("hop_hint", "single"):
                score += 0.1
            pet = set(p.entity_types)
            if target_et or pet:
                jaccard = len(target_et & pet) / max(len(target_et | pet), 1)
                score += 0.3 * jaccard
            else:
                score += 0.3
            if best is None or score > best[0]:
                best = (score, p)
        if best and best[0] >= 0.6:
            return best[1]
        return None

    def _estimate_token_budget(
        self, pattern: QueryPattern, features: Dict[str, Any]
    ) -> Optional[int]:
        """Estimate optimal token budget from historical pattern data.

        Uses Pareto principle: budget = avg_tokens * complexity_factor,
        clamped to reasonable bounds. Returns None if insufficient data.
        """
        if pattern.sample_count < 3:
            return None
        avg_tokens = pattern.optimal_params.get("avg_tokens", 0)
        if avg_tokens <= 0:
            return None

        complexity_factor = {"simple": 0.8, "moderate": 1.0, "complex": 1.3}
        factor = complexity_factor.get(features.get("complexity", "moderate"), 1.0)
        budget = int(avg_tokens * factor)

        # Clamp to reasonable bounds
        return max(10000, min(budget, 150000))

    def _find_similar_patterns(self, features: Dict[str, Any], k: int = 5) -> List[Tuple[float, QueryPattern]]:
        """Find k most similar patterns by heuristic feature similarity.

        Uses same scoring as _soft_match but returns top-k with lower
        threshold (0.3 vs 0.6) to enable warm transfer aggregation.
        """
        candidates: List[Tuple[float, QueryPattern]] = []
        for pattern in self._patterns.values():
            if pattern.sample_count < 1:
                continue
            score = 0.0
            # Same scoring dimensions as existing _soft_match
            if pattern.query_type == features.get("query_type"):
                score += 0.4
            if pattern.complexity == features.get("complexity"):
                score += 0.2
            if pattern.hop_hint == features.get("hop_hint"):
                score += 0.2
            # Entity type overlap
            pat_types = set(pattern.entity_types or [])
            feat_types = set(features.get("entity_types", []))
            if pat_types and feat_types:
                overlap = len(pat_types & feat_types) / max(len(pat_types | feat_types), 1)
                score += 0.2 * overlap

            if score >= 0.3:  # lower threshold for warm transfer
                candidates.append((score, pattern))

        # Return top-k sorted by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:k]

    def _suggest_warm_transfer(self, features: Dict[str, Any]) -> Optional[StrategyHint]:
        """CTS-WT: Warm transfer from similar patterns for cold-start queries.

        Aggregates Beta posteriors from k nearest patterns weighted by
        feature similarity, then samples from the aggregate distribution.
        Zero LLM cost — uses heuristic feature similarity only.
        """
        with self._lock:
            similar = self._find_similar_patterns(features, k=5)

        if not similar:
            return None

        # Weighted aggregation of Beta parameters
        alpha_warm = 0.0
        beta_warm = 0.0
        total_weight = 0.0
        best_mode: Optional[str] = None
        best_params: Dict[str, Any] = {}
        best_pattern: Optional[QueryPattern] = None
        best_weight = 0.0

        for score, pattern in similar:
            w = score  # similarity score as weight
            alpha_warm += w * pattern.alpha
            beta_warm += w * pattern.beta_param
            total_weight += w
            if w > best_weight:
                best_weight = w
                best_mode = pattern.optimal_mode
                best_params = pattern.optimal_params
                best_pattern = pattern

        if total_weight < 1e-9:
            return None

        # Normalize
        alpha_warm /= total_weight
        beta_warm /= total_weight

        # Sample from warm-transferred Beta
        sampled = random.betavariate(max(alpha_warm, 0.01), max(beta_warm, 0.01))

        if sampled >= self._MIN_SUCCESS_RATE and best_mode:
            token_budget = None
            if best_pattern:
                token_budget = self._estimate_token_budget(best_pattern, features)
            return StrategyHint(
                mode=best_mode,
                top_k_files=best_params.get("top_k_files") if best_params else None,
                max_loops=best_params.get("max_loops") if best_params else None,
                enable_dir_scan=best_params.get("enable_dir_scan") if best_params else None,
                keyword_strategy=best_params.get("keyword_strategy") if best_params else None,
                confidence=round(sampled, 3),
                source_pattern_id="warm_transfer",
                resolution_level=0,  # warm transfer is lowest confidence level
                token_budget=token_budget,
            )
        return None

    def _update_pattern_stats(
        self,
        pattern: QueryPattern,
        confidence: float,
        mode: str,
        params: Dict[str, Any],
        latency: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Update statistics on a pattern (shared by primary and ancestor updates).

        This is a helper for record_outcome() to avoid code duplication when
        propagating updates to ancestor patterns in the HMRPL hierarchy.
        """
        success = confidence >= 0.5
        now = datetime.now(timezone.utc).isoformat()

        pattern.sample_count += 1
        if success:
            pattern.success_count += 1
        pattern.success_rate = (
            pattern.success_count / max(pattern.sample_count, 1)
        )

        # Thompson Sampling: update Beta posterior
        pattern.alpha += confidence
        pattern.beta_param += (1.0 - confidence)

        # HMRPL: update total_visits and avg_reward
        pattern.total_visits += 1
        ema_reward = 0.1  # slower EMA for hierarchical reward
        pattern.avg_reward = (
            ema_reward * confidence + (1 - ema_reward) * pattern.avg_reward
        )

        ema = self._EMA_ALPHA
        pattern.avg_latency = (
            ema * latency + (1 - ema) * pattern.avg_latency
        )
        pattern.avg_tokens = int(
            ema * tokens + (1 - ema) * pattern.avg_tokens
        )

        # Also track avg_tokens in optimal_params for token budget estimation
        if tokens and tokens > 0:
            old_avg = pattern.optimal_params.get("avg_tokens", 0)
            if old_avg > 0:
                new_avg_tokens = int(0.3 * tokens + 0.7 * old_avg)
            else:
                new_avg_tokens = tokens
            pattern.optimal_params["avg_tokens"] = new_avg_tokens
        else:
            new_avg_tokens = pattern.optimal_params.get("avg_tokens")

        if success and pattern.success_rate >= 0.5:
            pattern.optimal_mode = mode
            pattern.optimal_params = params
            # Preserve avg_tokens in new optimal_params
            if new_avg_tokens:
                pattern.optimal_params["avg_tokens"] = new_avg_tokens

        pattern.updated_at = now

    def record_outcome(
        self,
        query: str,
        confidence: float,
        mode: str,
        params: Dict[str, Any],
        latency: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Record a search outcome with continuous *confidence* (0-1).

        The old binary ``success`` is replaced by a gradient signal:
        ``confidence >= 0.5`` counts as a success for rate tracking,
        and the Beta distribution is updated proportionally.

        HMRPL: Also propagates updates to all ancestor levels and performs
        auto-split/merge when conditions are met.
        """
        features = self.classify_query(query)

        # Extract feature components for hierarchical pattern IDs
        query_type = features["query_type"]
        complexity = features["complexity"]
        entity_types = features["entity_types"]
        entity_count = features.get("entity_count", 0)
        hop_hint = features.get("hop_hint", "single")

        # Primary pattern at finest level (L4)
        pid = compute_pattern_id_at_level(
            query_type, complexity, entity_types,
            entity_count, hop_hint, level=4,
        )
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                pattern = QueryPattern(
                    pattern_id=pid,
                    query_type=query_type,
                    entity_types=entity_types,
                    complexity=complexity,
                    entity_count=entity_count,
                    hop_hint=hop_hint,
                    resolution_level=4,
                    optimal_mode=mode,
                    optimal_params=params,
                    created_at=now,
                    updated_at=now,
                )
                self._patterns[pid] = pattern

            # Update primary pattern stats using helper
            self._update_pattern_stats(
                pattern, confidence, mode, params, latency, tokens,
            )
            current_level = pattern.resolution_level

            # HMRPL: Propagate update to all ancestor levels (L3 → L0)
            for ancestor_level in range(current_level - 1, -1, -1):
                ancestor_pid = compute_pattern_id_at_level(
                    query_type, complexity, entity_types,
                    entity_count, hop_hint, level=ancestor_level,
                )
                ancestor = self._patterns.get(ancestor_pid)
                if ancestor is None:
                    # Create ancestor pattern with minimal initialization
                    ancestor = QueryPattern(
                        pattern_id=ancestor_pid,
                        query_type=query_type,
                        resolution_level=ancestor_level,
                        complexity=complexity if ancestor_level >= 1 else "moderate",
                        hop_hint=hop_hint if ancestor_level >= 2 else "single",
                        entity_count=entity_count if ancestor_level >= 3 else 0,
                        entity_types=entity_types if ancestor_level >= 4 else [],
                        optimal_mode=mode,
                        optimal_params=params,
                        created_at=now,
                        updated_at=now,
                    )
                    self._patterns[ancestor_pid] = ancestor
                # Update ancestor's stats (same update logic)
                self._update_pattern_stats(
                    ancestor, confidence, mode, params, latency, tokens,
                )

                # HMRPL: Auto-split on ancestor if it has enough samples
                if (ancestor.sample_count > 20
                        and ancestor.resolution_level < 4
                        and not ancestor.children_ids):
                    self._try_split_pattern(ancestor, mode, params, now)

                # HMRPL: Auto-merge children if they've converged
                self._try_merge_children(ancestor)

                # Link parent-child relationship
                if ancestor_level == current_level - 1:
                    pattern.parent_id = ancestor_pid
                    if pid not in ancestor.children_ids:
                        ancestor.children_ids.append(pid)

        self._mark_dirty()

    def _try_split_pattern(
        self,
        pattern: QueryPattern,
        mode: str,
        params: Dict[str, Any],
        now: str,
    ) -> None:
        """Attempt to create child patterns at finer resolution level.

        Children inherit half of the parent's Beta priors (α/2, β/2) to
        bootstrap with some prior knowledge while allowing divergence.
        """
        child_level = pattern.resolution_level + 1
        if child_level > 4:
            return

        # Create a single child pattern at the finer level
        # (actual child patterns are created lazily when queries arrive)
        child_pid = compute_pattern_id_at_level(
            pattern.query_type, pattern.complexity,
            pattern.entity_types, pattern.entity_count,
            pattern.hop_hint, level=child_level,
        )

        if child_pid not in self._patterns:
            child = QueryPattern(
                pattern_id=child_pid,
                query_type=pattern.query_type,
                resolution_level=child_level,
                complexity=pattern.complexity,
                hop_hint=pattern.hop_hint,
                entity_count=pattern.entity_count,
                entity_types=pattern.entity_types,
                optimal_mode=mode,
                optimal_params=params,
                # Inherited Beta priors (halved for uncertainty)
                alpha=max(pattern.alpha / 2, 1.0),
                beta_param=max(pattern.beta_param / 2, 1.0),
                parent_id=pattern.pattern_id,
                created_at=now,
                updated_at=now,
            )
            self._patterns[child_pid] = child
            pattern.children_ids.append(child_pid)

    def _try_merge_children(self, pattern: QueryPattern) -> None:
        """Merge children stats back to parent if they have similar success rates.

        Merges when all children exist and have success rates within epsilon.
        This helps consolidate learning when finer granularity doesn't help.
        """
        if not pattern.children_ids or len(pattern.children_ids) < 2:
            return

        epsilon = 0.05
        children = [self._patterns.get(cid) for cid in pattern.children_ids]
        children = [c for c in children if c is not None and c.sample_count > 0]

        if len(children) < 2:
            return

        success_rates = [c.success_rate for c in children]
        rate_range = max(success_rates) - min(success_rates)

        if rate_range <= epsilon:
            # All children have similar success rates; merge by averaging
            total_samples = sum(c.sample_count for c in children)
            total_success = sum(c.success_count for c in children)
            total_alpha = sum(c.alpha for c in children)
            total_beta = sum(c.beta_param for c in children)

            # Update parent with averaged stats (weighted by sample count)
            if total_samples > 0:
                pattern.success_rate = total_success / total_samples
                pattern.alpha = total_alpha / len(children)
                pattern.beta_param = total_beta / len(children)

    def record_chain(
        self,
        query: str,
        steps: List[Dict[str, str]],
        success: bool,
    ) -> None:
        """Record (or update) a reasoning chain trace."""
        features = self.classify_query(query)
        pid = compute_pattern_id(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
            entity_count=features.get("entity_count", 0),
            hop_hint=features.get("hop_hint", "single"),
        )
        now = datetime.now(timezone.utc).isoformat()
        tid = f"{pid}_chain"

        with self._lock:
            chain = self._chains.get(tid)
            if not chain:
                chain = ReasoningChain(
                    template_id=tid,
                    pattern_id=pid,
                    steps=steps,
                    created_at=now,
                    updated_at=now,
                )
                self._chains[tid] = chain

            chain.total_count += 1
            if success:
                chain.success_count += 1
                chain.steps = steps
            chain.updated_at = now

        self._mark_dirty()

    def get_chain_hint(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Return a reasoning chain template if available and reliable."""
        features = self.classify_query(query)
        pid = compute_pattern_id(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
            entity_count=features.get("entity_count", 0),
            hop_hint=features.get("hop_hint", "single"),
        )
        tid = f"{pid}_chain"

        with self._lock:
            chain = self._chains.get(tid)
            if (chain and chain.success_count >= 2
                    and chain.total_count > 0
                    and chain.success_count / chain.total_count >= 0.4):
                return chain.steps
        return None

    # ── Warmup seeding ────────────────────────────────────────────────

    _DEFAULT_STRATEGIES: List[Dict[str, Any]] = [
        {
            "query_type": "factual", "complexity": "simple",
            "entity_types": ["named_entity"], "entity_count": 1,
            "hop_hint": "single",
            "mode": "DEEP", "params": {"max_loops": 3, "top_k_files": 5},
        },
        {
            "query_type": "bridge", "complexity": "moderate",
            "entity_types": ["named_entity"], "entity_count": 2,
            "hop_hint": "multi",
            "mode": "DEEP", "params": {"max_loops": 5, "top_k_files": 5},
        },
        {
            "query_type": "comparison", "complexity": "moderate",
            "entity_types": ["named_entity"], "entity_count": 2,
            "hop_hint": "multi",
            "mode": "DEEP", "params": {"max_loops": 5, "top_k_files": 5},
        },
        {
            "query_type": "factual", "complexity": "moderate",
            "entity_types": ["named_entity"], "entity_count": 1,
            "hop_hint": "single",
            "mode": "DEEP", "params": {"max_loops": 4, "top_k_files": 5},
        },
        {
            "query_type": "definition", "complexity": "simple",
            "entity_types": [], "entity_count": 0,
            "hop_hint": "single",
            "mode": "FAST", "params": {"max_loops": 2, "top_k_files": 3},
        },
    ]

    def seed_defaults(self) -> int:
        """Insert default strategy patterns if absent (idempotent).

        Returns the number of newly inserted patterns.
        """
        seeded = 0
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for spec in self._DEFAULT_STRATEGIES:
                pid = compute_pattern_id(
                    spec["query_type"], spec["complexity"],
                    spec["entity_types"],
                    entity_count=spec.get("entity_count", 0),
                    hop_hint=spec.get("hop_hint", "single"),
                )
                if pid in self._patterns:
                    continue
                self._patterns[pid] = QueryPattern(
                    pattern_id=pid,
                    query_type=spec["query_type"],
                    complexity=spec["complexity"],
                    entity_types=spec["entity_types"],
                    entity_count=spec.get("entity_count", 0),
                    hop_hint=spec.get("hop_hint", "single"),
                    optimal_mode=spec["mode"],
                    optimal_params=spec["params"],
                    created_at=now,
                    updated_at=now,
                )
                seeded += 1
        if seeded:
            self._mark_dirty()
            logger.debug(f"PatternMemory: seeded {seeded} default patterns")
        return seeded
