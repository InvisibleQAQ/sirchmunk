# Copyright (c) ModelScope Contributors. All rights reserved.
"""Belief-Augmented ReAct (BA-ReAct) state tracker.

Maintains per-file relevance beliefs and provides analytical (train-free)
decision support for the ReAct search agent:

- **Bayesian belief updates** from keyword_search and file_read results.
- **UCB-based file ranking** for exploration–exploitation balance.
- **MCES trigger decisions** for high-value large files.
- **ESS-based stopping signals** when evidence is sufficiently concentrated.
- **Advisory text** injected into the continuation prompt.

All computations are closed-form; no learned parameters or training required.
"""

import math
from typing import Dict, List, Optional, Set, Tuple


class BeliefState:
    """Track and update beliefs about file relevance during a search session.

    Each discovered file is assigned a belief score in [0, 1] representing
    the system's confidence that it contains relevant evidence.  Beliefs are
    updated via lightweight Bayesian rules after each tool observation.

    The UCB (Upper Confidence Bound) ranking balances exploitation (reading
    high-belief files) with exploration (trying under-sampled files), and
    the adaptive exploration coefficient ``c_t = c_0 * budget_ratio``
    naturally reduces exploration as the token budget shrinks.
    """

    # UCB exploration base coefficient
    _UCB_C0: float = 1.0

    # Lazy MCES activation thresholds
    _MCES_BELIEF_THRESHOLD: float = 0.6
    _MCES_SIZE_THRESHOLD: int = 20_000  # characters
    _MCES_BUDGET_MIN: int = 12_000  # tokens

    def __init__(self) -> None:
        self._beliefs: Dict[str, float] = {}
        self._reads: Dict[str, int] = {}
        self._sizes: Dict[str, int] = {}
        self._mces_done: Set[str] = set()
        self._actions: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def file_beliefs(self) -> Dict[str, float]:
        """Snapshot of current file beliefs (read-only copy)."""
        return dict(self._beliefs)

    @property
    def mces_completed_files(self) -> Set[str]:
        """Files that have already undergone deep MCES extraction."""
        return set(self._mces_done)

    # ------------------------------------------------------------------
    # Belief updates
    # ------------------------------------------------------------------

    def update_from_search(self, ranked_paths: List[str]) -> None:
        """Update beliefs from keyword_search discovered files.

        Uses rank-based initialization: higher-ranked files receive
        higher initial beliefs.  For files seen in a previous search,
        beliefs are combined with Bayesian-style aggregation.
        """
        if not ranked_paths:
            return
        for rank, fp in enumerate(ranked_paths):
            new_signal = max(0.2, 0.7 - 0.05 * rank)
            prior = self._beliefs.get(fp)
            if prior is not None:
                # Bayesian combination: 1 - (1 - prior)(1 - signal)
                self._beliefs[fp] = 1 - (1 - prior) * (1 - new_signal)
            else:
                self._beliefs[fp] = new_signal
        self._actions += 1

    def update_from_read(
        self,
        file_id: str,
        content_chars: int,
        found_new_info: bool,
    ) -> None:
        """Update beliefs after a standard file_read.

        Boosts belief if new information was found; applies Bayesian
        exclusion (halves belief) otherwise.
        """
        self._reads[file_id] = self._reads.get(file_id, 0) + 1
        self._sizes[file_id] = content_chars
        if file_id in self._beliefs:
            if found_new_info:
                self._beliefs[file_id] = min(1.0, self._beliefs[file_id] * 1.2)
            else:
                self._beliefs[file_id] *= 0.5
        self._actions += 1

    def update_from_mces(
        self,
        file_id: str,
        best_score: float,
        is_found: bool,
    ) -> None:
        """Update beliefs with MCES LLM-evaluated evidence scores.

        MCES scores (1–10 scale) provide high-confidence relevance
        signals that override the coarser keyword-search priors.
        Also marks the file as read so it no longer appears in
        "promising unread" advisories.
        """
        if is_found and best_score >= 4.0:
            self._beliefs[file_id] = min(1.0, best_score / 10.0)
        else:
            self._beliefs[file_id] = self._beliefs.get(file_id, 0.3) * 0.1
        self._mces_done.add(file_id)
        self._reads[file_id] = self._reads.get(file_id, 0) + 1
        self._actions += 1

    # ------------------------------------------------------------------
    # Decision support
    # ------------------------------------------------------------------

    def should_trigger_mces(
        self,
        file_id: str,
        file_size: int,
        budget_remaining: int,
    ) -> bool:
        """Decide whether to activate Lazy MCES for a file.

        Three conditions must all hold:
        1. High belief (file likely relevant).
        2. Large file (BM25 chunk selection is lossy).
        3. Sufficient budget (MCES consumes ~7–15K tokens).
        """
        if file_id in self._mces_done:
            return False
        if file_id in self._reads:
            return False
        belief = self._beliefs.get(file_id, 0.3)
        return (
            belief >= self._MCES_BELIEF_THRESHOLD
            and file_size > self._MCES_SIZE_THRESHOLD
            and budget_remaining > self._MCES_BUDGET_MIN
        )

    def rank_files_ucb(
        self,
        candidates: List[str],
        budget_ratio: float,
    ) -> List[Tuple[str, float]]:
        """Rank candidate files by UCB score.

        Q_UCB(f) = μ(f) + c_t * √(ln t / max(1, n(f)))

        where μ is the belief, c_t adapts to remaining budget, and n(f)
        is the number of reads for that file.
        """
        t = max(1, self._actions)
        c_t = self._UCB_C0 * max(0.1, budget_ratio)

        ranked: List[Tuple[str, float]] = []
        for fp in candidates:
            mu = self._beliefs.get(fp, 0.3)
            n_i = self._reads.get(fp, 0)
            exploration = c_t * math.sqrt(math.log(t + 1) / max(1, n_i))
            ranked.append((fp, mu + exploration))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Convergence monitoring
    # ------------------------------------------------------------------

    def compute_ess(self) -> float:
        """File-level Effective Sample Size.

        ESS = (Σb)² / Σ(b²).  Low ESS/N indicates beliefs are
        concentrated on few files (evidence convergence).
        """
        beliefs = [b for b in self._beliefs.values() if b > 0]
        if not beliefs:
            return 0.0
        total = sum(beliefs)
        sum_sq = sum(b * b for b in beliefs)
        if sum_sq < 1e-12:
            return 0.0
        return (total * total) / sum_sq

    def should_stop_early(self) -> bool:
        """Heuristic stopping signal based on evidence concentration.

        Returns True when beliefs are concentrated on ≤2 files and
        enough files have been explored to be confident.
        """
        n = len(self._beliefs)
        if n < 3:
            return False
        ess = self.compute_ess()
        n_read = sum(1 for v in self._reads.values() if v > 0)
        return ess < 2.5 and n_read >= 3

    # ------------------------------------------------------------------
    # Advisory signals
    # ------------------------------------------------------------------

    def get_advisory(self) -> str:
        """Format advisory signals for the continuation prompt.

        Returns an empty string when there is nothing actionable to
        report, keeping the prompt clean.
        """
        parts: List[str] = []

        # Highlight high-value unread files
        unread_high = [
            (fp, b)
            for fp, b in self._beliefs.items()
            if self._reads.get(fp, 0) == 0 and b >= 0.4
        ]
        if unread_high:
            unread_high.sort(key=lambda x: x[1], reverse=True)
            names = ", ".join(
                f"{fp.rsplit('/', 1)[-1]}({b:.0%})"
                for fp, b in unread_high[:3]
            )
            parts.append(f"Promising unread: {names}")

        # ESS-based concentration signal
        n = len(self._beliefs)
        if n >= 3:
            ess = self.compute_ess()
            if ess / n < 0.3:
                parts.append("Evidence concentrated — consider answering")

        return " | ".join(parts)
