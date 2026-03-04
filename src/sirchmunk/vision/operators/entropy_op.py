"""EntropyOperator — complexity-based pre-pruning.

Uses image Shannon entropy and edge density to quickly separate simple /
clean images from visually complex / cluttered ones.
"""

from __future__ import annotations

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator

# Empirical ranges for "low" and "high" entropy of natural images (bits).
_LOW_ENTROPY = (0.0, 5.5)
_HIGH_ENTROPY = (5.5, 8.0)


class EntropyOperator(GrepOperator):
    """Score based on how well image complexity matches the preference."""

    @property
    def name(self) -> str:
        return "entropy"

    @property
    def weight(self) -> float:
        return 0.8

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        pref = con.entropy_preference.lower().strip()
        e = sig.entropy

        if pref == "low":
            lo, hi = _LOW_ENTROPY
        elif pref == "high":
            lo, hi = _HIGH_ENTROPY
        else:
            return 0.5

        if lo <= e <= hi:
            return 1.0
        dist = min(abs(e - lo), abs(e - hi))
        return max(0.0, 1.0 - dist / 3.0)

    def is_applicable(self, con: VisualConstraint) -> bool:
        return con.entropy_preference.lower().strip() not in ("any", "")
