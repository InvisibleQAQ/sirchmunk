"""Abstract base class and shared helpers for Visual Grep operators.

Every operator receives an :class:`ImageSignature` (pre-computed features)
and a :class:`VisualConstraint` (VLM-compiled search parameters) and returns
a relevance score in [0, 1].  Operators declare their own weight and can
signal non-applicability to be excluded from the fusion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from sirchmunk.schema.vision import ImageSignature, VisualConstraint

# ------------------------------------------------------------------ #
# Named-colour → OpenCV HSV hue range (H: 0-180)
# ------------------------------------------------------------------ #
HUE_RANGES: Dict[str, Tuple[int, int]] = {
    "red":    (0, 10),
    "orange": (10, 25),
    "yellow": (25, 35),
    "green":  (35, 85),
    "cyan":   (85, 100),
    "blue":   (100, 130),
    "purple": (130, 155),
    "pink":   (155, 170),
    "white":  (-1, -1),
    "gray":   (-1, -1),
    "grey":   (-1, -1),
    "black":  (-1, -1),
}


# ------------------------------------------------------------------ #
# Shared scoring helpers
# ------------------------------------------------------------------ #

def range_score(value: float, rng: Tuple[float, float]) -> float:
    """1.0 if *value* is inside *rng*, decaying linearly outside."""
    lo, hi = rng
    if lo <= value <= hi:
        return 1.0
    dist = min(abs(value - lo), abs(value - hi))
    return max(0.0, 1.0 - dist * 3.0)


def hue_score(hue_mean: float, target_hues: List[str]) -> float:
    """Score how well *hue_mean* (0-180) matches a list of named colours."""
    best = 0.0
    for name in target_hues:
        rng = HUE_RANGES.get(name.lower().strip())
        if rng is None:
            continue
        lo, hi = rng
        if lo < 0:
            # Achromatic colour — check saturation externally, skip hue.
            continue
        if name.lower().strip() == "red" and hue_mean > 160:
            best = max(best, max(0.0, 1.0 - abs(hue_mean - 175) / 20.0))
            continue
        if lo <= hue_mean <= hi:
            return 1.0
        dist = min(abs(hue_mean - lo), abs(hue_mean - hi))
        best = max(best, max(0.0, 1.0 - dist / 30.0))
    return best if best > 0.0 else 0.3


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hexadecimal hash strings."""
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except (ValueError, TypeError):
        return 64


# ------------------------------------------------------------------ #
# Operator ABC
# ------------------------------------------------------------------ #

class GrepOperator(ABC):
    """Base class for all Visual Grep scoring operators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier shown in logs."""

    @property
    def weight(self) -> float:
        """Relative weight in the multi-operator fusion score."""
        return 1.0

    @abstractmethod
    def score(
        self,
        sig: ImageSignature,
        con: VisualConstraint,
    ) -> float:
        """Return a [0, 1] relevance score for *sig* given *con*."""

    def is_applicable(self, con: VisualConstraint) -> bool:
        """Return ``False`` to opt out of the fusion for this constraint."""
        return True
