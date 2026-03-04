"""ContrastiveColorOperator — centre–edge colour contrast.

Distinguishes images where the target hue genuinely dominates the subject
(centre region) from those where it is merely background noise.
"""

from __future__ import annotations

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator, hue_score

_H_MEAN, _S_MEAN = 0, 3


class ContrastiveColorOperator(GrepOperator):
    """Score high when the centre contains the target colour more than edges."""

    @property
    def name(self) -> str:
        return "contrast"

    @property
    def weight(self) -> float:
        return 1.2

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        if len(sig.center_moments) < 9 or len(sig.edge_moments) < 9:
            return 0.5
        if not con.dominant_hues:
            return 0.5

        center_hue = sig.center_moments[_H_MEAN]
        edge_hue = sig.edge_moments[_H_MEAN]
        center_sat = sig.center_moments[_S_MEAN] / 255.0
        edge_sat = sig.edge_moments[_S_MEAN] / 255.0

        center_match = hue_score(center_hue, con.dominant_hues)
        edge_match = hue_score(edge_hue, con.dominant_hues)

        # Reward: target colour strong in centre, weaker at edges
        hue_contrast = max(0.0, min(1.0, 0.5 + (center_match - edge_match)))

        # Saturation contrast: salient subjects are often more saturated
        sat_contrast = max(0.0, min(1.0, 0.5 + (center_sat - edge_sat)))

        return 0.7 * hue_contrast + 0.3 * sat_contrast

    def is_applicable(self, con: VisualConstraint) -> bool:
        return bool(con.dominant_hues)
