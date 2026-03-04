"""ColorOperator — global HSV colour matching (original pHash-era scoring).

Scores hue, saturation, and brightness from the full-image colour moments
against the VLM-compiled constraint ranges.
"""

from __future__ import annotations

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator, hue_score, range_score

_H_MEAN, _S_MEAN, _V_MEAN = 0, 3, 6


class ColorOperator(GrepOperator):
    """Hue / saturation / brightness scoring on global colour moments."""

    @property
    def name(self) -> str:
        return "color"

    @property
    def weight(self) -> float:
        return 1.0

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        m = sig.color_moments
        if len(m) < 9:
            return 0.5

        parts: list[float] = []

        if con.dominant_hues:
            parts.append(hue_score(m[_H_MEAN], con.dominant_hues))

        sat = m[_S_MEAN] / 255.0
        parts.append(range_score(sat, con.saturation_range))

        val = m[_V_MEAN] / 255.0
        parts.append(range_score(val, con.brightness_range))

        return float(sum(parts) / len(parts)) if parts else 0.5

    def is_applicable(self, con: VisualConstraint) -> bool:
        return (
            bool(con.dominant_hues)
            or con.saturation_range != (0.0, 1.0)
            or con.brightness_range != (0.0, 1.0)
        )
