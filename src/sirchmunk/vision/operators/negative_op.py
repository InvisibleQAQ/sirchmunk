"""NegativeOperator — penalty scoring for negative constraints.

Returns 1.0 (no penalty) when the image does NOT match any negative cue,
decreasing toward 0.0 as negative signals strengthen.
"""

from __future__ import annotations

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator, HUE_RANGES

_H_MEAN = 0


class NegativeOperator(GrepOperator):
    """Penalise images that match negative hue or texture constraints."""

    @property
    def name(self) -> str:
        return "negative"

    @property
    def weight(self) -> float:
        return 1.5

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        penalty = 0.0

        # Negative hue: penalise if image's dominant hue is unwanted.
        if con.negative_hues and len(sig.color_moments) >= 9:
            hue_mean = sig.color_moments[_H_MEAN]
            for neg_hue in con.negative_hues:
                rng = HUE_RANGES.get(neg_hue.lower().strip())
                if rng is None or rng[0] < 0:
                    continue
                lo, hi = rng
                if neg_hue.lower().strip() == "red" and hue_mean > 160:
                    penalty += 0.3
                    break
                if lo <= hue_mean <= hi:
                    penalty += 0.3
                    break

        # Negative texture: use edge density as proxy for text / grid.
        if con.negative_textures:
            for tex in con.negative_textures:
                tex_l = tex.lower()
                if any(kw in tex_l for kw in ("text", "writing", "caption")):
                    if sig.edge_density > 0.15:
                        penalty += 0.3
                        break
                elif any(kw in tex_l for kw in ("grid", "line", "stripe")):
                    if sig.edge_density > 0.10:
                        penalty += 0.2
                        break

        return max(0.0, 1.0 - penalty)

    def is_applicable(self, con: VisualConstraint) -> bool:
        return bool(con.negative_hues) or bool(con.negative_textures)
