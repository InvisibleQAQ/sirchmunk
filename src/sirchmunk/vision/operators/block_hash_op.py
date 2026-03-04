"""BlockHashOperator — Multi-Scale Block Hashing (MSBH).

Scores spatial awareness: when the constraint specifies a subject position,
the operator checks that the *relevant* grid region is visually distinct
from the background and matches the expected hue.
"""

from __future__ import annotations

from typing import Dict, List

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator, hamming_distance, hue_score

# 3×3 grid layout: [0,1,2 / 3,4,5 / 6,7,8]
_POSITION_CELLS: Dict[str, List[int]] = {
    "top":    [0, 1, 2],
    "bottom": [6, 7, 8],
    "left":   [0, 3, 6],
    "right":  [2, 5, 8],
    "center": [3, 4, 5],
}
_ALL_CELLS = list(range(9))


class BlockHashOperator(GrepOperator):
    """Spatial-aware scoring via 3×3 grid hashing."""

    @property
    def name(self) -> str:
        return "block_hash"

    @property
    def weight(self) -> float:
        return 0.8

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        if len(sig.block_hashes) < 9:
            return 0.5

        pos = con.subject_position.lower().strip()
        relevant = _POSITION_CELLS.get(pos, _ALL_CELLS)
        other = [i for i in _ALL_CELLS if i not in relevant]

        parts: list[float] = []

        # Spatial heterogeneity: relevant region should differ from background
        if other:
            total, count = 0, 0
            for r in relevant:
                for o in other:
                    total += hamming_distance(sig.block_hashes[r], sig.block_hashes[o])
                    count += 1
            avg_dist = total / count if count else 0
            parts.append(min(1.0, avg_dist / 20.0))

        # Region hue match (if hues are specified)
        if con.dominant_hues and len(sig.block_hue_means) >= 9:
            region_hue = sum(sig.block_hue_means[i] for i in relevant) / len(relevant)
            parts.append(hue_score(region_hue, con.dominant_hues))

        return float(sum(parts) / len(parts)) if parts else 0.5

    def is_applicable(self, con: VisualConstraint) -> bool:
        pos = con.subject_position.lower().strip()
        return pos not in ("any", "full", "")
