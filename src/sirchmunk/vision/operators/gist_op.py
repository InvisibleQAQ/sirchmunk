"""GISTOperator — scene-level pre-filtering via GIST spatial envelope.

Derives three meta-features from the 192-d Gabor-energy descriptor and
matches them against heuristic scene-type profiles.  Instantly separates
indoor from outdoor, natural from urban, etc., without any neural network.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator, range_score

# GIST layout: 3 scales × 4 orientations × 16 cells = 192
_N_SCALES = 3
_N_ORIENT = 4
_N_CELLS = 16

# Heuristic scene profiles: (min, max) ranges for the three meta-features.
_SCENE_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "indoor":      {"naturalness": (0.3, 0.7), "openness": (0.0, 0.5), "roughness": (0.2, 0.6)},
    "outdoor":     {"naturalness": (0.3, 1.0), "openness": (0.4, 1.0), "roughness": (0.2, 0.8)},
    "natural":     {"naturalness": (0.6, 1.0), "openness": (0.3, 1.0), "roughness": (0.3, 0.9)},
    "urban":       {"naturalness": (0.0, 0.5), "openness": (0.2, 0.7), "roughness": (0.4, 0.9)},
    "aerial":      {"naturalness": (0.3, 0.8), "openness": (0.6, 1.0), "roughness": (0.1, 0.6)},
    "underwater":  {"naturalness": (0.5, 1.0), "openness": (0.3, 0.8), "roughness": (0.2, 0.7)},
    "studio":      {"naturalness": (0.2, 0.6), "openness": (0.5, 1.0), "roughness": (0.0, 0.4)},
}


class GISTOperator(GrepOperator):
    """Scene-type filtering via GIST Gabor-energy meta-features."""

    @property
    def name(self) -> str:
        return "gist"

    @property
    def weight(self) -> float:
        return 1.0

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        gist = sig.gist_descriptor
        expected = _N_SCALES * _N_ORIENT * _N_CELLS
        if len(gist) < expected:
            return 0.5

        scene = con.scene_type.lower().strip()
        profile = _SCENE_PROFILES.get(scene)
        if profile is None:
            return 0.5

        g = np.array(gist[:expected], dtype=np.float64).reshape(
            _N_SCALES, _N_ORIENT, _N_CELLS,
        )

        # Naturalness: diagonal (45°/135°) vs rectilinear (0°/90°) energy
        diag = g[:, [1, 3], :].mean()
        rect = g[:, [0, 2], :].mean()
        naturalness = float(diag / (diag + rect + 1e-8))

        # Openness: coarse-scale proportion of total energy
        coarse = g[_N_SCALES - 1].mean()
        total_mean = g.mean() + 1e-8
        openness = float(min(1.0, coarse / total_mean / 3.0))

        # Roughness: normalised spatial std of Gabor energy
        roughness = float(min(1.0, g.std() / (total_mean * 2.0)))

        parts = [
            range_score(naturalness, profile["naturalness"]),
            range_score(openness, profile["openness"]),
            range_score(roughness, profile["roughness"]),
        ]
        return float(np.mean(parts))

    def is_applicable(self, con: VisualConstraint) -> bool:
        scene = con.scene_type.lower().strip()
        return scene not in ("any", "")
