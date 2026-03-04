"""SigLIPEmbedOperator — high-level semantic scoring via cached SigLIP2 embeddings.

Activates only when the image has a pre-computed ``siglip_embed`` (backfilled
after the first search's Phase 2) and the query's text embedding is available
on the constraint.  When active, this operator dominates the fusion score
(weight=5.0), giving Visual Grep near-SigLIP semantic discrimination.

On the first search (cold start), this operator is inactive and the
pipeline falls back to the original 7 low-level operators.
"""

from __future__ import annotations

import numpy as np

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator


class SigLIPEmbedOperator(GrepOperator):
    """Cosine similarity between cached image embedding and query text embedding."""

    @property
    def name(self) -> str:
        return "siglip_embed"

    @property
    def weight(self) -> float:
        return 5.0

    def is_applicable(self, con: VisualConstraint) -> bool:
        return con.clip_query_embed is not None

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        if not sig.siglip_embed or con.clip_query_embed is None:
            return 0.5

        img_vec = np.asarray(sig.siglip_embed, dtype=np.float32)
        txt_vec = np.asarray(con.clip_query_embed, dtype=np.float32)

        norm_i = np.linalg.norm(img_vec)
        norm_t = np.linalg.norm(txt_vec)
        if norm_i < 1e-8 or norm_t < 1e-8:
            return 0.5

        # Cosine similarity in [-1, 1] → rescale to [0, 1]
        cosine = float(np.dot(img_vec, txt_vec) / (norm_i * norm_t))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
