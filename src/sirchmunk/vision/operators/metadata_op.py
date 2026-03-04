"""MetadataOperator — filename / path keyword overlap scoring."""

from __future__ import annotations

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from .base import GrepOperator


class MetadataOperator(GrepOperator):
    """High-confidence scoring based on semantic tag presence in file path."""

    @property
    def name(self) -> str:
        return "metadata"

    @property
    def weight(self) -> float:
        return 1.5

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        if not con.semantic_tags:
            return 0.5
        lower = sig.path.lower()
        hits = sum(1 for t in con.semantic_tags if t.lower() in lower)
        return min(1.0, hits / len(con.semantic_tags))

    def is_applicable(self, con: VisualConstraint) -> bool:
        return bool(con.semantic_tags)
