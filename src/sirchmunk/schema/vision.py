"""Data models for the Sirchmunk-Vision pipeline.

All pipeline stages communicate through these immutable data structures.
Uses the same ``dataclass`` conventions as the rest of ``sirchmunk.schema``
(no Pydantic dependency).

Layers:
    1. Visual Grep    — :class:`ImageSignature`, :class:`VisualConstraint`,
                         :class:`ImageCandidate`
    2. SigLIP2 Scout  — :class:`ScoredCandidate`
    3. VLM Verifier   — :class:`VerificationResult`
    Persistence       — :class:`ImageKnowledge`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Layer 1: Visual Grep
# ---------------------------------------------------------------------------

@dataclass
class ImageSignature:
    """Visual fingerprint computed by ImageSigner.

    Core features (always present):
        phash, color_moments, file_size, width, height, exif

    Enhanced features (for multi-operator Visual Grep):
        block_hashes      — 3x3 grid pHash for spatial awareness
        block_hue_means   — per-cell hue mean for region-level colour matching
        center_moments    — HSV moments of the inner 50 % region
        edge_moments      — HSV moments of the outer 50 % ring
        gist_descriptor   — Gabor-energy GIST scene descriptor (192-d)
        entropy           — Shannon entropy of the grayscale histogram
        edge_density      — fraction of Canny-detected edge pixels
    """

    path: str
    phash: str = ""
    color_moments: List[float] = field(default_factory=list)
    file_size: int = 0
    width: int = 0
    height: int = 0
    exif: Dict[str, Any] = field(default_factory=dict)

    block_hashes: List[str] = field(default_factory=list)
    block_hue_means: List[float] = field(default_factory=list)
    center_moments: List[float] = field(default_factory=list)
    edge_moments: List[float] = field(default_factory=list)
    gist_descriptor: List[float] = field(default_factory=list)
    entropy: float = 0.0
    edge_density: float = 0.0

    siglip_embed: List[float] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualConstraint:
    """Machine-readable visual search parameters compiled by VLM.

    Extended fields (for enhanced operators):
        scene_type         — expected scene category (indoor / outdoor / ...)
        subject_position   — where the subject sits in the frame
        negative_hues      — hues that should NOT dominate
        negative_textures  — texture cues indicating non-matches
        entropy_preference — expected image complexity (low / high / any)
    """

    clip_query: str = ""
    dominant_hues: List[str] = field(default_factory=list)
    saturation_range: Tuple[float, float] = (0.0, 1.0)
    brightness_range: Tuple[float, float] = (0.0, 1.0)
    contrast_level: str = "any"
    texture_keywords: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)
    phash_variance_threshold: float = 0.5
    raw_json: Dict[str, Any] = field(default_factory=dict)

    scene_type: str = "any"
    subject_position: str = "any"
    negative_hues: List[str] = field(default_factory=list)
    negative_textures: List[str] = field(default_factory=list)
    entropy_preference: str = "any"

    expanded_clip_queries: List[str] = field(default_factory=list)

    # Transient (not persisted): pre-computed query text embedding
    clip_query_embed: Optional[Any] = field(default=None, repr=False)


@dataclass
class ImageCandidate:
    """Image that survived Layer-1 Visual Grep filtering."""

    path: str
    signature: ImageSignature
    grep_score: float = 0.0


# ---------------------------------------------------------------------------
# Layer 2: SigLIP2 Scout
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """Candidate with semantic relevance score from SigLIP2 ranking."""

    path: str
    score: float = 0.0
    top_patches: List[Any] = field(default_factory=list)
    round_scores: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 3: VLM Verification
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Output from VLM semantic verification."""

    path: str
    verified: bool = False
    confidence: float = 0.0
    caption: str = ""
    semantic_tags: List[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Knowledge persistence
# ---------------------------------------------------------------------------

@dataclass
class ImageKnowledge:
    """Persistent knowledge entry for a VLM-verified image."""

    path: str
    caption: str = ""
    semantic_tags: List[str] = field(default_factory=list)
    phash: str = ""
    color_moments: List[float] = field(default_factory=list)
    verified_at: str = field(
        default_factory=lambda: datetime.now().isoformat(),
    )
    confidence: float = 0.0
    query_history: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
