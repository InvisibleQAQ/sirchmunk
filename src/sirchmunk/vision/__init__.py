"""Sirchmunk Vision — indexless, self-evolving image search.

Three-layer pipeline:
  Layer 1 — Visual Grep:  pHash + colour filtering     (CPU, milliseconds)
                           + persistent signature cache  (DuckDB, mtime-based)
                           + parallel operator scoring   (ThreadPoolExecutor)
  Layer 2 — SigLIP2:      multi-scale + query expansion (CPU/GPU, seconds)
  Layer 3 — VLM Verifier: adaptive collage verification (API, single call)

Self-evolution:
  Agent Memory — episodic search logs, path-pattern learning, query
                  clustering, pipeline statistics.

Install extra dependencies via::

    pip install sirchmunk[vision]
"""

from __future__ import annotations


def _check_deps() -> None:
    """Raise a clear error if vision extras are missing."""
    missing = []
    for pkg in ("cv2", "imagehash", "torch", "transformers"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Vision dependencies not installed: {missing}. "
            "Install them with: pip install sirchmunk[vision]"
        )


_check_deps()

from .vision_search import VisionSearch, VisionSearchResult
from sirchmunk.llm.vlm_chat import VLMClient, VLMResponse
from sirchmunk.memory.agent_memory import AgentMemory, SearchAdvice, SearchEpisode

__all__ = [
    "VisionSearch",
    "VisionSearchResult",
    "VLMClient",
    "VLMResponse",
    "AgentMemory",
    "SearchAdvice",
    "SearchEpisode",
]
