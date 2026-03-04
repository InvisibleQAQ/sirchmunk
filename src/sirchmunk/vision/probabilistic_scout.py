"""ProbabilisticScout — Layer 2: SigLIP2-based multi-scale semantic ranking.

Scoring strategies:
  1. **Text-to-image** — encode text queries (+ expanded variants from query
     expansion) via SigLIP2, rank candidates using multi-scale image
     embeddings with max-pooling across all (query, scale) pairs.
  2. **Image-to-image** — encode query images through SigLIP2 vision encoder,
     rank against multi-scale candidate embeddings.
  3. **Hybrid** — weighted combination of text and image similarity scores.

Multi-scale encoding (6 crops per image):
  - Global thumbnail       (full image context)
  - 4 quadrant crops       (TL / TR / BL / BR — local detail)
  - Centre crop            (inner 50 % — subject focus)

Performance features:
  - SigLIP2 model is lazy-loaded on first use.
  - Multi-scale image embeddings are cached to disk; the first search
    pays the encoding cost, subsequent searches only need text encoding
    + a dot product (~0.1 s regardless of image count).
  - Multiple query variants are scored via max-pooling.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from sirchmunk.schema.vision import ImageCandidate, ScoredCandidate

# ------------------------------------------------------------------ #
# Lazy-loaded SigLIP2 singleton (module-level for reuse across calls)
# ------------------------------------------------------------------ #
_model: Any = None
_processor: Any = None
_device: str = "cpu"

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"

_THUMB_SIZE = (224, 224)
_CACHE_VERSION = 3
_N_SCALES = 6  # global, TL, TR, BL, BR, center

# FAST early-stopping parameters
_FAST_CHUNK_SIZE = 100
_FAST_PATIENCE = 2
_FAST_MIN_EVAL_FACTOR = 5   # process at least top_k * this before stopping

# Backward-compatible alias used by external modules
DEFAULT_CLIP_MODEL = DEFAULT_MODEL_ID


def _resolve_model_path(model_id: str) -> str:
    """Resolve model to a local path via ModelScope, falling back to HuggingFace."""
    try:
        from modelscope import snapshot_download
        local_path = snapshot_download(model_id)
        print(f"    [SigLIPScout] Model downloaded via ModelScope: {model_id}")
        return local_path
    except Exception as e:
        print(
            f"    [SigLIPScout] ModelScope unavailable ({e!r}), "
            f"falling back to HuggingFace: {model_id}"
        )
        return model_id


def _ensure_model(model_id: str = DEFAULT_MODEL_ID) -> None:
    """Download and cache the SigLIP2 model on first invocation."""
    global _model, _processor, _device

    if _model is not None:
        return

    print(f"    [SigLIPScout] Loading SigLIP2 model: {model_id} ...")
    import torch
    from transformers import AutoModel, AutoProcessor

    resolved = _resolve_model_path(model_id)
    _processor = AutoProcessor.from_pretrained(resolved)
    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    _model = AutoModel.from_pretrained(resolved).to(_device).eval()
    print(f"    [SigLIPScout] SigLIP2 model loaded on device: {_device}")


def encode_text_query(query: str) -> Any:
    """Encode a single text query if the SigLIP2 model is already loaded.

    Returns an L2-normalised numpy vector ``(embed_dim,)`` or ``None``
    if the model hasn't been loaded yet (cold start).  This allows
    the SigLIPEmbedOperator in Visual Grep to score cached embeddings
    without forcing an early model load.
    """
    if _model is None:
        return None
    import torch
    tokens = _processor(
        text=[query], return_tensors="pt",
        padding="max_length", truncation=True,
    )
    tokens = {k: v.to(_device) for k, v in tokens.items()}
    with torch.no_grad():
        raw = _model.get_text_features(**tokens)
        feat = _extract_features(raw)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]  # (embed_dim,)


def _extract_features(output: Any) -> Any:
    """Extract the feature tensor from a model output.

    ``get_text_features`` / ``get_image_features`` may return a raw
    ``torch.Tensor`` (newer transformers) or a ``BaseModelOutput*``
    wrapper (older versions / some architectures).  This helper
    normalises both cases to a plain tensor.
    """
    import torch
    if isinstance(output, torch.Tensor):
        return output
    # BaseModelOutputWithPooling — pooled_output is at index [1]
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output[1]


# ------------------------------------------------------------------ #
# Multi-scale crop generation
# ------------------------------------------------------------------ #

def _generate_crops(
    img: Image.Image,
    size: Tuple[int, int] = _THUMB_SIZE,
    global_only: bool = False,
) -> List[Image.Image]:
    """Generate image crops for SigLIP2 encoding.

    When *global_only* is ``True`` (FAST mode), returns a single resized
    thumbnail — 6× cheaper than the full multi-scale crop set.
    """
    if global_only:
        thumb = img.copy()
        thumb.thumbnail(size, Image.LANCZOS)
        return [thumb.convert("RGB")]

    w, h = img.size
    half_w, half_h = max(w // 2, 1), max(h // 2, 1)
    qw, qh = max(w // 4, 1), max(h // 4, 1)

    regions = [
        (0, 0, w, h),                      # global
        (0, 0, half_w, half_h),             # top-left
        (half_w, 0, w, half_h),             # top-right
        (0, half_h, half_w, h),             # bottom-left
        (half_w, half_h, w, h),             # bottom-right
        (qw, qh, w - qw, h - qh),          # centre (inner 50 %)
    ]

    crops: List[Image.Image] = []
    for box in regions:
        crop = img.crop(box)
        crop.thumbnail(size, Image.LANCZOS)
        crops.append(crop.convert("RGB"))
    return crops


# ------------------------------------------------------------------ #
# Multi-scale embedding cache
# ------------------------------------------------------------------ #

class _EmbeddingCache:
    """File-backed cache: image path → (N_SCALES, embed_dim) L2-normalised."""

    def __init__(self, cache_dir: str, model_id: str):
        os.makedirs(cache_dir, exist_ok=True)
        safe = model_id.replace("/", "_").replace("\\", "_")
        self._path = os.path.join(
            cache_dir, f"siglip_embeds_{safe}_v{_CACHE_VERSION}.pkl",
        )
        self._data: Dict[str, np.ndarray] = {}
        self._dirty = False
        if os.path.exists(self._path):
            try:
                with open(self._path, "rb") as f:
                    self._data = pickle.load(f)
                print(
                    f"    [SigLIPCache] Loaded {len(self._data)} "
                    f"cached multi-scale embeddings"
                )
            except Exception:
                self._data = {}

    def lookup(
        self,
        paths: List[str],
        expected_dim: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Return ``(cached_dict, uncached_list)``.

        Cached values are ``(N_SCALES, embed_dim)`` matrices.  Entries
        with wrong shape or dimension are silently discarded.
        """
        cached: Dict[str, np.ndarray] = {}
        uncached: List[str] = []
        for p in paths:
            vec = self._data.get(p)
            if (
                vec is not None
                and vec.ndim == 2
                and vec.shape[0] == _N_SCALES
                and (expected_dim == 0 or vec.shape[1] == expected_dim)
            ):
                cached[p] = vec
            else:
                uncached.append(p)
        return cached, uncached

    def store(self, embeddings: Dict[str, np.ndarray]) -> None:
        self._data.update(embeddings)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        with open(self._path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._dirty = False
        print(f"    [SigLIPCache] Persisted {len(self._data)} embeddings to disk")


# ------------------------------------------------------------------ #
# ProbabilisticScout
# ------------------------------------------------------------------ #

class ProbabilisticScout:
    """SigLIP2-based image ranking with multi-scale encoding and query expansion.

    On cold start, multi-scale image embeddings (6 crops per image) are
    computed in batches and cached to disk.  Warm searches only need text
    encoding + a matrix dot product.
    """

    def __init__(
        self,
        patch_size: int = 224,
        base_patches: int = 2,
        survival_ratio: float = 0.5,
        max_rounds: int = 4,
        clip_model_id: str = DEFAULT_MODEL_ID,
        clip_batch_size: int = 32,
        cache_dir: str = "",
    ):
        self.patch_size = patch_size
        self.base_patches = base_patches
        self.survival_ratio = survival_ratio
        self.max_rounds = max_rounds
        self._model_id = clip_model_id
        self._batch_size = clip_batch_size
        self._cache: Optional[_EmbeddingCache] = None
        if cache_dir:
            self._cache = _EmbeddingCache(cache_dir, clip_model_id)

    # ------------------------------------------------------------------ #
    # Text encoding
    # ------------------------------------------------------------------ #

    def _encode_texts(self, queries: List[str]) -> np.ndarray:
        """Encode text queries → ``(Q, embed_dim)`` L2-normalised matrix."""
        import torch

        _ensure_model(self._model_id)
        tokens = _processor(
            text=queries,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        tokens = {k: v.to(_device) for k, v in tokens.items()}
        with torch.no_grad():
            raw = _model.get_text_features(**tokens)
            text_feat = _extract_features(raw)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Image encoding (multi-scale)
    # ------------------------------------------------------------------ #

    def _get_candidate_multiscale(
        self,
        candidates: List[ImageCandidate],
        embed_dim: int,
        label: str = "",
        global_only: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Load cached + compute embeddings for candidates.

        When *global_only* is ``True``, only the global thumbnail is
        encoded (1 crop vs 6), yielding ``(1, embed_dim)`` matrices for
        uncached images.  Cached multi-scale entries are returned as-is.

        Returns:
            ``{path: np.ndarray(S, embed_dim)}`` where S is 1 or N_SCALES.
        """
        import torch

        all_paths = [c.path for c in candidates]
        if self._cache is not None:
            cached_embeds, uncached_paths = self._cache.lookup(
                all_paths, expected_dim=embed_dim,
            )
        else:
            cached_embeds, uncached_paths = {}, list(all_paths)

        n_crops = 1 if global_only else _N_SCALES
        tag = f" ({label})" if label else ""
        mode_tag = "global-only" if global_only else "multi-scale"
        print(
            f"    [SigLIPScout]{tag} Embeddings: {len(cached_embeds)} cached, "
            f"{len(uncached_paths)} to compute ({mode_tag})"
        )

        if not uncached_paths:
            return dict(cached_embeds)

        new_embeds: Dict[str, np.ndarray] = {}

        # Flatten all crops into one big list for efficient batching
        batch_crops: List[Image.Image] = []
        valid_paths: List[str] = []
        crop_offsets: List[int] = []

        for p in uncached_paths:
            try:
                img = Image.open(p)
                crops = _generate_crops(img, _THUMB_SIZE, global_only=global_only)
                crop_offsets.append(len(batch_crops))
                batch_crops.extend(crops)
                valid_paths.append(p)
            except Exception:
                continue

        if not batch_crops:
            return dict(cached_embeds)

        # Encode all crops in batches
        all_feats: List[np.ndarray] = []
        bs = self._batch_size
        total_batches = (len(batch_crops) + bs - 1) // bs

        for bi in range(total_batches):
            start = bi * bs
            crop_batch = batch_crops[start: start + bs]
            pixel = _processor(images=crop_batch, return_tensors="pt")
            with torch.no_grad():
                raw = _model.get_image_features(
                    pixel_values=pixel["pixel_values"].to(_device),
                )
                feat = _extract_features(raw)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_feats.append(feat.cpu().numpy())

            if (bi + 1) % 10 == 0 or bi == total_batches - 1:
                print(
                    f"    [SigLIPScout] crop batch {bi + 1}/{total_batches}: "
                    f"{len(crop_batch)} crops"
                )

        if not all_feats:
            return dict(cached_embeds)

        feat_matrix = np.concatenate(all_feats, axis=0)  # (total_crops, D)

        for i, path in enumerate(valid_paths):
            offset = crop_offsets[i]
            end = offset + n_crops
            if end <= feat_matrix.shape[0]:
                new_embeds[path] = feat_matrix[offset:end]

        # Only persist full multi-scale embeddings (don't pollute cache
        # with global-only entries that would fail shape validation).
        if new_embeds and self._cache is not None and not global_only:
            self._cache.store(new_embeds)
            self._cache.flush()

        return {**cached_embeds, **new_embeds}

    def _encode_images_global(
        self,
        pil_images: List[Image.Image],
    ) -> np.ndarray:
        """Encode PIL Images at global scale → ``(N, embed_dim)``."""
        import torch

        _ensure_model(self._model_id)
        feats: List[np.ndarray] = []
        for img in pil_images:
            thumb = img.copy()
            thumb.thumbnail(_THUMB_SIZE, Image.LANCZOS)
            px = _processor(
                images=[thumb.convert("RGB")], return_tensors="pt",
            )
            with torch.no_grad():
                raw = _model.get_image_features(
                    pixel_values=px["pixel_values"].to(_device),
                )
                feat = _extract_features(raw)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.cpu().numpy()[0])
        return np.stack(feats)

    # ------------------------------------------------------------------ #
    # FAST: global-only encoding helper
    # ------------------------------------------------------------------ #

    def _encode_global_batch(
        self,
        paths: List[str],
    ) -> Dict[str, np.ndarray]:
        """Encode global thumbnails for a list of image paths.

        Returns ``{path: np.ndarray(embed_dim,)}`` — one vector per image.
        Used by :meth:`_batch_rank_fast` for uncached candidates.
        """
        import torch

        imgs: List[Image.Image] = []
        valid: List[str] = []
        for p in paths:
            try:
                img = Image.open(p)
                img.thumbnail(_THUMB_SIZE, Image.LANCZOS)
                imgs.append(img.convert("RGB"))
                valid.append(p)
            except Exception:
                continue

        if not imgs:
            return {}

        result: Dict[str, np.ndarray] = {}
        bs = self._batch_size
        for bi in range(0, len(imgs), bs):
            batch = imgs[bi: bi + bs]
            batch_paths = valid[bi: bi + bs]
            px = _processor(images=batch, return_tensors="pt")
            with torch.no_grad():
                raw = _model.get_image_features(
                    pixel_values=px["pixel_values"].to(_device),
                )
                feat = _extract_features(raw)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            feats = feat.cpu().numpy()
            for j, p in enumerate(batch_paths):
                result[p] = feats[j]
        return result

    # ------------------------------------------------------------------ #
    # FAST: greedy ranking with early stopping
    # ------------------------------------------------------------------ #

    def _batch_rank_fast(
        self,
        candidates: List[ImageCandidate],
        queries: List[str],
        top_k: int = 9,
    ) -> List[ScoredCandidate]:
        """FAST greedy ranking: global-crop-only + batch-wise early stopping.

        Candidates are assumed pre-sorted by ``grep_score`` (descending).
        Processes them in chunks of ``_FAST_CHUNK_SIZE``; stops when
        ``_FAST_PATIENCE`` consecutive chunks fail to improve the running
        top-k, provided at least ``top_k * _FAST_MIN_EVAL_FACTOR``
        candidates have been evaluated.
        """
        import torch

        _ensure_model(self._model_id)
        t0 = time.time()

        text_matrix = self._encode_texts(queries)  # (Q, D)
        embed_dim = text_matrix.shape[1]
        print(
            f"    [SigLIPScout] Encoded {len(queries)} query variant(s), "
            f"dim={embed_dim}"
        )

        # Bulk cache lookup — extract global row from cached multi-scale
        cached_global: Dict[str, np.ndarray] = {}
        if self._cache is not None:
            cached_ms, _ = self._cache.lookup(
                [c.path for c in candidates], expected_dim=embed_dim,
            )
            cached_global = {p: ms[0] for p, ms in cached_ms.items()}

        n_to_compute = len(candidates) - len(cached_global)
        print(
            f"    [SigLIPScout] (fast) {len(cached_global)} cached, "
            f"{n_to_compute} to compute (global-only)"
        )

        top_scores: Dict[str, float] = {}
        min_top = float("-inf")
        stale = 0
        total_encoded = 0
        min_before_stop = min(
            top_k * _FAST_MIN_EVAL_FACTOR, len(candidates),
        )
        processed = 0

        for ci in range(0, len(candidates), _FAST_CHUNK_SIZE):
            chunk = candidates[ci: ci + _FAST_CHUNK_SIZE]
            improved = False

            # Separate cached vs. uncached within this chunk
            to_encode: List[str] = []
            chunk_embeds: Dict[str, np.ndarray] = {}
            for c in chunk:
                g = cached_global.get(c.path)
                if g is not None:
                    chunk_embeds[c.path] = g
                else:
                    to_encode.append(c.path)

            if to_encode:
                new = self._encode_global_batch(to_encode)
                chunk_embeds.update(new)
                total_encoded += len(new)

            # Score each candidate
            for c in chunk:
                emb = chunk_embeds.get(c.path)
                if emb is None:
                    continue
                score = float((text_matrix @ emb).max())

                if len(top_scores) < top_k:
                    top_scores[c.path] = score
                    improved = True
                    if len(top_scores) >= top_k:
                        min_top = min(top_scores.values())
                elif score > min_top:
                    worst = min(top_scores, key=top_scores.get)
                    del top_scores[worst]
                    top_scores[c.path] = score
                    min_top = min(top_scores.values())
                    improved = True

            processed = ci + len(chunk)
            stale = 0 if improved else stale + 1

            if (
                stale >= _FAST_PATIENCE
                and len(top_scores) >= top_k
                and processed >= min_before_stop
            ):
                print(
                    f"    [SigLIPScout] Early stop at "
                    f"{processed}/{len(candidates)} "
                    f"(encoded {total_encoded} new)"
                )
                break

        elapsed = time.time() - t0
        scored = [
            ScoredCandidate(path=p, score=s, round_scores=[s])
            for p, s in top_scores.items()
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        print(
            f"    [SigLIPScout] Ranked {processed} images (fast) "
            f"in {elapsed:.1f}s"
        )
        return scored

    # ------------------------------------------------------------------ #
    # Text-to-image ranking (multi-query × multi-scale)
    # ------------------------------------------------------------------ #

    def _batch_rank(
        self,
        candidates: List[ImageCandidate],
        queries: List[str],
    ) -> List[ScoredCandidate]:
        """Rank candidates by max-pooled SigLIP2 similarity.

        ``score = max_{q in queries, s in scales} sim(q, image_s)``
        """
        import torch

        _ensure_model(self._model_id)
        t0 = time.time()

        text_matrix = self._encode_texts(queries)   # (Q, D)
        embed_dim = text_matrix.shape[1]
        print(
            f"    [SigLIPScout] Encoded {len(queries)} query variant(s), "
            f"dim={embed_dim}"
        )

        all_embeds = self._get_candidate_multiscale(
            candidates, embed_dim, label="text-search",
        )

        scored: List[ScoredCandidate] = []
        for c in candidates:
            ms_embed = all_embeds.get(c.path)
            if ms_embed is not None:
                sim_matrix = text_matrix @ ms_embed.T     # (Q, 6)
                best_score = float(sim_matrix.max())
                global_score = float(sim_matrix[:, 0].max())
                scored.append(ScoredCandidate(
                    path=c.path,
                    score=best_score,
                    round_scores=[global_score, best_score],
                ))

        elapsed = time.time() - t0
        scored.sort(key=lambda s: s.score, reverse=True)
        print(f"    [SigLIPScout] Ranked {len(scored)} images in {elapsed:.1f}s")
        return scored

    # ------------------------------------------------------------------ #
    # Image-to-image ranking (multi-scale)
    # ------------------------------------------------------------------ #

    def _image_rank(
        self,
        candidates: List[ImageCandidate],
        query_images: List[Image.Image],
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank by SigLIP2 image-to-image similarity (multi-scale on candidates)."""
        _ensure_model(self._model_id)
        t0 = time.time()

        query_matrix = self._encode_images_global(query_images)  # (Nq, D)
        embed_dim = query_matrix.shape[1]

        all_embeds = self._get_candidate_multiscale(
            candidates, embed_dim, label="image-search",
            global_only=fast,
        )

        scored: List[ScoredCandidate] = []
        for c in candidates:
            ms_embed = all_embeds.get(c.path)
            if ms_embed is not None:
                sim_matrix = query_matrix @ ms_embed.T   # (Nq, 6)
                sim = float(sim_matrix.max())
                scored.append(ScoredCandidate(
                    path=c.path, score=sim, round_scores=[sim],
                ))

        elapsed = time.time() - t0
        scored.sort(key=lambda s: s.score, reverse=True)
        print(
            f"    [SigLIPScout] Image-to-image ranked "
            f"{len(scored)} in {elapsed:.1f}s"
        )
        return scored

    # ------------------------------------------------------------------ #
    # Hybrid ranking (multi-scale + multi-query)
    # ------------------------------------------------------------------ #

    def _hybrid_rank(
        self,
        candidates: List[ImageCandidate],
        text_queries: List[str],
        image_queries: List[Image.Image],
        text_weight: float = 0.5,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Weighted combination of text and image SigLIP2 scores (multi-scale)."""
        _ensure_model(self._model_id)
        t0 = time.time()

        text_matrix = self._encode_texts(text_queries)       # (Qt, D)
        image_matrix = self._encode_images_global(image_queries)  # (Qi, D)
        embed_dim = text_matrix.shape[1]

        all_embeds = self._get_candidate_multiscale(
            candidates, embed_dim, label="hybrid",
            global_only=fast,
        )

        img_w = 1.0 - text_weight
        scored: List[ScoredCandidate] = []
        for c in candidates:
            ms_embed = all_embeds.get(c.path)
            if ms_embed is not None:
                t_sim = float((text_matrix @ ms_embed.T).max())
                i_sim = float((image_matrix @ ms_embed.T).max())
                combined = text_weight * t_sim + img_w * i_sim
                scored.append(ScoredCandidate(
                    path=c.path, score=combined,
                    round_scores=[t_sim, i_sim],
                ))

        elapsed = time.time() - t0
        scored.sort(key=lambda s: s.score, reverse=True)
        print(f"    [SigLIPScout] Hybrid ranked {len(scored)} in {elapsed:.1f}s")
        return scored

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def scout(
        self,
        candidates: List[ImageCandidate],
        query: str,
        top_k: int = 10,
        expanded_queries: Optional[List[str]] = None,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank candidates using SigLIP2 scoring with query expansion.

        When *fast* is ``True``, uses global-crop-only encoding with
        batch-wise early stopping — typically 6-10× faster than the
        full multi-scale pipeline.

        Args:
            expanded_queries: Additional semantic variants of *query* for
                max-pooled scoring.  If ``None``, only the primary query
                is used.
            fast: Use greedy ranking with early stopping (FAST mode).

        Returns:
            Up to *top_k* :class:`ScoredCandidate`, sorted by score desc.
        """
        if not candidates:
            return []

        queries = [query]
        if expanded_queries:
            queries.extend(expanded_queries)

        mode = "fast (global-only + early-stop)" if fast else "full (multi-scale)"
        print(
            f"  [SigLIPScout] {mode} ranking: {len(candidates)} "
            f"candidates, {len(queries)} query variant(s) → top {top_k}"
        )

        if fast:
            ranked = await asyncio.to_thread(
                self._batch_rank_fast, candidates, queries, top_k,
            )
        else:
            ranked = await asyncio.to_thread(
                self._batch_rank, candidates, queries,
            )

        if not ranked:
            print("  [SigLIPScout] No candidates scored")
            return []

        top_score = ranked[0].score
        kth = min(top_k - 1, len(ranked) - 1)
        print(
            f"  [SigLIPScout] Done: #1={os.path.basename(ranked[0].path)} "
            f"({top_score:.3f}), #{kth + 1}={ranked[kth].score:.3f}"
        )
        return ranked[:top_k]

    async def scout_by_image(
        self,
        candidates: List[ImageCandidate],
        query_images: List[Image.Image],
        top_k: int = 10,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank candidates by SigLIP2 image-to-image similarity."""
        if not candidates:
            return []

        mode = "fast" if fast else "full"
        print(
            f"  [SigLIPScout] Image-to-image ranking ({mode}): "
            f"{len(candidates)} candidates → top {top_k}"
        )

        ranked = await asyncio.to_thread(
            self._image_rank, candidates, query_images, fast,
        )

        if not ranked:
            print("  [SigLIPScout] No candidates scored")
            return []

        top_score = ranked[0].score
        kth = min(top_k - 1, len(ranked) - 1)
        print(
            f"  [SigLIPScout] Done: #1={os.path.basename(ranked[0].path)} "
            f"({top_score:.3f}), #{kth + 1}={ranked[kth].score:.3f}"
        )
        return ranked[:top_k]

    async def scout_hybrid(
        self,
        candidates: List[ImageCandidate],
        text_query: str,
        image_queries: List[Image.Image],
        top_k: int = 10,
        text_weight: float = 0.5,
        expanded_queries: Optional[List[str]] = None,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank by combined text + image SigLIP2 similarity."""
        if not candidates:
            return []

        text_queries = [text_query]
        if expanded_queries:
            text_queries.extend(expanded_queries)

        mode = "fast" if fast else "full"
        print(
            f"  [SigLIPScout] Hybrid ranking ({mode}): "
            f"{len(candidates)} candidates, "
            f"text={len(text_queries)} variants, "
            f"images={len(image_queries)} → top {top_k}"
        )

        ranked = await asyncio.to_thread(
            self._hybrid_rank,
            candidates, text_queries, image_queries, text_weight, fast,
        )

        if not ranked:
            print("  [SigLIPScout] No candidates scored")
            return []

        top = ranked[0]
        kth = min(top_k - 1, len(ranked) - 1)
        print(
            f"  [SigLIPScout] Done: #1={os.path.basename(top.path)} "
            f"({top.score:.3f}), #{kth + 1}={ranked[kth].score:.3f}"
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------ #
    # Legacy MCS patch methods (kept for localisation tasks)
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_saliency(image: np.ndarray) -> np.ndarray:
        """Gradient-magnitude saliency map normalised to [0, 1]."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        peak = mag.max()
        return (mag / peak) if peak > 0 else mag

    def sample_patches(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        n_patches: int,
    ) -> List[np.ndarray]:
        """Sample *n_patches* from *image*, weighted by *saliency*."""
        h, w = image.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            return [cv2.resize(image, (ps, ps))]
        valid_h, valid_w = h - ps, w - ps
        if valid_h <= 0 or valid_w <= 0:
            return [cv2.resize(image, (ps, ps))]

        grid_h = min(valid_h, 50)
        grid_w = min(valid_w, 50)
        sal_grid = cv2.resize(
            saliency[:valid_h, :valid_w], (grid_w, grid_h),
        )
        flat = sal_grid.flatten() + 1e-8
        prob = flat / flat.sum()

        n_samples = min(n_patches, len(prob))
        indices = np.random.choice(
            len(prob), size=n_samples, replace=False, p=prob,
        )

        patches: List[np.ndarray] = []
        for idx in indices:
            gy, gx = divmod(int(idx), grid_w)
            y = int(gy * valid_h / grid_h)
            x = int(gx * valid_w / grid_w)
            patch = image[y: y + ps, x: x + ps]
            if patch.shape[0] == ps and patch.shape[1] == ps:
                patches.append(patch)
        return patches
