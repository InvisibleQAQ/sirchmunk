"""BatchVerifier — Layer 3: VLM semantic verification.

Verification strategies:
  1. **Adaptive collage mode** — automatically selects optimal grid:
       - ≤4 candidates → 2×2 (larger per-image resolution = higher accuracy)
       - ≤9 candidates → 3×3 (current default)
       - >9 candidates → multiple collage calls
  2. **Refinement mode** — send a single full-resolution image (or its
     highest-scoring evidence patches) for detailed verification.

Verified images receive structured captions and semantic tags that are
persisted into the KnowledgeStore for future text-based retrieval.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from sirchmunk.schema.vision import ScoredCandidate, VerificationResult
from sirchmunk.llm.vlm_chat import VLMClient

# ------------------------------------------------------------------ #
# Prompt templates
# ------------------------------------------------------------------ #

_COLLAGE_PROMPT = """\
You are a strict image search verification engine.  Your job is to \
determine whether each candidate image genuinely matches the user's \
search query.  Be PRECISE — only mark an image as matching if the \
queried subject/object/scene is clearly and unambiguously visible.

Search query: "{query}"

Below is a {grid}×{grid} grid of {n} candidate images numbered 1–{n} \
(left-to-right, top-to-bottom).

For EACH image, evaluate along these dimensions:
  A. Subject presence — Is the specific subject/object from the query \
clearly visible?  (e.g. if query asks for "passenger airplane", a \
bird or helicopter does NOT count.)
  B. Scene relevance — Does the overall scene match the query context?
  C. Prominence — Is the subject a prominent part of the image, not \
just a tiny background element?

Confidence calibration:
  0.90–1.00 = Subject is the main focus, perfect match
  0.70–0.89 = Subject clearly present, good match
  0.50–0.69 = Subject likely present but small, partially occluded, \
or ambiguous
  0.30–0.49 = Weak / uncertain — tangentially related at best
  0.00–0.29 = No match — queried subject is absent

Rules:
  - If the queried subject is NOT visible → match=false, confidence<0.30
  - If unsure whether the subject is present → match=false
  - Do NOT inflate confidence — be conservative

Respond with ONLY a JSON array (no markdown, no explanation):
[
  {{"index": 1, "match": true, "confidence": 0.92, "reason": "one-sentence why", "caption": "descriptive sentence", "tags": ["tag1"]}},
  ...
]"""

_REFINE_PROMPT = """\
You are a strict image search verification engine.

Search query: "{query}"

Evaluate this single image:
  A. Subject presence — Is the queried subject clearly visible?
  B. Scene relevance — Does the scene match the query?
  C. Prominence — Is the subject prominent (not a tiny background element)?

Confidence: 0.9–1.0 = perfect, 0.7–0.89 = good, 0.5–0.69 = partial, \
<0.5 = no match.  Be conservative — only match=true if confident.

Respond with ONLY a JSON object (no markdown):
{{"match": true, "confidence": 0.92, "reason": "why this matches or not", "caption": "...", "tags": ["..."]}}"""


class BatchVerifier:
    """VLM-powered semantic verification with collage batching."""

    def __init__(
        self,
        vlm: VLMClient,
        cell_size: int = 512,
        min_confidence: float = 0.50,
    ):
        self._vlm = vlm
        self._cell_size = cell_size
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------ #
    # Collage construction
    # ------------------------------------------------------------------ #

    def create_collage(
        self,
        image_paths: List[str],
        grid_size: int = 3,
    ) -> Image.Image:
        """Tile images into an N×N grid with index labels.

        Empty cells are filled with neutral gray.  Each image is
        centre-fitted and labelled with its 1-based index number.
        """
        cs = self._cell_size
        canvas = Image.new(
            "RGB", (cs * grid_size, cs * grid_size), (128, 128, 128),
        )
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", size=28)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for idx, path in enumerate(image_paths[: grid_size ** 2]):
            row, col = divmod(idx, grid_size)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((cs, cs), Image.LANCZOS)
                x_off = col * cs + (cs - img.width) // 2
                y_off = row * cs + (cs - img.height) // 2
                canvas.paste(img, (x_off, y_off))
            except Exception:
                pass

            # Draw 1-based index label
            label = str(idx + 1)
            lx = col * cs + 8
            ly = row * cs + 6
            # Shadow for readability
            draw.text((lx + 1, ly + 1), label, fill="black", font=font)
            draw.text((lx, ly), label, fill="white", font=font)

        return canvas

    # ------------------------------------------------------------------ #
    # Batch verification (collage)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _choose_grid(n: int) -> int:
        """Select optimal grid size based on candidate count."""
        if n <= 4:
            return 2
        return 3

    async def verify_batch(
        self,
        candidates: List[ScoredCandidate],
        query: str,
        grid_size: Optional[int] = None,
    ) -> List[VerificationResult]:
        """Verify candidates via adaptive collage(s).

        Grid size is chosen automatically unless overridden:
          - ≤4 candidates → 2×2 (512px cells, higher per-image resolution)
          - ≤9 candidates → 3×3
          - >9 candidates → multiple collage calls

        Returns one :class:`VerificationResult` per candidate (same order).
        """
        if not candidates:
            return []

        if grid_size is None:
            grid_size = self._choose_grid(len(candidates))

        max_per_call = grid_size ** 2

        if len(candidates) <= max_per_call:
            return await self._verify_single_collage(
                candidates, query, grid_size,
            )

        # Multiple collage calls for large candidate sets
        all_results: List[VerificationResult] = []
        for i in range(0, len(candidates), max_per_call):
            batch = candidates[i: i + max_per_call]
            gs = self._choose_grid(len(batch))
            print(
                f"  [BatchVerifier] Collage chunk {i // max_per_call + 1}: "
                f"{len(batch)} candidates ({gs}×{gs} grid)"
            )
            chunk_results = await self._verify_single_collage(
                batch, query, gs,
            )
            all_results.extend(chunk_results)

        return all_results

    async def _verify_single_collage(
        self,
        candidates: List[ScoredCandidate],
        query: str,
        grid_size: int,
    ) -> List[VerificationResult]:
        """Run a single collage verification pass."""
        n = min(len(candidates), grid_size ** 2)
        batch = candidates[:n]
        paths = [c.path for c in batch]

        print(
            f"  [BatchVerifier] Creating {grid_size}×{grid_size} collage "
            f"with {n} images (cell={self._cell_size}px)..."
        )
        collage = self.create_collage(paths, grid_size)
        prompt = _COLLAGE_PROMPT.format(query=query, grid=grid_size, n=n)
        msg = VLMClient.build_user_message(text=prompt, images=[collage])
        print(f"  [BatchVerifier] Sending collage to VLM for verification...")
        resp = await self._vlm.achat(messages=[msg], temperature=0.0)

        verdicts = _parse_json_array(resp.content, n)
        print(f"  [BatchVerifier] Received {len(verdicts)} verdicts from VLM")

        # Sort verdicts by index if present to handle out-of-order VLM output
        idx_map: Dict[int, Dict] = {}
        for v in verdicts:
            idx = v.get("index")
            if isinstance(idx, (int, float)):
                idx_map[int(idx)] = v

        results: List[VerificationResult] = []
        for i, cand in enumerate(batch):
            # Prefer index-mapped verdict, fall back to positional
            v = idx_map.get(i + 1) or (verdicts[i] if i < len(verdicts) else {})

            raw_match = bool(v.get("match", False))
            conf = float(v.get("confidence", 0.0))
            reason = str(v.get("reason", ""))

            # Apply minimum confidence gate — prevents hallucinated matches
            verified = raw_match and conf >= self._min_confidence

            status = "PASS" if verified else "FAIL"
            detail = ""
            if raw_match and not verified:
                detail = f" (below min_confidence={self._min_confidence})"
            print(
                f"    [BatchVerifier] #{i + 1} "
                f"{os.path.basename(cand.path)}: "
                f"{status} confidence={conf:.2f}{detail}"
                f"{f'  reason: {reason[:500]}' if reason else ''}"
            )
            results.append(VerificationResult(
                path=cand.path,
                verified=verified,
                confidence=conf,
                caption=str(v.get("caption", "")),
                semantic_tags=list(v.get("tags", [])),
                reasoning=reason,
            ))
        return results

    # ------------------------------------------------------------------ #
    # Single-image refinement
    # ------------------------------------------------------------------ #

    async def refine_single(
        self,
        candidate: ScoredCandidate,
        query: str,
    ) -> VerificationResult:
        """Send one full-resolution image to VLM for detailed verification."""
        print(f"  [BatchVerifier] Refining single image: {os.path.basename(candidate.path)}")
        prompt = _REFINE_PROMPT.format(query=query)
        msg = VLMClient.build_user_message(
            text=prompt, images=[candidate.path],
        )
        resp = await self._vlm.achat(messages=[msg], temperature=0.0)
        parsed = _parse_json_object(resp.content)

        raw_match = bool(parsed.get("match", False))
        conf = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", parsed.get("reasoning", "")))
        verified = raw_match and conf >= self._min_confidence

        print(
            f"  [BatchVerifier] Refine result: "
            f"{'PASS' if verified else 'FAIL'} "
            f"confidence={conf:.2f}"
        )
        return VerificationResult(
            path=candidate.path,
            verified=verified,
            confidence=conf,
            caption=str(parsed.get("caption", "")),
            semantic_tags=list(parsed.get("tags", [])),
            reasoning=reason,
        )


# ------------------------------------------------------------------ #
# JSON parsing helpers
# ------------------------------------------------------------------ #

def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    for src in (text, _strip_fences(text)):
        try:
            return json.loads(src)
        except (json.JSONDecodeError, TypeError):
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _parse_json_array(text: str, expected: int) -> List[Dict[str, Any]]:
    """Parse a JSON array from VLM output, with progressive fallback.

    Handles: markdown fences, Python-style bools, trailing commas,
    JSONL (one object per line), and scattered ``{...}`` objects.
    """
    text = text.strip()

    # --- Attempt 1: direct parse ---
    for src in (text, _strip_fences(text)):
        arr = _try_loads_array(src)
        if arr is not None:
            return arr

    # --- Attempt 2: extract [...] via regex ---
    cleaned = _sanitise_json(text)
    for src in (cleaned, _strip_fences(cleaned)):
        arr = _try_loads_array(src)
        if arr is not None:
            return arr

    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        arr = _try_loads_array(m.group())
        if arr is not None:
            return arr

    # --- Attempt 3: gather individual {...} objects (JSONL / scattered) ---
    objs = _extract_objects(cleaned)
    if objs:
        return objs

    # --- Parsing failed — print raw response for debugging ---
    preview = text[:500] + ("..." if len(text) > 500 else "")
    print(
        f"    [BatchVerifier:WARN] Failed to parse VLM response "
        f"({len(text)} chars). Preview:\n      {preview}"
    )
    return []


def _try_loads_array(src: str) -> Optional[List[Dict[str, Any]]]:
    """Try to JSON-parse *src* as a list; return ``None`` on failure."""
    try:
        arr = json.loads(src)
        if isinstance(arr, list):
            return arr
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _sanitise_json(text: str) -> str:
    """Fix common VLM JSON quirks so ``json.loads`` can handle them."""
    text = _strip_fences(text)
    # Python-style booleans / None
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    # Trailing commas before closing bracket
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _extract_objects(text: str) -> List[Dict[str, Any]]:
    """Find all top-level ``{...}`` JSON objects in *text*.

    Works when the VLM returns objects on separate lines (JSONL)
    or scattered among prose.
    """
    results: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start: i + 1])
                    if isinstance(obj, dict):
                        results.append(obj)
                except (json.JSONDecodeError, TypeError):
                    pass
                start = -1
    return results


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE).strip()
