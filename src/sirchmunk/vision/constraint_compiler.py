"""ConstraintCompiler — translate natural language to visual search parameters.

Runs the VLM in *text-only* mode exactly once per search to produce a
structured JSON specification of the visual properties expected in
matching images.  This is the cheapest VLM call in the pipeline and
drives all subsequent filtering in Layer 1 (Visual Grep).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from sirchmunk.schema.vision import VisualConstraint
from sirchmunk.llm.vlm_chat import VLMClient

_COMPILER_PROMPT = """\
You are a visual search constraint compiler.  Given the user's image
search query, output a JSON object that describes the expected visual
properties of matching images.  Do NOT search — only analyse the query
linguistically and produce constraints.

Output JSON schema (all fields optional except clip_query and expanded_queries):
{{
  "clip_query": "<REQUIRED: short English phrase for CLIP/SigLIP matching, e.g. 'a photo of sheep in a field'>",
  "expanded_queries": ["<REQUIRED: 3-5 semantically equivalent English phrases describing the same visual target from different angles, e.g. for 'passenger airplane': ['commercial airliner on runway', 'Boeing passenger jet in flight', 'large airplane with windows at airport', 'jetliner taking off']>"],
  "dominant_hues": ["<colour name>", ...],
  "saturation": [<min 0-1>, <max 0-1>],
  "brightness": [<min 0-1>, <max 0-1>],
  "contrast": "<low|medium|high|any>",
  "texture": ["<keyword>", ...],
  "semantic_tags": ["<object / scene tag>", ...],
  "phash_variance": <0.0-1.0>,
  "scene_type": "<indoor|outdoor|natural|urban|aerial|underwater|studio|any>",
  "subject_position": "<top|bottom|center|left|right|any>",
  "negative_hues": ["<colours that should NOT dominate in matching images>"],
  "negative_textures": ["<texture cues indicating non-matches, e.g. 'dense_text', 'grid_lines'>"],
  "entropy_preference": "<low|high|any>"
}}

Guidelines:
- clip_query: the single best English phrase for matching.
- expanded_queries: generate 3-5 semantically equivalent but lexically
  diverse variations of clip_query.  Include:
    * Synonym substitution (e.g. "airplane" → "aircraft", "jetliner")
    * Visual attribute expansion (e.g. "golden retriever" → "large yellow
      dog outdoor", "golden fur dog playing")
    * Scene context variation (e.g. add/remove background hints)
  Each variant should independently describe the same visual target.
- scene_type: infer the most likely scene category from the query.
- subject_position: where the main subject likely sits in the frame.
- negative_hues / negative_textures: identify visual noise that could
  cause false positives (e.g. blue text on a webpage is NOT "blue sky").
- entropy_preference: "low" for clean/simple subjects, "high" for busy scenes.

User query: {query}

Respond with ONLY the JSON object.  No markdown fences, no explanation."""


class ConstraintCompiler:
    """One-shot VLM compiler: natural language → visual constraint dict."""

    def __init__(self, vlm: VLMClient):
        self._vlm = vlm

    async def compile(self, query: str) -> VisualConstraint:
        """Compile *query* into a :class:`VisualConstraint`.

        The VLM also generates semantically equivalent *expanded_queries*
        for multi-query max-pooled SigLIP2 ranking (query expansion).
        """
        print(f"    [ConstraintCompiler] Sending query to VLM for constraint extraction...")
        prompt = _COMPILER_PROMPT.format(query=query)
        resp = await self._vlm.achat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = _parse_json(resp.content)
        clip_query = raw.get("clip_query", "")
        semantic_tags = raw.get("semantic_tags", [])
        if not clip_query and semantic_tags:
            clip_query = "a photo of " + ", ".join(semantic_tags)

        expanded = raw.get("expanded_queries", []) or []
        expanded = [q for q in expanded if isinstance(q, str) and q.strip()]
        print(
            f"    [ConstraintCompiler] clip_query='{clip_query}', "
            f"expanded={expanded[:3]}{'...' if len(expanded) > 3 else ''}, "
            f"tags={semantic_tags}"
        )

        scene = raw.get("scene_type", "any") or "any"
        position = raw.get("subject_position", "any") or "any"
        neg_hues = raw.get("negative_hues", []) or []
        neg_tex = raw.get("negative_textures", []) or []
        ent_pref = raw.get("entropy_preference", "any") or "any"

        print(
            f"    [ConstraintCompiler] scene={scene}, position={position}, "
            f"neg_hues={neg_hues}, entropy={ent_pref}"
        )
        return VisualConstraint(
            clip_query=clip_query,
            dominant_hues=raw.get("dominant_hues", []),
            saturation_range=tuple(raw.get("saturation", [0.0, 1.0])),
            brightness_range=tuple(raw.get("brightness", [0.0, 1.0])),
            contrast_level=raw.get("contrast", "any"),
            texture_keywords=raw.get("texture", []),
            semantic_tags=semantic_tags,
            phash_variance_threshold=raw.get("phash_variance", 0.5),
            raw_json=raw,
            scene_type=scene,
            subject_position=position,
            negative_hues=neg_hues,
            negative_textures=neg_tex,
            entropy_preference=ent_pref,
            expanded_clip_queries=expanded,
        )


# ------------------------------------------------------------------ #
# JSON parsing (tolerant of markdown fences / trailing text)
# ------------------------------------------------------------------ #

def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    for attempt in (
        text,
        re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE),
    ):
        attempt = re.sub(r"\s*```\s*$", "", attempt, flags=re.MULTILINE).strip()
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return {}
