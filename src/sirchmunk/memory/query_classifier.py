# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lightweight heuristic query classification (zero-LLM cost).

Extracts a compact feature vector from a query string for PatternMemory
pattern-id computation.  Supports both English and Chinese queries.

Extracted from PatternMemory to satisfy Single Responsibility Principle:
PatternMemory handles strategy learning; this module handles classification.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# ── English keyword sets ─────────────────────────────────────────────

_COMPARISON_WORDS = frozenset({
    "which", "both", "difference", "compare", "versus", "vs",
    "better", "more", "less", "rather", "prefer", "between",
})
_FACTUAL_WORDS = frozenset({
    "when", "where", "who", "what", "how many", "how much",
})
_DEFINITION_WORDS = frozenset({
    "what is", "what are", "define", "definition", "meaning",
})
_PROCEDURAL_WORDS = frozenset({
    "how to", "how do", "steps", "process", "procedure",
})

# ── Chinese classification patterns ──────────────────────────────────

_ZH_COMPARISON_RE = re.compile(r"哪个更|对比|区别|比较|和.+哪个|还是.+好")
_ZH_FACTUAL_RE = re.compile(r"谁|什么时候|哪里|多少|几个|在哪")
_ZH_DEFINITION_RE = re.compile(r"什么是|是什么|定义|含义|意思是")
_ZH_PROCEDURAL_RE = re.compile(r"如何|怎么|怎样|步骤|流程|方法是")
_ZH_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# ── Entity extraction patterns ───────────────────────────────────────

_EN_NAMED_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_YEAR_RE = re.compile(r"\b\d{4}\b")
_LOCATION_RE = re.compile(r"\b(?:city|country|state|river|mountain)\b")
_CJK_LOC_RE = re.compile(r"[\u4e00-\u9fff]{2,}(?:市|省|县|国|河|山|湖)")
_CJK_TITLE_RE = re.compile(r"《[^》]+》")


def classify_query(query: str) -> Dict[str, Any]:
    """Extract lightweight feature vector from a query string.

    Supports both English and Chinese queries.

    Returns dict with ``query_type``, ``complexity``, ``entity_types``,
    ``entity_count``, and ``hop_hint``.
    """
    q_lower = query.lower().strip()
    words = q_lower.split()
    has_cjk = bool(_ZH_CJK_RE.search(query))

    # --- Query type ---
    query_type = "factual"
    if has_cjk:
        if _ZH_DEFINITION_RE.search(query):
            query_type = "definition"
        elif _ZH_PROCEDURAL_RE.search(query):
            query_type = "procedural"
        elif _ZH_COMPARISON_RE.search(query):
            query_type = "comparison"
        elif _ZH_FACTUAL_RE.search(query):
            query_type = "factual"
        else:
            query_type = "bridge"
    else:
        if any(w in q_lower for w in _DEFINITION_WORDS):
            query_type = "definition"
        elif any(w in q_lower for w in _PROCEDURAL_WORDS):
            query_type = "procedural"
        elif any(w in words for w in _COMPARISON_WORDS):
            query_type = "comparison"
        elif any(w in q_lower for w in _FACTUAL_WORDS):
            query_type = "factual"
        else:
            query_type = "bridge"

    # --- Complexity ---
    if has_cjk:
        char_count = len(query.strip())
        if char_count <= 10:
            complexity = "simple"
        elif char_count <= 30:
            complexity = "moderate"
        else:
            complexity = "complex"
    else:
        if len(words) <= 6:
            complexity = "simple"
        elif len(words) <= 15:
            complexity = "moderate"
        else:
            complexity = "complex"

    # --- Entity types + entity count ---
    entity_types: List[str] = []
    entity_count = 0

    en_named = _EN_NAMED_RE.findall(query)
    if en_named:
        entity_types.append("named_entity")
        entity_count += len(en_named)

    year_matches = _YEAR_RE.findall(query)
    if year_matches:
        entity_types.append("date")
        entity_count += len(year_matches)

    if _LOCATION_RE.search(q_lower):
        entity_types.append("location")

    if has_cjk:
        cjk_entities = _CJK_LOC_RE.findall(query)
        if cjk_entities:
            if "location" not in entity_types:
                entity_types.append("location")
            entity_count += len(cjk_entities)
        cjk_names = _CJK_TITLE_RE.findall(query)
        if cjk_names:
            entity_types.append("title")
            entity_count += len(cjk_names)

    # Bucket entity_count for stable hashing
    ec_bucket = min(entity_count, 3)

    # --- Hop hint ---
    hop_hint = "single"
    if query_type == "comparison" or entity_count >= 2:
        hop_hint = "multi"
    elif query_type == "bridge":
        hop_hint = "multi" if entity_count >= 1 else "single"

    return {
        "query_type": query_type,
        "complexity": complexity,
        "entity_types": entity_types,
        "entity_count": ec_bucket,
        "hop_hint": hop_hint,
    }
