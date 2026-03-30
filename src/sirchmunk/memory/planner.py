# Copyright (c) ModelScope Contributors. All rights reserved.
"""Memory-Augmented Planner (MAP) — generates search plans from meta-knowledge.

The planner sits between RetrievalMemory (meta-knowledge source) and the
ReAct agent (plan consumer).  It makes a *single* LLM call to produce a
structured :class:`SearchPlan` that can drive either:

* **Guided execution** (confidence >= 0.7): plan steps are executed as
  direct tool calls, bypassing per-step LLM reasoning.
* **Warm-start hint** (confidence < 0.7): the plan is injected into the
  ReAct system prompt to steer the agent's first few actions.

Cost model:
  ~2-3K tokens per planning call.  When guided execution succeeds, this
  replaces 3-5 ReAct LLM calls (~10-15K tokens), yielding 3-5x savings.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .schemas import SearchPlan

logger = logging.getLogger(__name__)


class MemoryAugmentedPlanner:
    """Generate a search plan from query features + learned meta-knowledge.

    Parameters
    ----------
    llm : OpenAIChat
        Shared LLM client (same instance used by the search pipeline).
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def plan(
        self,
        query: str,
        meta_knowledge: Dict[str, Any],
    ) -> Optional[SearchPlan]:
        """Produce a :class:`SearchPlan` from query + meta-knowledge.

        Parameters
        ----------
        query : str
            The user's search query.
        meta_knowledge : dict
            Aggregated meta-knowledge from PatternMemory, containing:
            ``query_type``, ``complexity``, ``entity_count``, ``hop_hint``,
            ``distilled_rules``, ``failure_warnings``, ``success_rate``,
            ``avg_loops``, ``avg_tokens``, ``best_keyword_strategy``.

        Returns
        -------
        SearchPlan or None
            Structured plan on success; ``None`` when the LLM call fails
            or the response cannot be parsed.
        """
        from sirchmunk.agentic.prompts import MAP_PLANNING_PROMPT

        rules = meta_knowledge.get("distilled_rules", [])
        warnings = meta_knowledge.get("failure_warnings", [])

        # Skip planning when no meta-knowledge is available (cold start)
        if not rules and not warnings:
            return None

        prompt = MAP_PLANNING_PROMPT.format(
            query=query,
            query_type=meta_knowledge.get("query_type", "factual"),
            complexity=meta_knowledge.get("complexity", "moderate"),
            entity_count=meta_knowledge.get("entity_count", 0),
            hop_hint=meta_knowledge.get("hop_hint", "single"),
            strategy_rules="\n".join(f"- {r}" for r in rules) or "(none)",
            failure_warnings="\n".join(f"- {w}" for w in warnings) or "(none)",
            success_rate=meta_knowledge.get("success_rate", 0.0),
            avg_loops=meta_knowledge.get("avg_loops", 4.0),
            avg_tokens=meta_knowledge.get("avg_tokens", 0),
        )

        try:
            resp = await self._llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            raw = (resp.content or "").strip()
            return self._parse_plan(raw)
        except Exception as exc:
            logger.debug("MAP planning failed: %s", exc)
            return None

    @staticmethod
    def _parse_plan(raw: str) -> Optional[SearchPlan]:
        """Parse LLM response into a SearchPlan, tolerant of markdown."""
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start < 0 or brace_end <= brace_start:
            return None
        try:
            data = json.loads(raw[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            return None

        steps = data.get("plan_steps", [])
        if not steps:
            return None

        return SearchPlan(
            plan_steps=steps,
            keyword_strategy=data.get("keyword_strategy", "direct"),
            expected_hops=data.get("expected_hops", 2),
            confidence=float(data.get("confidence", 0.0)),
            warnings=data.get("warnings", []),
            reasoning_type=data.get("reasoning_type", "simple"),
            answer_format=data.get("answer_format", "entity"),
            source="MAP",
        )
