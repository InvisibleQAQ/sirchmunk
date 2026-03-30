# Copyright (c) ModelScope Contributors. All rights reserved.
"""StrategyDistiller — LLM-powered batch reflection for meta-knowledge.

After every N search trajectories, the distiller invokes the LLM to
extract *generalizable* strategy rules from recent experience.  These
rules are stored in PatternMemory and consumed by the
:class:`MemoryAugmentedPlanner` for future searches.

The distiller bridges statistical RL (which learns correlations) with
LLM reasoning (which provides causal attribution and generalization):

* Statistical: "bridge questions succeed 65% of the time"
* Distilled:  "bridge questions succeed BECAUSE searching the bridge
  entity first prevents dead-end loops"

Cost model:
  ~3K tokens per distillation call.  Amortized over N=10-50 queries,
  this adds ~60-300 tokens/query.  Each distilled rule saves 2-3
  ReAct LLM calls in future queries, yielding ROI of 1:50 to 1:250.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schemas import AbstractTrajectory, StrategyDistillation

logger = logging.getLogger(__name__)


class StrategyDistiller:
    """Extract generalizable strategy rules from a batch of trajectories.

    Parameters
    ----------
    llm : OpenAIChat
        Shared LLM client.
    """

    _MIN_TRAJECTORIES = 3
    _MAX_TRAJECTORIES_PER_PROMPT = 20
    _MAX_STEPS_PER_TRAJECTORY = 6
    _MAX_RULES = 10
    _MAX_WARNINGS = 5
    _SUCCESS_THRESHOLD = 0.5

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def distill(
        self,
        trajectories: List[AbstractTrajectory],
        query_type: str,
        complexity: str,
    ) -> Optional[StrategyDistillation]:
        """Distill generalizable rules from a batch of trajectories.

        Parameters
        ----------
        trajectories : list[AbstractTrajectory]
            Recent search trajectories (strategy-level, no instance data).
        query_type : str
            The query type these trajectories belong to.
        complexity : str
            Complexity level of these trajectories.

        Returns
        -------
        StrategyDistillation or None
            Distilled rules on success; ``None`` on failure.
        """
        if len(trajectories) < self._MIN_TRAJECTORIES:
            return None

        from sirchmunk.agentic.prompts import STRATEGY_DISTILLATION_PROMPT

        traj_text = self._format_trajectories(trajectories)
        prompt = STRATEGY_DISTILLATION_PROMPT.format(
            n_trajectories=len(trajectories),
            query_type=query_type,
            complexity=complexity,
            trajectories_text=traj_text,
        )

        try:
            resp = await asyncio.wait_for(
                self._llm.achat(
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                ),
                timeout=30.0,
            )
            raw = (resp.content or "").strip()
            return self._parse_distillation(raw, trajectories, query_type, complexity)
        except asyncio.TimeoutError:
            logger.info("Strategy distillation timed out after 30s, skipping")
            return None
        except Exception as exc:
            logger.debug("Strategy distillation failed: %s", exc)
            return None

    @staticmethod
    def _format_trajectories(trajectories: List[AbstractTrajectory]) -> str:
        """Format trajectories into compact text for the LLM prompt."""
        lines: List[str] = []
        max_traj = StrategyDistiller._MAX_TRAJECTORIES_PER_PROMPT
        max_steps = StrategyDistiller._MAX_STEPS_PER_TRAJECTORY
        threshold = StrategyDistiller._SUCCESS_THRESHOLD
        for i, t in enumerate(trajectories[:max_traj], 1):
            outcome_label = "SUCCESS" if t.outcome >= threshold else "FAIL"
            steps_desc = " → ".join(
                f"{s.action}({s.strategy})" for s in t.steps[:max_steps]
            )
            lines.append(
                f"{i}. [{outcome_label}] type={t.query_type} "
                f"complexity={t.complexity} hops={t.hop_hint} "
                f"loops={t.loops_used} tokens={t.total_tokens} "
                f"steps: {steps_desc}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_distillation(
        raw: str,
        trajectories: List[AbstractTrajectory],
        query_type: str,
        complexity: str,
    ) -> Optional[StrategyDistillation]:
        """Parse LLM response into StrategyDistillation."""
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start < 0 or brace_end <= brace_start:
            return None
        try:
            data = json.loads(raw[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            return None

        rules = data.get("rules", [])
        if not rules:
            return None

        threshold = StrategyDistiller._SUCCESS_THRESHOLD
        n_success = sum(1 for t in trajectories if t.outcome >= threshold)
        return StrategyDistillation(
            query_type=query_type,
            complexity=complexity,
            rules=rules[:StrategyDistiller._MAX_RULES],
            failure_warnings=data.get("failure_warnings", [])[:StrategyDistiller._MAX_WARNINGS],
            best_keyword_strategy=data.get("best_keyword_strategy", ""),
            sample_count=len(trajectories),
            success_rate=data.get("success_rate", n_success / max(len(trajectories), 1)),
            avg_loops=data.get("avg_loops", 0.0),
            avg_tokens=data.get(
                "avg_tokens",
                sum(t.total_tokens for t in trajectories) / max(len(trajectories), 1),
            ),
            distilled_at=datetime.now(timezone.utc).isoformat(),
        )
