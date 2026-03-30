# Copyright (c) ModelScope Contributors. All rights reserved.
"""Self-evolving retrieval memory system with Meta-RL.

Provides a layered memory architecture that learns generalizable
strategies from each search session:

- **PatternMemory**: query type → optimal search strategy mapping
  (+ Meta-RL: trajectory storage, distilled rules, loop budget)
- **CorpusMemory**: entity-path index + semantic keyword expansion
- **FailureMemory**: noise keywords, dead paths, failed strategies
- **FeedbackMemory**: implicit/explicit signal collection and dispatch

Usage::

    from sirchmunk.memory import RetrievalMemory, FeedbackSignal

    memory = RetrievalMemory(work_path="/path/to/workspace")
    hint = memory.suggest_strategy("Who invented the telephone?")
"""

from .bridge import MemoryBridge, MemoryPrior
from .manager import RetrievalMemory
from .schemas import (
    FeedbackSignal,
    SearchPlan,
    StrategyHint,
)

__all__ = [
    "MemoryBridge",
    "MemoryPrior",
    "RetrievalMemory",
    "FeedbackSignal",
    "SearchPlan",
    "StrategyHint",
]
