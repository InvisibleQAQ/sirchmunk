# Copyright (c) ModelScope Contributors. All rights reserved.
"""Global agent memory — self-evolving episodic and semantic memory.

Supports both text-based and vision-based search pipelines.  The memory
records search episodes, path patterns, query clusters, and pipeline
statistics, making the agent progressively smarter over time.

Install via::

    pip install sirchmunk
"""

from .agent_memory import AgentMemory, SearchAdvice, SearchEpisode

__all__ = ["AgentMemory", "SearchAdvice", "SearchEpisode"]
