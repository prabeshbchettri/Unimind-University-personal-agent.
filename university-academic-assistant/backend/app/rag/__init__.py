"""RAG orchestration (Phase 5).

Components that assemble retrieval + context building + generation into a
complete answer pipeline. Phase 5 provides a single context builder; hybrid
and graph strategies are added in later phases.
"""

from .context import ContextBuilder, SYSTEM_PROMPT

__all__ = ["ContextBuilder", "SYSTEM_PROMPT"]
