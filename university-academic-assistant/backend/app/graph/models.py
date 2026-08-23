"""Knowledge-graph entity models (Phase 7).

Represents the focused academic graph:

    University -[:HAS_PROGRAM]-> Program -[:HAS_SEMESTER]-> Semester
    Semester -[:HAS_SUBJECT]-> Subject
    Subject -[:HAS_TOPIC]-> Topic -[:HAS_SUBTOPIC]-> Subtopic
    Subject -[:HAS_QUESTION]-> PastQuestion -[:ABOUT]-> Topic

Only important academic entities are represented — not every sentence or chunk.
Every node/relationship keeps source fields (``document_id``, ``page``,
``source_type``) so the graph is traceable back to the original PDF.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

NODE_LABELS: tuple[str, ...] = (
    "University",
    "Program",
    "Semester",
    "Subject",
    "Topic",
    "Subtopic",
    "PastQuestion",
)

RELATIONSHIP_TYPES: tuple[str, ...] = (
    "HAS_PROGRAM",
    "HAS_SEMESTER",
    "HAS_SUBJECT",
    "HAS_TOPIC",
    "HAS_SUBTOPIC",
    "HAS_QUESTION",
    "ABOUT",
)

# Source fields recorded on every node and relationship.
SOURCE_FIELDS: tuple[str, ...] = ("document_id", "page", "source_type", "title")


class GraphNode(BaseModel):
    """A node to merge into the graph.

    ``key`` is the stable identity used to avoid duplicate nodes (e.g. a
    normalized name, ``sem:5``, or ``document_id:question_number``).
    """

    label: str
    key: str
    props: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """A directed relationship between two nodes (matched by label + key)."""

    type: str
    start_label: str
    start_key: str
    end_label: str
    end_key: str
    props: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        """Stable identity used to avoid duplicate relationships."""
        return (self.start_label, self.start_key, self.type, self.end_label, self.end_key)


class GraphExtraction(BaseModel):
    """Output of the deterministic entity extraction step."""

    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)
