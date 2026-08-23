"""Graph schema: stable keys and relationship validation (Phase 7).

Defines the only relationships allowed into the graph and the stable keys used
to avoid duplicate nodes. Unsupported relationships are rejected before they
can be written, so an extraction step (or a future LLM-based extractor) can
never introduce relationships outside this model.
"""

from __future__ import annotations

import re

from app.graph.models import (
    NODE_LABELS,
    RELATIONSHIP_TYPES,
    GraphNode,
    GraphRelationship,
)

# Allowed (start_label, relationship_type, end_label) triples.
ALLOWED_RELATIONSHIPS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("University", "HAS_PROGRAM", "Program"),
        ("Program", "HAS_SEMESTER", "Semester"),
        ("Semester", "HAS_SUBJECT", "Subject"),
        ("Subject", "HAS_TOPIC", "Topic"),
        ("Topic", "HAS_SUBTOPIC", "Subtopic"),
        ("Subject", "HAS_QUESTION", "PastQuestion"),
        ("PastQuestion", "ABOUT", "Topic"),
    }
)


def normalize_name(value: str) -> str:
    """Stable canonical key for name-based nodes (case/whitespace insensitive)."""
    return re.sub(r"\s+", " ", value.strip().lower())


def node_key(label: str, value) -> str:
    """Build the stable identity key for a node label + value.

    ``value`` is a human name for University/Program/Subject/Topic/Subtopic, a
    semester number for Semester, or a ``(document_id, question_number)`` pair
    for PastQuestion.
    """
    if label in {"University", "Program", "Subject", "Topic", "Subtopic"}:
        return normalize_name(str(value))
    if label == "Semester":
        return f"sem:{int(value)}"
    if label == "PastQuestion":
        document_id, question_number = value
        return f"{document_id}:{question_number}"
    raise ValueError(f"Unknown node label: {label!r}")


def is_supported_node(label: str) -> bool:
    return label in NODE_LABELS


def is_supported_relationship(rel: GraphRelationship) -> bool:
    return (
        (rel.start_label, rel.type, rel.end_label) in ALLOWED_RELATIONSHIPS
        and rel.type in RELATIONSHIP_TYPES
    )


def validate_relationships(
    relationships: list[GraphRelationship],
) -> tuple[list[GraphRelationship], list[GraphRelationship]]:
    """Split relationships into ``(valid, invalid)`` per the schema.

    Invalid relationships (anything not in ``ALLOWED_RELATIONSHIPS``) are never
    written to the graph.
    """
    valid: list[GraphRelationship] = []
    invalid: list[GraphRelationship] = []
    for rel in relationships:
        if is_supported_relationship(rel):
            valid.append(rel)
        else:
            invalid.append(rel)
    return valid, invalid


def deduplicate_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    """Collapse nodes to unique (label, key) pairs, merging properties."""
    unique: dict[tuple[str, str], GraphNode] = {}
    for node in nodes:
        identity = (node.label, node.key)
        if identity in unique:
            unique[identity].props.update(node.props)
        else:
            unique[identity] = node
    return list(unique.values())


def deduplicate_relationships(relationships: list[GraphRelationship]) -> list[GraphRelationship]:
    """Collapse relationships to unique identities, merging properties."""
    unique: dict[tuple, GraphRelationship] = {}
    for rel in relationships:
        if rel.identity in unique:
            unique[rel.identity].props.update(rel.props)
        else:
            unique[rel.identity] = rel
    return list(unique.values())
