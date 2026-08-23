"""Graph repository: Neo4j + in-memory backends (Phase 7).

Neo4j holds the explicit academic relationships (university -> program ->
semester -> subject -> topic/subtopic -> past question). Qdrant remains the
vector/semantic store — the two databases are not interchangeable and are kept
separate; a later phase will combine them behind one retrieval interface.

The in-memory backend mirrors the graph operations for tests and offline dev
(no server required), exactly like the embedded Qdrant client.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from app.config import Settings
from app.graph.models import GraphNode, GraphRelationship
from app.graph.schema import node_key
from app.repositories.base import Repository

_IN_MEMORY_URLS = {"", ":memory:", "memory"}


class GraphUnavailableError(RuntimeError):
    """Raised when the backing graph store cannot be reached."""


class GraphRepository(Repository):
    """Base interface for knowledge-graph storage and queries."""

    async def ping(self) -> bool:
        raise NotImplementedError

    async def merge_nodes(self, nodes: Sequence[GraphNode]) -> None:
        raise NotImplementedError

    async def merge_relationships(self, relationships: Sequence[GraphRelationship]) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def node_count(self, label: str | None = None) -> int:
        raise NotImplementedError

    async def relationship_count(self, rel_type: str | None = None) -> int:
        raise NotImplementedError

    async def graph_summary(self) -> dict:
        """Per-label node counts and per-type relationship counts."""
        raise NotImplementedError

    # --- Semantic queries ---------------------------------------------------

    async def subjects_in_semester(self, semester: int) -> list[dict]:
        raise NotImplementedError

    async def topics_of_subject(self, subject: str) -> list[dict]:
        raise NotImplementedError

    async def subtopics_of_topic(self, topic: str) -> list[dict]:
        raise NotImplementedError

    async def questions_about_topic(self, topic: str) -> list[dict]:
        raise NotImplementedError

    async def subjects_containing_topic(self, topic: str) -> list[dict]:
        raise NotImplementedError


class InMemoryGraphRepository(GraphRepository):
    """In-process graph used by tests and offline development."""

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[str, dict[str, dict]] = {}
        self._relationships: dict[str, dict] = {}

    async def ping(self) -> bool:
        return True

    def _ensure_label(self, label: str) -> None:
        self._nodes.setdefault(label, {})

    async def merge_nodes(self, nodes: Sequence[GraphNode]) -> None:
        for node in nodes:
            self._ensure_label(node.label)
            bucket = self._nodes[node.label]
            # First-seen wins: a duplicate entity (same label + key) keeps its
            # original properties and source instead of being overwritten.
            if node.key not in bucket:
                bucket[node.key] = dict(node.props)

    async def merge_relationships(self, relationships: Sequence[GraphRelationship]) -> None:
        for rel in relationships:
            # Endpoints must already exist; skip dangling relationships.
            start = self._nodes.get(rel.start_label, {}).get(rel.start_key)
            end = self._nodes.get(rel.end_label, {}).get(rel.end_key)
            if start is None or end is None:
                continue
            # First-seen wins, mirroring the node semantics.
            if rel.identity not in self._relationships:
                self._relationships[rel.identity] = dict(rel.props)

    async def clear(self) -> None:
        self._nodes = {}
        self._relationships = {}

    async def node_count(self, label: str | None = None) -> int:
        if label is None:
            return sum(len(bucket) for bucket in self._nodes.values())
        return len(self._nodes.get(label, {}))

    async def relationship_count(self, rel_type: str | None = None) -> int:
        if rel_type is None:
            return len(self._relationships)
        return sum(1 for identity in self._relationships if identity[2] == rel_type)

    async def graph_summary(self) -> dict:
        return {
            "nodes": {label: len(bucket) for label, bucket in self._nodes.items()},
            "relationships": _count_by_type(self._relationships),
        }

    def _follow_outgoing(self, node, rel_type: str | None = None) -> list[dict]:
        return [
            self._relationships[identity]
            for identity, props in self._relationships.items()
            if identity[0] == node[0] and identity[1] == node[1] and (rel_type is None or identity[2] == rel_type)
        ]

    def _follow_incoming(self, node, rel_type: str | None = None) -> list[dict]:
        return [
            self._relationships[identity]
            for identity, props in self._relationships.items()
            if identity[3] == node[0] and identity[4] == node[1] and (rel_type is None or identity[2] == rel_type)
        ]

    async def subjects_in_semester(self, semester: int) -> list[dict]:
        semester_key = node_key("Semester", semester)
        subject_keys = [identity[4] for identity in self._relationships if identity[0] == "Semester" and identity[1] == semester_key and identity[2] == "HAS_SUBJECT"]
        return [dict(self._nodes["Subject"][key]) for key in sorted(subject_keys) if key in self._nodes.get("Subject", {})]

    async def topics_of_subject(self, subject: str) -> list[dict]:
        subject_key = node_key("Subject", subject)
        topic_keys = [identity[4] for identity in self._relationships if identity[0] == "Subject" and identity[1] == subject_key and identity[2] == "HAS_TOPIC"]
        return [dict(self._nodes["Topic"][key]) for key in sorted(topic_keys) if key in self._nodes.get("Topic", {})]

    async def subtopics_of_topic(self, topic: str) -> list[dict]:
        topic_key = node_key("Topic", topic)
        subtopic_keys = [identity[4] for identity in self._relationships if identity[0] == "Topic" and identity[1] == topic_key and identity[2] == "HAS_SUBTOPIC"]
        return [dict(self._nodes["Subtopic"][key]) for key in sorted(subtopic_keys) if key in self._nodes.get("Subtopic", {})]

    async def questions_about_topic(self, topic: str) -> list[dict]:
        topic_key = node_key("Topic", topic)
        question_keys = [identity[1] for identity in self._relationships if identity[3] == "Topic" and identity[4] == topic_key and identity[2] == "ABOUT"]
        return [dict(self._nodes["PastQuestion"][key]) for key in sorted(question_keys) if key in self._nodes.get("PastQuestion", {})]

    async def subjects_containing_topic(self, topic: str) -> list[dict]:
        topic_key = node_key("Topic", topic)
        subject_keys = [identity[1] for identity in self._relationships if identity[3] == "Topic" and identity[4] == topic_key and identity[2] == "HAS_TOPIC"]
        return [dict(self._nodes["Subject"][key]) for key in sorted(subject_keys) if key in self._nodes.get("Subject", {})]


class Neo4jGraphRepository(GraphRepository):
    """Graph repository backed by a real Neo4j database (bolt)."""

    def __init__(self, uri: str, username: str, password: str, database: str | None = None, timeout: float = 5.0) -> None:
        super().__init__()
        from neo4j import GraphDatabase

        self._database = database or "neo4j"
        self._driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=timeout,
        )

    async def ping(self) -> bool:
        def _ping() -> bool:
            try:
                self._driver.verify_connectivity()
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_ping)

    def _execute(self, query: str, params: dict | None = None):
        def _run() -> None:
            try:
                with self._driver.session(database=self._database) as session:
                    session.run(query, parameters=params or {})
            except Exception as exc:  # pragma: no cover - network
                raise GraphUnavailableError(f"Neo4j unavailable: {exc}") from exc

        return asyncio.to_thread(_run)

    async def merge_nodes(self, nodes: Sequence[GraphNode]) -> None:
        for node in nodes:
            query = (
                f"MERGE (n:`{node.label}` {{key: $key}}) "
                # First-seen wins: a duplicate entity keeps its original
                # properties and source instead of being overwritten.
                "ON CREATE SET n += $props"
            )
            await self._execute(query, {"key": node.key, "props": dict(node.props)})

    async def merge_relationships(self, relationships: Sequence[GraphRelationship]) -> None:
        for rel in relationships:
            query = (
                f"MATCH (a:`{rel.start_label}` {{key: $start_key}}), "
                f"(b:`{rel.end_label}` {{key: $end_key}}) "
                f"MERGE (a)-[r:`{rel.type}` {{key: $rel_key}}]->(b) "
                "ON CREATE SET r += $props"
            )
            await self._execute(
                query,
                {
                    "start_key": rel.start_key,
                    "end_key": rel.end_key,
                    "rel_key": "|".join(rel.identity),
                    "props": dict(rel.props),
                },
            )

    async def clear(self) -> None:
        await self._execute("MATCH (n) DETACH DELETE n")

    def _read(self, query: str, params: dict | None = None):
        async def _run() -> list[dict]:
            try:
                with self._driver.session(database=self._database) as session:
                    records = session.run(query, parameters=params or {}).data()
                    return [dict(record) for record in records]
            except Exception as exc:  # pragma: no cover - network
                raise GraphUnavailableError(f"Neo4j unavailable: {exc}") from exc

        return _run()

    async def node_count(self, label: str | None = None) -> int:
        query = f"MATCH (n{':' + label if label else ''}) RETURN count(n) AS c"
        records = await self._read(query)
        return int(records[0]["c"]) if records else 0

    async def relationship_count(self, rel_type: str | None = None) -> int:
        query = f"MATCH ()-[r{':' + rel_type if rel_type else ''}]->() RETURN count(r) AS c"
        records = await self._read(query)
        return int(records[0]["c"]) if records else 0

    async def graph_summary(self) -> dict:
        from app.graph.models import NODE_LABELS, RELATIONSHIP_TYPES

        nodes: dict[str, int] = {}
        for label in NODE_LABELS:
            nodes[label] = await self.node_count(label)
        relationships: dict[str, int] = {}
        for rel_type in RELATIONSHIP_TYPES:
            relationships[rel_type] = await self.relationship_count(rel_type)
        return {"nodes": nodes, "relationships": relationships}

    async def subjects_in_semester(self, semester: int) -> list[dict]:
        records = await self._read(
            "MATCH (sem:Semester {semester: $semester})-[:HAS_SUBJECT]->(sub:Subject) "
            "RETURN sub ORDER BY sub.name",
            {"semester": semester},
        )
        return [dict(record["sub"]) for record in records]

    async def topics_of_subject(self, subject: str) -> list[dict]:
        records = await self._read(
            "MATCH (sub:Subject {key: $subject_key})-[:HAS_TOPIC]->(t:Topic) "
            "RETURN t ORDER BY t.name",
            {"subject_key": node_key("Subject", subject)},
        )
        return [dict(record["t"]) for record in records]

    async def subtopics_of_topic(self, topic: str) -> list[dict]:
        records = await self._read(
            "MATCH (t:Topic {key: $topic_key})-[:HAS_SUBTOPIC]->(st:Subtopic) "
            "RETURN st ORDER BY st.name",
            {"topic_key": node_key("Topic", topic)},
        )
        return [dict(record["st"]) for record in records]

    async def questions_about_topic(self, topic: str) -> list[dict]:
        records = await self._read(
            "MATCH (q:PastQuestion)-[:ABOUT]->(t:Topic {key: $topic_key}) "
            "RETURN q ORDER BY q.question_number",
            {"topic_key": node_key("Topic", topic)},
        )
        return [dict(record["q"]) for record in records]

    async def subjects_containing_topic(self, topic: str) -> list[dict]:
        records = await self._read(
            "MATCH (sub:Subject)-[:HAS_TOPIC]->(t:Topic {key: $topic_key}) "
            "RETURN sub ORDER BY sub.name",
            {"topic_key": node_key("Topic", topic)},
        )
        return [dict(record["sub"]) for record in records]

    async def close(self) -> None:
        def _close() -> None:
            self._driver.close()

        await asyncio.to_thread(_close)


def build_graph_repository(settings: Settings) -> GraphRepository:
    """Build a graph repository from settings.

    ``GRAPH_BACKEND=memory`` (default) uses the in-process graph. ``neo4j``
    (or ``auto`` with a real ``NEO4J_URI``) uses the Neo4j driver.
    """
    backend = (settings.graph_backend or "auto").lower()
    if backend == "memory" or (backend == "auto" and settings.neo4j_uri in _IN_MEMORY_URLS):
        return InMemoryGraphRepository()
    if backend in {"neo4j", "auto"}:
        return Neo4jGraphRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
        )
    raise ValueError(f"Unknown GRAPH_BACKEND: {settings.graph_backend!r}")


def _count_by_type(relationships: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for identity in relationships:
        rel_type = identity[2]
        counts[rel_type] = counts.get(rel_type, 0) + 1
    return counts
