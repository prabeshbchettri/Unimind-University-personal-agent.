"""Graph endpoint schemas (Phase 7)."""

from pydantic import BaseModel


class GraphSummarySchema(BaseModel):
    """Aggregate status and entity counts of the knowledge graph."""

    status: str
    backend: str
    nodes: dict[str, int] = {}
    relationships: dict[str, int] = {}