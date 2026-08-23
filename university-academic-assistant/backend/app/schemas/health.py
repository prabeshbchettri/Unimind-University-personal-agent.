"""Health check schema."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint."""

    status: str
