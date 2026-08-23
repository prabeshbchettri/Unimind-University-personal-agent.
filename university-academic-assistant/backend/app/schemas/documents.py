"""Document endpoint schemas."""

from pydantic import BaseModel


class PageContentSchema(BaseModel):
    """A single extracted page."""

    page_number: int
    text: str


class NormalizedDocumentSchema(BaseModel):
    """Normalized output of the ingestion pipeline."""

    document_id: str
    filename: str
    page_count: int
    extraction_method: str
    pages: list[PageContentSchema]
    metadata: dict


class DocumentUploadResponse(NormalizedDocumentSchema):
    """Alias for the upload response (the normalized document itself)."""


class DocumentResponse(BaseModel):
    """A single ingested document record (placeholder for later phases)."""

    document_id: str
    title: str
    document_type: str


class DocumentListResponse(BaseModel):
    """List of documents (placeholder for later phases)."""

    items: list[DocumentResponse]
    total: int