"""Document endpoints.

Phase 2 provides PDF upload + ingestion. Listing persisted document records
arrives in a later phase (once a document repository exists).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.embedding.errors import EmbeddingError, EmbeddingUnavailableError
from app.ingestion.errors import (
    EmptyPdfError,
    IngestionError,
    InvalidPdfError,
    OcrUnavailableError,
    PdfOpenError,
    UnsupportedFileTypeError,
)
from app.config import settings
from app.schemas import DocumentListResponse, IndexResultSchema, NormalizedDocumentSchema
from app.structure.models import StructuredDocument
from app.utils.paths import resolve_data_dir

router = APIRouter(prefix="/documents", tags=["documents"])


def _ingestion_status_code(error: IngestionError) -> int:
    """Map ingestion errors to sensible HTTP status codes."""
    if isinstance(error, (InvalidPdfError, PdfOpenError, EmptyPdfError)):
        return 400
    if isinstance(error, UnsupportedFileTypeError):
        return 415
    if isinstance(error, OcrUnavailableError):
        return 503
    return 500


async def _save_upload(file: UploadFile) -> Path:
    """Validate and persist an uploaded PDF, returning its path."""
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    raw_dir = resolve_data_dir("raw")
    safe_name = Path(file.filename).name
    target = raw_dir / f"{uuid.uuid4().hex[:12]}_{safe_name}"

    # Read in bounded chunks so an oversized upload is rejected with 413
    # before its full content ever reaches memory or disk.
    content = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the {limit_mb:g} MB size limit.",
            )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    target.write_bytes(bytes(content))
    return target


@router.post(
    "/upload",
    response_model=NormalizedDocumentSchema,
    summary="Upload and ingest a PDF document",
)
async def upload_document(request: Request, file: UploadFile) -> NormalizedDocumentSchema:
    """Save an uploaded PDF to the raw data directory and ingest it.

    The response is the normalized document (pages preserved), so both text and
    scanned PDFs return the identical structure.
    """
    target = await _save_upload(file)
    try:
        document = await request.app.state.ingestion_service.ingest_document(str(target))
    except IngestionError as exc:
        raise HTTPException(
            status_code=_ingestion_status_code(exc),
            detail=str(exc),
        ) from exc

    return NormalizedDocumentSchema(**document.to_dict())


@router.post(
    "/analyze",
    response_model=StructuredDocument,
    summary="Upload, ingest and structure a PDF document",
)
async def analyze_document(request: Request, file: UploadFile) -> StructuredDocument:
    """Save, ingest and classify an uploaded PDF.

    Returns the schema-validated structured document: metadata (document type,
    semester, subject, ...) plus type-specific structure.
    """
    target = await _save_upload(file)
    try:
        return await request.app.state.ingestion_service.analyze_document(str(target))
    except IngestionError as exc:
        raise HTTPException(
            status_code=_ingestion_status_code(exc),
            detail=str(exc),
        ) from exc


@router.post(
    "/index",
    response_model=IndexResultSchema,
    summary="Upload, ingest, structure and index a PDF into Qdrant",
)
async def index_document(request: Request, file: UploadFile) -> IndexResultSchema:
    """Save, ingest, classify and vector-index an uploaded PDF.

    Runs the full Phase 2/3/4 pipeline: PDF text/OCR extraction, document
    classification and metadata extraction, semantic chunking, embedding, and
    storage in the Qdrant collection for the document type.
    """
    target = await _save_upload(file)
    try:
        structured = await request.app.state.ingestion_service.analyze_document(str(target))
        result = await request.app.state.vector_indexing_service.index(structured)
    except IngestionError as exc:
        raise HTTPException(
            status_code=_ingestion_status_code(exc),
            detail=str(exc),
        ) from exc
    except (EmbeddingError, EmbeddingUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IndexResultSchema(
        document_id=result["document_id"],
        document_type=structured.metadata.document_type,
        chunk_count=result["chunk_count"],
        collections=result["collections"],
        graph=result.get("graph"),
    )


@router.get("", response_model=DocumentListResponse, summary="List ingested documents")
async def list_documents() -> DocumentListResponse:
    raise HTTPException(
        status_code=501,
        detail="Document listing is not implemented yet.",
    )