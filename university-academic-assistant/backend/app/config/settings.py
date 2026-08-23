"""Centralized application configuration.

All configuration is loaded from environment variables (via a ``.env`` file
when present). Values not provided fall back to sensible local defaults so the
application can start without any external service running.

Secrets must never be hard-coded here; they must come from the environment.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- General -----------------------------------------------------------
    app_name: str = "University Academic Assistant"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    # "json" produces machine-parseable JSON log lines; "text" produces readable output.
    log_format: str = "text"

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Comma-separated list of origins allowed to call the API from the browser
    # (the React dev server). Add your deployed frontend origin in production.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # --- Rate limiting (Phase 13) ------------------------------------------
    # In-process sliding window per client IP. Disabled in the test suite
    # (tests/conftest.py) since many tests share one client address.
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 120

    # --- PostgreSQL --------------------------------------------------------
    database_url: str = "postgresql+psycopg://assistant:assistant@localhost:5432/academic_assistant"
    # Chat history backend: "memory" | "postgres" | "auto".
    #   memory  = in-process SQLite (tests / offline, no server needed)
    #   postgres = PostgreSQL via DATABASE_URL (psycopg 3)
    #   auto    = memory when DATABASE_URL is an in-memory sentinel, else postgres
    database_backend: str = "memory"

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    # Use ":memory:" for an embedded in-memory client (no server), or set a
    # QDRANT_URL to a running Qdrant (e.g. http://localhost:6333).
    # Vector collection names (routed by document type).
    qdrant_collection_university_docs: str = "university_docs"
    qdrant_collection_past_questions: str = "past_questions"
    qdrant_collection_library_books: str = "library_books"

    # --- Neo4j / knowledge graph (Phase 7) ----------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    # Graph backend: "memory" | "neo4j" | "auto".
    #   memory = in-process in-memory graph (tests / offline, no server)
    #   neo4j  = Neo4j via NEO4J_URI (bolt)
    #   auto   = memory when NEO4J_URI is an in-memory sentinel, else neo4j
    graph_backend: str = "memory"

    # --- LLM (Ollama) ------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    # LLM backend for answer generation: "auto" | "ollama" | "stub".
    #   ollama = Llama 3.1 8B via Ollama /api/generate (uses OLLAMA_MODEL)
    #   stub   = deterministic local fallback (tests / offline, no model needed)
    #   auto   = ollama when OLLAMA_BASE_URL is set, else stub
    llm_backend: str = "auto"
    # Sampling temperature for grounded answer generation (low = faithful).
    llm_temperature: float = 0.2
    # Per-request timeout (seconds) for Ollama generation calls.
    llm_timeout: float = 120.0

    # --- RAG (Phase 5) -----------------------------------------------------
    # Default number of retrieved chunks grounded into every answer.
    rag_top_k: int = 5
    # Maximum characters of retrieved context placed in the LLM prompt.
    context_max_chars: int = 4000
    # Maximum number of source chunks included in the LLM prompt.
    context_max_sources: int = 8
    # Relevance floor for retrieval: chunks scoring below this are treated as
    # no evidence (0.0 disables gating).
    rag_min_score: float = 0.0

    # --- Hybrid retrieval (Phase 6) --------------------------------------
    # Retrieval strategy: "auto" (adaptive router, Phase 8) | "normal" | "hybrid".
    #   auto   = Query Analyzer + AdaptiveRouter decide per query (default)
    #   normal = dense vector only (NormalRetriever) for every query
    #   hybrid = dense vector + BM25 sparse, fused with Reciprocal Rank Fusion
    rag_retrieval_strategy: str = "auto"
    # RRF constant k in 1/(k + rank). Larger k compresses the rank advantage.
    hybrid_rrf_k: float = 60.0
    # Rerank blend: weight on the rank-based RRF score vs the dense similarity.
    hybrid_rerank_rrf_weight: float = 0.7
    hybrid_rerank_dense_weight: float = 0.3
    # Candidate pool multiplier: each subsystem returns pool_factor * top_k
    # results before fusion, so the final top_k is chosen across both lists.
    hybrid_pool_factor: int = 2

    # --- Adaptive router (Phase 8) ------------------------------------------
    # Query classifier: "rules" | "llm" | "auto".
    #   rules = deterministic rules only (explainable, no model needed)
    #   llm   = LLM classification; rules first, LLM double-checks low
    #           confidence verdicts
    #   auto  = rules first; LLM consulted only when rules are inconclusive
    router_classifier: str = "auto"
    # Strategy used when classification is uncertain: "hybrid" | "normal".
    router_fallback_strategy: str = "hybrid"

    # --- Recommendation (Phase 9) --------------------------------------------
    # Minimum embedding cosine similarity for a semantic topic match (0..1).
    # Names that are equal after normalization always match, regardless of this.
    recommendation_semantic_threshold: float = 0.4
    # Coverage formula: score = topic_weight * topic_coverage
    #                            + subtopic_weight * subtopic_coverage.
    recommendation_topic_weight: float = 0.85
    recommendation_subtopic_weight: float = 0.15

    # --- Chat history (Phase 10) ----------------------------------------------
    # Maximum characters of recent conversation history placed in the LLM
    # prompt (the most recent messages are kept, older ones dropped).
    history_max_chars: int = 2000
    # Maximum number of previous messages considered for context.
    history_max_messages: int = 12
    # A session's title is derived from its first user message, truncated to
    # this many characters.
    session_title_max_chars: int = 60

    # --- Web search (Phase 11) -------------------------------------------------
    # Web search backend: "stub" | "duckduckgo".
    #   stub       = deterministic canned results (tests / offline, no network)
    #   duckduckgo = live DuckDuckGo HTML search via urllib (no API key;
    #                WEB_SEARCH_TIMEOUT applies, results may be limited)
    web_search_backend: str = "stub"
    # Per-request timeout (seconds) for live web search providers. A provider
    # that exceeds it is treated as a search failure and the router falls back.
    web_search_timeout: float = 8.0
    # Maximum number of web results included in the prompt and returned.
    web_max_sources: int = 4

    # --- Embeddings --------------------------------------------------------
    # Embedding backend: "auto" | "deterministic" | "ollama".
    # "deterministic" is a local token-hash fallback (tests/offline). "ollama"
    # calls Ollama's /api/embed with EMBEDDING_MODEL (e.g. bge-m3), which is a
    # different model from the chat LLM (OLLAMA_MODEL).
    embedder_backend: str = "auto"
    embedding_model: str = "bge-m3"
    # Dimensionality for the deterministic fallback; real backends infer theirs
    # from the first embedding response.
    embedding_dimensions: int = 384

    # --- Chunking (Phase 4) ------------------------------------------------
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150

    # --- Caching (Phase 13) ------------------------------------------------
    # Embeddings are deterministic per model, so memoizing them is always
    # correct. Bounded by max entries (LRU eviction); no explicit TTL needed.
    embedding_cache_max_entries: int = 5000
    # Retrieval cache: stores the final retrieval results per query with a
    # TTL. It is invalidated wholesale by any document index mutation, so it
    # can never serve stale data after an ingestion.
    retrieval_cache_enabled: bool = True
    retrieval_cache_max_entries: int = 256
    retrieval_cache_ttl_seconds: int = 300

    # --- Ingestion / PDF ---------------------------------------------------
    # Maximum accepted upload size in bytes. Larger files are rejected with
    # HTTP 413 without being written to disk.
    max_upload_bytes: int = 20 * 1024 * 1024
    # Path to the Tesseract OCR binary. Leave empty for auto-detection
    # (searches common Windows/Unix locations and PATH).
    tesseract_path: str = ""
    ocr_language: str = "eng"
    # Rendering resolution (DPI) used when rasterizing scanned PDF pages.
    ocr_dpi: int = 300

    # PDF type detection heuristics. A page is considered "text-readable" when
    # it yields at least min_chars_per_page characters. The document is treated
    # as text-readable when that ratio of pages qualifies.
    pdf_analyzer_min_chars_per_page: int = 20
    pdf_analyzer_min_text_ratio: float = 0.5
    # Force a method regardless of analysis: "auto" | "pymupdf" | "ocr".
    pdf_analyzer_force_method: str = "auto"

    # Data directories. Empty values resolve to <project_root>/data/<name>.
    data_raw_dir: str = ""
    data_processed_dir: str = ""

    # --- Structure extraction (Phase 3) -------------------------------------
    # Extraction strategy: "deterministic" | "llm".
    # "llm" requires an LLM service (available from Phase 5 onward).
    structure_extractor: str = "deterministic"
    # Extra university names (comma-separated) detected in documents, in
    # addition to the built-in list.
    university_names: str = ""
    # Minimum classification score before a document is labelled "unknown".
    document_classifier_min_score: int = 3

    @property
    def collection_names(self) -> list[str]:
        """Ordered names of the Qdrant vector collections."""
        return [
            self.qdrant_collection_university_docs,
            self.qdrant_collection_past_questions,
            self.qdrant_collection_library_books,
        ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()


settings = get_settings()
