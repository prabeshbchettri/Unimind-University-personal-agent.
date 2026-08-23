# Deployment

Production deployment guide for the University Academic Assistant (Phase 13).
The stack is containerized with docker-compose; nothing assumes a GPU.

## Quick start (full stack, hermetic)

```bash
docker compose up -d
```

This starts, with hermetic backends (deterministic embedder, stub LLM —
no model server required):

| Service | Container | Host | Purpose |
|---------|-----------|------|---------|
| frontend | nginx | `http://localhost:8080` | React SPA, proxies `/api/*` to backend |
| backend | uvicorn | `http://localhost:8000` | FastAPI app |
| qdrant | qdrant/qdrant:v1.13.2 | `:6333` | vector store |
| neo4j | neo4j:5.26-community | `:7474`, `:7687` | knowledge graph |
| postgres | postgres:16-alpine | `:5432` | chat history |

Health checks: `docker compose ps`, or `curl http://localhost:8000/health`.

## With a local LLM (Ollama, CPU)

```bash
docker compose --profile llm up -d
docker compose --profile llm exec ollama ollama pull llama3.1:8b
docker compose --profile llm exec ollama ollama pull bge-m3
export LLM_BACKEND=ollama EMBEDDER_BACKEND=ollama
docker compose up -d --force-recreate backend
```

Ollama runs on CPU with no GPU reservations; generation is slower than on a
GPU but works on any machine. Re-export the two variables whenever you
recreate the backend (or put them in a `.env` file next to
`docker-compose.yml` — compose interpolates `${LLM_BACKEND:-stub}`).

## Secrets

Nothing is hard-coded. The only credentials are the Neo4j and Postgres
passwords, which default to local-dev values (`assistant`) via
`${NEO4J_PASSWORD:-assistant}` / `${POSTGRES_PASSWORD:-assistant}`.

**In production, always override them:**

```bash
export NEO4J_PASSWORD='<long random>'
export POSTGRES_PASSWORD='<long random>'
docker compose up -d
```

Keep secrets in a secret manager or an untracked `.env` (see
`backend/.env.production` for the full template — copy it, fill it in, never
commit it).

## Environment templates

- `backend/.env.development` — local dev: hermetic backends, verbose logs.
- `backend/.env.test` — hermetic test/eval config (mirrors the test suite).
- `backend/.env.production` — full production config with `CHANGE_ME`
  placeholders; copy to `.env` and fill in.

`Settings` reads `.env` from the working directory (the backend container's
`/app`), and environment variables always win.

## Image building

```bash
docker compose build            # both images
docker build -t assistant-backend ./backend
docker build -t assistant-frontend ./frontend
```

The frontend image runs nginx and serves the SPA; `location /api/` strips
the prefix and proxies to `backend:8000` (same behaviour as the Vite dev
proxy). The backend image runs a single uvicorn worker by design: the
in-process caches and in-memory graph rely on one process. Scale out
horizontally with a load balancer and a shared rate limiter/cache in front.

## Production hardening checklist

1. Override `NEO4J_PASSWORD` and `POSTGRES_PASSWORD`.
2. Set `CORS_ORIGINS` to your real frontend origin (or leave empty if the
   browser never calls the API directly).
3. Restrict or remove the published ports of qdrant/neo4j/postgres
   (`ports: ["127.0.0.1:6333:6333", ...]` or drop them; the backend reaches
   them over the compose network).
4. Terminate TLS at the public entry point (nginx or a load balancer).
5. Keep `MAX_UPLOAD_BYTES` and `RATE_LIMIT_PER_MINUTE` at production
   values; raise the limit only if real usage demands it.
6. Point `LOG_FORMAT=json` so logs are machine-parseable; ship stdout to a
   log aggregator and correlate by `request_id`.
7. Review `docs/SECURITY.md` for the full measures list.

## Local (non-Docker) run

See the README: `pip install -r requirements.txt`, `uvicorn app.main:app`,
`npm install && npm run dev` (frontend on :5173, proxying to :8000).