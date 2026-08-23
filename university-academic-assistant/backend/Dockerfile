# Backend image for the University Academic Assistant.
#
# Runs the FastAPI service with the hermetic backends by default
# (deterministic embedder / stub LLM); point EMBEDDER_BACKEND and
# LLM_BACKEND at "ollama" via docker-compose when an Ollama server is
# available. No GPU is assumed anywhere.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# One uvicorn worker is intentional: the in-process caches (embedding +
# retrieval) and the in-memory graph backend are per-process, and a single
# worker keeps their hit rates meaningful. Scale out horizontally behind a
# load balancer if needed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]