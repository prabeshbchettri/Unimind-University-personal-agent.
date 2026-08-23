# Performance Profile (Phase 13, Part 2)

Reproducible with `scripts/profile_performance.py` on the hermetic offline
environment (in-memory Qdrant, deterministic embedder, stub LLM):

```powershell
cd backend
python scripts/profile_performance.py
```

## Results (Windows x64, Python 3.12, in-memory Qdrant)

Corpus: 8 documents, ~1,926 characters, 20 chunks, 384-dim embeddings.

| stage | latency |
|---|---|
| chunking (full corpus) | 0.84 ms |
| embedding (20 chunks, batch) | 3.64 ms |
| indexing per document | 64 ms (incl. collection creation) |
| retrieval NORMAL | 11.14 ms per query |
| retrieval HYBRID | 11.55 ms per query |
| retrieval GRAPH | 1.62 ms per query |
| full chat answer() (stub LLM) | 9.73 ms per query |
| library recommendation | 8.15 ms per query |

## Conclusions

- The pipeline overhead (router + retrieval + context assembly) is
  ~10 ms per query. Nothing here warrants optimization; per the Phase 13
  rule, the pipeline was **not** modified during profiling because no
  bottleneck was measured.
- The dominant production cost is the LLM call (Ollama); that is a model
  and hardware property, not a pipeline one. Embeddings are already sent to
  the embedder in batches during indexing.
- GRAPH retrieval is the cheapest strategy (in-memory relationship lookups).
- The profile should be re-run after any change to the chunker, retrievers,
  or the context builder; a change that adds >10x to these numbers is a
  regression.
