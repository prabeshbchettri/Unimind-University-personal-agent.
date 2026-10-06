# scripts

Utility scripts for the project. Run them with the backend virtualenv:

- `generate_demo_corpus.py` — regenerates the deterministic synthetic demo
  PDFs in `data/documents/`:

  ```bash
  cd backend && python ../scripts/generate_demo_corpus.py
  ```

- `verify_ingestion.py` — after ingesting, runs a few demo questions through
  embedding + similarity search and prints the top hits with metadata (a quick
  manual end-to-end check of the vector store):

  ```bash
  cd backend && python ../scripts/verify_ingestion.py
  ```
