# Docker

Container definitions for the full stack (PostgreSQL, Qdrant, Neo4j, Ollama,
API, frontend) are introduced in later phases.

Phase 1 runs entirely without containers.

For local development of later phases, the backing stores can be started with:

```powershell
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
docker run -d --name postgres -p 5432:5432 -e POSTGRES_USER=assistant -e POSTGRES_PASSWORD=assistant -e POSTGRES_DB=academic_assistant postgres:16
docker run -d --name ollama -p 11434:11434 ollama/ollama
```
