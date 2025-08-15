# Docker Quickstart

## Prerequisites
- Docker & Docker Compose
- Your project structure like:
```
local-rag-starter/
  api/app.py
  scripts/
  docs/               # your JSON knowledge base (mounted as volume)
  vector_store/       # Chroma persistence (mounted as volume)
```
> If `docs/` or `vector_store/` are missing, create empty folders.

## 1) Drop these files into your project root
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `requirements.txt`
- `.env.docker.example` (copy to `.env` and edit)

## 2) Configure `.env`
Copy:
```
cp .env.docker.example .env
```
Edit `.env` as needed:
- `MODEL_PRIORITY=ollama,openai`
- `OLLAMA_MODEL=gpt-oss:20b` (or `llama3`)
- If Ollama runs on the host, set `OLLAMA_HOST=http://host.docker.internal:11434`

## 3) Start
```
docker compose up --build -d
```
First time, pull a model into the `ollama` container:
```
docker exec -it ollama ollama pull gpt-oss:20b
# or: docker exec -it ollama ollama pull llama3
```

## 4) Test
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Sample calls (Swagger)
- `POST /search`
```json
{ "query": "遠征模式 核心規則", "k": 6, "rerank": true }
```
- `POST /compose`
```json
{
  "query": "以霽華與蒼辰雪山重逢為前情，寫 300-400 字遠征開場旁白",
  "mode": "creative",
  "k": 6,
  "rerank": true,
  "engine": "ollama"
}
```

## Troubleshooting
- **ModuleNotFoundError: api**  
  Ensure `api/app.py` exists and `uvicorn api.app:app` is correct.
- **Ollama unreachable**  
  Scenario A (compose service): `OLLAMA_HOST=http://ollama:11434`  
  Scenario B (host): set `OLLAMA_HOST=http://host.docker.internal:11434` and uncomment `extra_hosts` in `docker-compose.yml`.
- **Vectors not persisted**  
  Ensure the volumes are mounted: `./vector_store:/app/vector_store`.
