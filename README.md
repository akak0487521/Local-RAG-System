# Local RAG System

A lightweight starter kit for building a **retrieval‑augmented generation (RAG)** service that runs entirely on local resources. It couples a FastAPI backend with a minimal web front‑end and supports both local and hosted LLMs.

## Features

- **FastAPI backend** with configurable environment variables for model selection, vector store location, and API key enforcement.
- **Vector search** powered by [Chroma](https://www.trychroma.com/) and SentenceTransformer embeddings.
- **Pluggable LLM backends**: call OpenAI models or run local models through [Ollama](https://ollama.ai/).
- **Document ingestion scripts** to transform JSON/Markdown documents into vector embeddings.
- **Streaming responses** for interactive chat or composition.
- **Docker support** for reproducible deployment.

## Project Structure

Local-RAG-System/
├── api/ # FastAPI application
├── modelfiles/ # Ollama model definitions
├── scripts/ # CLI utilities for ingestion and querying
├── vector_store/ # Persistent Chroma database
├── web/ # Static assets for the web UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt


## Getting Started (Local Python)

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt

2. Configure environment

   Copy .env.example or create your own .env file. Key variables include:

OPENAI_API_KEY

OLLAMA_MODEL (e.g. llama3:latest)

PERSIST_DIR (path to Chroma data)

DOCS_DIR (path to your knowledge base)

3. Ingest documents

Place JSON documents in docs/, then build the index:

python scripts/build_index.py

4. Run the API

uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
Visit http://localhost:8000/docs for interactive API docs.

## Getting Started (Docker)
1. Prepare files

Ensure Dockerfile, docker-compose.yml, .env.docker.example, and requirements.txt are in your project root.
Create docs/ and vector_store/ directories if they don't exist.

2. Configure environment

bash

cp .env.docker.example .env
# edit .env as needed

3. Launch

bash

docker compose up --build -d

Pull your desired model into the Ollama container on first run:

bash

docker exec -it ollama ollama pull llama3

## API Overview
Endpoint	Description
POST /search	Retrieve relevant documents
POST /compose	Generate a response in one request
POST /compose_stream	Stream tokens for interactive chats

Refer to the auto-generated Swagger docs at /docs for schemas and request examples.

## Scripts
scripts/build_index.py – convert documents into embeddings and store them in Chroma.
scripts/ingest_docs_to_chroma.py – alternative ingestion helper.
scripts/query_local.py – command-line querying of the index.
scripts/rag_ask.py – convenience wrapper for asking questions via the API.

## Contributing
This project is provided as a starter template. Feel free to fork and adapt it for your own RAG experiments or internal tools.
