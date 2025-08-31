from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR: str = os.getenv("DOCS_DIR", "/app/docs")
KB_DB_PATH: str = os.getenv("KB_DB_PATH", "/app/data/kb.sqlite")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "gamefantasy")
PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./vector_store")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
API_KEY_ENV: str = os.getenv("API_KEY", "changeme")
READONLY_MODE: bool = os.getenv("READONLY_MODE", "false").lower() == "true"
DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "zh-tw")
FILTER_META_DEFAULT: bool = os.getenv("FILTER_META_DEFAULT", "true").lower() == "true"

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_USE_CHAT: bool = os.getenv("OLLAMA_USE_CHAT", "true").lower() == "true"

MODEL_PRIORITY: List[str] = [x.strip() for x in os.getenv("MODEL_PRIORITY", "ollama,openai").split(",") if x.strip()]
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
DB_PATH: str = os.getenv("CONV_DB_PATH", "/app/data/conversations.db")
META_TAGS: Set[str] = {"schema", "prompt", "config", "system", "curator", "meta"}
HALF_LIFE_DAYS: int = int(os.getenv("RECENCY_HALF_LIFE_DAYS", "45"))
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Ensure database directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
