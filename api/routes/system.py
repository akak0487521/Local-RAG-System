from fastapi import APIRouter, Security
from fastapi.responses import RedirectResponse
from typing import Optional, List

from ..app import _auth, api_key_header, collection
from ..config import (
    DB_PATH,
    PERSIST_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    DEFAULT_LANGUAGE,
    MODEL_PRIORITY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    OLLAMA_USE_CHAT,
)

router = APIRouter()


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/")


@router.get("/health")
def health(api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    chroma_ok, cnt = True, None
    try:
        cnt = collection.count()
    except Exception:
        chroma_ok = False
    ollama_alive = False
    ollama_models: List[str] = []
    try:
        import requests

        rr = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if rr.ok:
            ollama_alive = True
            jd = rr.json()
            ollama_models = [t.get("name") for t in jd.get("models", [])]
    except Exception:
        pass
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "persist_dir": PERSIST_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
        "docs_count": cnt,
        "chroma_ok": chroma_ok,
        "language_default": DEFAULT_LANGUAGE,
        "backends": {
            "priority": MODEL_PRIORITY,
            "openai": {"enabled": bool(OPENAI_API_KEY), "model": OPENAI_MODEL},
            "ollama": {
                "enabled": bool(OLLAMA_MODEL),
                "model": OLLAMA_MODEL,
                "host": OLLAMA_HOST,
                "use_chat": OLLAMA_USE_CHAT,
                "alive": ollama_alive,
                "tags": ollama_models,
            },
        },
    }


@router.get("/config")
def config(api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    return {
        "db_path": DB_PATH,
        "persist_dir": PERSIST_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
        "language_default": DEFAULT_LANGUAGE,
        "model_priority": MODEL_PRIORITY,
        "openai_model": OPENAI_MODEL,
        "openai_enabled": bool(OPENAI_API_KEY),
        "ollama_model": OLLAMA_MODEL,
        "ollama_host": OLLAMA_HOST,
        "ollama_use_chat": OLLAMA_USE_CHAT,
    }
