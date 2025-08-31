from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .config import DEFAULT_LANGUAGE


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    namespace: Optional[str] = None
    canonicality: Optional[str] = None
    rerank: bool = False
    highlight: bool = False


class StyleSpec(BaseModel):
    tone: Optional[str] = None
    directness: Optional[float] = None
    empathy: Optional[float] = None
    hedging: Optional[float] = None
    formality: Optional[float] = None


class ComposeRequest(BaseModel):
    query: str
    mode: str = "strict"  # "strict" | "creative"
    k: int = 6
    namespace: Optional[str] = None
    canonicality: Optional[str] = None
    rerank: bool = True
    engine: Optional[str] = None  # "openai" | "ollama"
    language: Optional[str] = DEFAULT_LANGUAGE
    selected_ids: Optional[List[str]] = None
    debug: Optional[bool] = False
    target_length: Optional[str] = None
    max_tokens: Optional[int] = None
    num_predict: Optional[int] = None
    thread_id: Optional[str] = None  # ← 對話 ID（前端可用 localStorage 保存）
    style: Optional[StyleSpec] = None


class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    k: int = 5
    namespace: Optional[str] = None
    canonicality: Optional[str] = None
    rerank: bool = False
    engine: Optional[str] = None  # "openai" | "ollama"
    language: Optional[str] = DEFAULT_LANGUAGE
    style: Optional[StyleSpec] = None


class SaveDocItem(BaseModel):
    title: str
    content: str
    metadata: Dict[str, Any] = {}
