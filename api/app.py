# api/app.py — RAG API (language fix, chat backend, context cap, cache, CORS, rich /health)
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
import os, re, json, time, hashlib

from dotenv import load_dotenv
load_dotenv()

# ---------- Config & Env ----------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./vector_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
API_KEY_ENV = os.getenv("API_KEY", "changeme")
READONLY_MODE = os.getenv("READONLY_MODE", "false").lower() == "true"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "zh-tw")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")        # 例如 "llama3:8b"
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_USE_CHAT = os.getenv("OLLAMA_USE_CHAT", "true").lower() == "true"

MODEL_PRIORITY = [x.strip() for x in os.getenv("MODEL_PRIORITY", "ollama,openai").split(",") if x.strip()]

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
SUMMARIZE_CONTEXT = os.getenv("SUMMARIZE_CONTEXT","false").lower() == "true"

# ---------- LLM clients ----------
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Vector store (Chroma) ----------
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
COLLECTION_NAME = "gamefantasy"
client = chromadb.PersistentClient(path=PERSIST_DIR)
embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

# ---------- Optional: cross-encoder for rerank (lazy) ----------
_CE = None
def get_cross_encoder():
    global _CE
    if _CE is None:
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CE

# ---------- FastAPI ----------
app = FastAPI(title="Local RAG API (Improved)", version="0.4.0")

# CORS（前端頁面好接）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# （可選）如果你把 /web 靜態頁面放專案根的 web/，取消註解即可掛前端
if os.path.isdir("web"):
    app.mount("/app", StaticFiles(directory="web", html=True), name="web")

# API key
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
def _auth(api_key: Optional[str]):
    if API_KEY_ENV and API_KEY_ENV != "changeme":
        if not api_key or api_key != API_KEY_ENV:
            raise HTTPException(status_code=401, detail="Invalid API key")

# ---------- Request models ----------
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    namespace: Optional[str] = None
    canonicality: Optional[str] = None
    rerank: bool = False
    highlight: bool = False

class ComposeRequest(BaseModel):
    query: str
    mode: str = "strict"  # "strict" | "creative"
    k: int = 6
    namespace: Optional[str] = None
    canonicality: Optional[str] = None
    rerank: bool = True
    engine: Optional[str] = None  # "openai" | "ollama" | None -> follow priority
    language: Optional[str] = DEFAULT_LANGUAGE
    selected_ids: Optional[List[str]] = None  # ← 新增：讓 /compose_stream 可選片段

# ---------- Helpers ----------
def backend_available(name: str) -> bool:
    n = (name or "").lower()
    if n == "openai":
        return bool(OPENAI_API_KEY)
    if n == "ollama":
        return bool(OLLAMA_MODEL)
    return False

def choose_backend(request_engine: Optional[str]) -> str:
    if request_engine:
        eng = request_engine.lower()
        if backend_available(eng):
            return eng
        raise HTTPException(status_code=400, detail=f"Requested engine '{request_engine}' not available.")
    for eng in MODEL_PRIORITY:
        if backend_available(eng):
            return eng
    raise HTTPException(status_code=500, detail="No LLM backend available (set OPENAI_API_KEY or OLLAMA_MODEL).")

def _language_hint(lang: Optional[str]) -> str:
    l = (lang or DEFAULT_LANGUAGE or "zh-tw").lower()
    m = {
        "zh": "（請用中文回答）",
        "zh-tw": "（請全程使用繁體中文回答）",
        "zh-cn": "（请全程使用简体中文回答）",
        "en": "Please answer in English.",
        "ja": "日本語で回答してください。",
        "ko": "한국어로만 답변해 주세요.",
        "fr": "Veuillez répondre en français.",
        "de": "Bitte nur auf Deutsch antworten.",
        "es": "Responde únicamente en español."
    }
    return m.get(l, f"Please answer in {lang}.")

_SENT_SPLIT = re.compile(r'[。！？!?；;\n]')

def _highlight(query: str, text: str, max_spans: int = 3) -> List[str]:
    kws = [w.strip() for w in re.split(r'[\s,，。.\-_/|]+', query) if w.strip()]
    if not kws: return []
    spans = []
    for sent in _SENT_SPLIT.split(text or ""):
        s = sent.strip()
        if not s: continue
        if any(k in s for k in kws):
            spans.append(s)
            if len(spans) >= max_spans: break
    return spans

def _query_chroma(req: SearchRequest) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if req.namespace: where["namespace"] = req.namespace
    if req.canonicality: where["canonicality"] = req.canonicality
    res = collection.query(
        query_texts=[req.query],
        n_results=max(1, min(req.k, 20)),
        where=where if where else None,
        include=["documents", "metadatas", "distances", "uris", "data"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0] if "ids" in res else [None] * len(docs)

    hits = []
    for i in range(len(docs)):
        hits.append({
            "rank": i + 1,
            "id": ids[i],
            "score": float(dists[i]),
            "metadata": metas[i],
            "text": docs[i],
        })
    return {"hits": hits}

def _rerank(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hits: return hits
    ce = get_cross_encoder()
    pairs = [[query, h["text"]] for h in hits]
    scores = ce.predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    for i, h in enumerate(hits, 1): h["rank"] = i
    return hits

def _summarize_context_block(text: str, lang_line: str) -> str:
    """可替換成你喜歡的摘要器；目前先做輕量規則摘要避免多一次 LLM 調用。"""
    # 取前 N 字 + 簡單句子抽取
    text = (text or "").strip()
    if not text: return text
    # 句子切分後取前 2-3 句
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    head = " / ".join(sents[:3])
    if len(head) > 400:
        head = head[:400] + "…"
    return head

def _build_context(hits: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS, summarize: bool = SUMMARIZE_CONTEXT, lang_line: str = "") -> Tuple[str, int]:
    ctx_parts, used, count = [], 0, 0
    for h in hits:
        meta = h.get("metadata", {}) or {}
        src = meta.get("file_path") or meta.get("title") or h.get("id") or "unknown_source"
        sec = meta.get("section")
        tag = f"{src}" + (f"#{sec}" if sec else "")
        body = h.get("text","").strip()
        if summarize:
            body = _summarize_context_block(body, lang_line)
        chunk = f"[{tag}]\n{body}"
        if used + len(chunk) > max_chars and ctx_parts:
            break
        ctx_parts.append(chunk)
        used += len(chunk); count += 1
    return ("\n\n---\n\n".join(ctx_parts), count)

def _search_internal(query: str, k: int, namespace: Optional[str], canonicality: Optional[str], rerank: bool):
    """重用 /search 的檢索流程，回傳 hits 陣列"""
    # 正確使用全域的 collection
    where: Dict[str, Any] = {}
    if namespace:
        where["namespace"] = namespace
    if canonicality:
        where["canonicality"] = canonicality

    res = collection.query(
        query_texts=[query],
        n_results=max(1, min(k or 6, 20)),
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0] if "ids" in res else [None] * len(docs)

    for i in range(len(docs)):
        hits.append({
            "rank": i + 1,
            "id": ids[i],
            "score": float(dists[i]),
            "metadata": metas[i],
            "text": docs[i],
        })

    if rerank and hits:
        hits = _rerank(query, hits)

    return hits

def _hits_signature(hits: List[Dict[str, Any]]) -> str:
    basis = [{"id":h.get("id"), "score":round(float(h.get("score", 0.0)), 6)} for h in hits[:6]]
    return hashlib.md5(json.dumps(basis, sort_keys=True).encode()).hexdigest()

# 簡易快取（記憶體），佔位 256 筆
_COMPOSE_CACHE: Dict[Tuple, Dict[str, Any]] = {}
def _cache_get(key: Tuple):
    v = _COMPOSE_CACHE.get(key)
    if v and (time.time() - v.get("_t", 0) < 3600):
        return v
def _cache_set(key: Tuple, value: Dict[str, Any]):
    if len(_COMPOSE_CACHE) > 256:
        _COMPOSE_CACHE.clear()
    value["_t"] = time.time()
    _COMPOSE_CACHE[key] = value

# ---------- System prompts ----------
STRICT_SYS   = "你是《遊戲幻想版》資料官。僅依據提供的片段回答；若資料不足，請列需要的節點與欄位，切勿臆測。"
CREATIVE_SYS = "你是《遊戲幻想版》作者助理。以提供片段為主、適度創作補足細節，但不要違反片段事實。"

# ---------- LLM backends ----------
def _compose_with_openai(query: str, context: str, mode: str, language: Optional[str]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未設定")
    lang_line = _language_hint(language)
    sys_a = "You write accurate content based on the context."
    sys_lang_guard = "Respond ONLY in the requested language. Do not use English if another language is requested."
    msg_user = (
        f"{(STRICT_SYS if mode=='strict' else CREATIVE_SYS)}\n\n"
        f"=== 片段開始 ===\n{context}\n=== 片段結束 ===\n\n"
        f"{lang_line}\n"
        f"Use ONLY the requested language.\n"
        f"Target length: 300–600 (characters or words).\n"
        f"Topic: 「{query}」。\n"
        f"Structure: paragraphs and/or bullet points are allowed."
    )
    # 簡單重試（429/5xx）
    for i in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_a},
                    {"role": "system", "content": sys_lang_guard},
                    {"role": "user", "content": msg_user},
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if i == 2: raise
            time.sleep(1.5 * (i + 1))
    raise RuntimeError("OpenAI call failed after retries.")

def _compose_with_ollama(query: str, context: str, mode: str, language: Optional[str]) -> str:
    import requests
    if not OLLAMA_MODEL:
        raise RuntimeError("OLLAMA_MODEL 未設定")
    lang_line = _language_hint(language)
    sys_prompt = STRICT_SYS if mode == "strict" else CREATIVE_SYS

    if OLLAMA_USE_CHAT:
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "system", "content": "Respond ONLY in the requested language."},
                {"role": "user", "content":
                    f"{lang_line}\n"
                    f"=== 片段開始 ===\n{context}\n=== 片段結束 ===\n"
                    f"請依上文完成「{query}」。字數約 300–600。"
                },
            ],
        }
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        # chat 端點回傳 format：{ "message": {"content": "..."}, ... }
        msg = data.get("message", {}).get("content")
        if msg: return msg
        return data.get("response", "")
    else:
        prompt = (
            f"{sys_prompt}\n\n=== 片段開始 ===\n{context}\n=== 片段結束 ===\n\n"
            f"{lang_line}\n"
            f"Use ONLY the requested language.\n"
            f"Target length: 300–600 (characters or words).\n"
            f"Topic: 「{query}」。\n"
            f"Structure: paragraphs and/or bullet points are allowed."
        )
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

# === 串流：OpenAI (SSE) ===
def _stream_with_openai(query: str, context: str, mode: str, language: Optional[str]):
    """
    以 OpenAI chat.completions 流式輸出，逐塊送出 SSE：
      data: {"token": "<partial text>"}
    需求：環境變數 OPENAI_API_KEY / OPENAI_MODEL
    """
    if not OPENAI_API_KEY:
        # 安全起見仍要 yield 一段錯誤，避免前端無訊號
        yield f'data: {json.dumps({"token": "[OpenAI 未設定 OPENAI_API_KEY]"})}\n\n'
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        sys_prompt = STRICT_SYS if (mode or "").lower() == "strict" else CREATIVE_SYS
        lang_line = _language_hint(language)

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "system", "content": "Respond ONLY in the requested language."},
                {
                    "role": "user",
                    "content": (
                        f"{lang_line}\n"
                        f"=== 片段開始 ===\n{context}\n=== 片段結束 ===\n"
                        f"請依上文完成「{query}」。字數約 300–600。"
                    ),
                },
            ],
            temperature=0.4,
            stream=True,
        )

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.get("content", "")
            except Exception:
                delta = ""
            if delta:
                # 逐段輸出
                yield f"data: {json.dumps({'token': delta})}\n\n"

    except Exception as e:
        # 將錯誤也以一段 SSE 傳回，方便前端顯示
        yield f"data: {json.dumps({'token': f'[OpenAI stream error] {e}'})}\n\n"

# === 新增: Ollama 流式生成 ===
def _stream_with_ollama(query: str, context: str, mode: str, language: Optional[str]):
    """
    以 Ollama /api/chat 串流輸出，逐行解析 JSONL：
      data: {"token": "<partial text>"}
    需求：環境變數 OLLAMA_HOST / OLLAMA_MODEL
    """
    import requests

    if not OLLAMA_MODEL:
        yield f'data: {json.dumps({"token": "[Ollama 未設定 OLLAMA_MODEL]"})}\n\n'
        return

    sys_prompt = STRICT_SYS if (mode or "").lower() == "strict" else CREATIVE_SYS
    lang_line = _language_hint(language)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "system", "content": "Respond ONLY in the requested language."},
            {
                "role": "user",
                "content": (
                    f"{lang_line}\n"
                    f"=== 片段開始 ===\n{context}\n=== 片段結束 ===\n"
                    f"請依上文完成「{query}」。字數約 300–600。"
                ),
            },
        ],
    }

    try:
        with requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                # ollama chat stream 會不斷給出 {"message":{"role":"assistant","content":"..."}} 的片段
                msg = (data.get("message") or {}).get("content")
                if msg:
                    yield f"data: {json.dumps({'token': msg})}\n\n"

                # 某些版本也可能回傳 {"done": true}，這裡不特別處理，讓 /compose_stream 負責最後的收尾事件
    except Exception as e:
        yield f"data: {json.dumps({'token': f'[Ollama stream error] {e}'})}\n\n"

# ---------- Endpoints ----------
@app.get("/health")
def health(api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    # 檢查 chroma
    chroma_ok, cnt = True, None
    try:
        cnt = collection.count()
    except Exception:
        chroma_ok = False
    # 檢查 ollama
    ollama_alive = False
    ollama_models = []
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
        "persist_dir": PERSIST_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
        "docs_count": cnt,
        "chroma_ok": chroma_ok,
        "read_only": READONLY_MODE,
        "language_default": DEFAULT_LANGUAGE,
        "context": {"max_chars": MAX_CONTEXT_CHARS, "summarize": SUMMARIZE_CONTEXT},
        "backends": {
            "priority": MODEL_PRIORITY,
            "openai": {"enabled": bool(OPENAI_API_KEY), "model": OPENAI_MODEL},
            "ollama": {"enabled": bool(OLLAMA_MODEL), "model": OLLAMA_MODEL, "host": OLLAMA_HOST, "use_chat": OLLAMA_USE_CHAT, "alive": ollama_alive, "tags": ollama_models},
        },
    }

@app.post("/search")
def search(req: SearchRequest, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    out = _query_chroma(req)
    hits = out["hits"]
    if req.rerank:
        try:
            hits = _rerank(req.query, hits)
            out["hits"] = hits
            out["reranked"] = True
        except Exception as e:
            out["reranked"] = False
            out["rerank_error"] = str(e)
    if req.highlight:
        for h in hits:
            try:
                h["highlights"] = _highlight(req.query, h.get("text",""))
            except Exception:
                h["highlights"] = []
    return out

@app.post("/compose")
def compose(req: ComposeRequest, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    # 先檢索
    sreq = SearchRequest(
        query=req.query, k=req.k,
        namespace=req.namespace, canonicality=req.canonicality,
        rerank=req.rerank, highlight=False
    )
    out = search(sreq, api_key)
    hits = out.get("hits", [])
    if not hits:
        return {"draft": "", "citations": [], "note": "無檢索命中；請調整 query 或新增資料。"}

    # 構建上下文（可選摘要 + 長度上限）
    lang_line = _language_hint(req.language)
    ctx, used_hits = _build_context(hits, max_chars=MAX_CONTEXT_CHARS, summarize=SUMMARIZE_CONTEXT, lang_line=lang_line)

    # 簡易快取
    engine = choose_backend(req.engine)
    sig = _hits_signature(hits)
    cache_key = (req.query, req.mode, req.language, engine, sig, MAX_CONTEXT_CHARS, SUMMARIZE_CONTEXT)
    cached = _cache_get(cache_key)
    if cached:
        return {k:v for k,v in cached.items() if k != "_t"}

    # 生成
    try:
        if engine == "openai":
            draft = _compose_with_openai(req.query, ctx, req.mode, req.language)
        elif engine == "ollama":
            draft = _compose_with_ollama(req.query, ctx, req.mode, req.language)
        else:
            raise RuntimeError(f"未知的 engine: {engine}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compose error ({engine}): {e}")

    # citations 不混入內文
    cits = []
    for h in hits:
        m = h.get("metadata", {}) or {}
        cits.append({"id": h.get("id"), "file_path": m.get("file_path"), "section": m.get("section")})

    result = {"draft": draft, "citations": cits, "used_hits": used_hits, "engine": engine, "language": req.language or DEFAULT_LANGUAGE}
    _cache_set(cache_key, result)
    return result

from fastapi.responses import StreamingResponse

@app.post("/compose_stream")
def compose_stream(req: ComposeRequest):
    """
    以 SSE 流式輸出：
      data: {"token": "..."}  # 多次
      data: {"citations":[...], "used_hits": N, "engine":"..."}  # 最後一次
    """
    # 檢索
    hits = _search_internal(
        query=req.query,
        k=req.k,
        namespace=req.namespace,
        canonicality=req.canonicality,
        rerank=req.rerank,
    )
    # 選取片段過濾
    sel = getattr(req, "selected_ids", None)
    if sel:
        idset = set(sel)
        hits = [h for h in hits if h.get("id") in idset]


    # 無命中：以 SSE 格式回友善提示（避免 500）
    if not hits:
        def empty_gen():
            yield f'data: {json.dumps({"token":"[沒有命中片段，請調整查詢或資料集] "})}\n\n'
            yield f'data: {json.dumps({"citations": [], "used_hits": 0, "engine": (req.engine or "auto")})}\n\n'
        return StreamingResponse(empty_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","Connection":"keep-alive"})

    # 組上下文
    try:
        lang_line = _language_hint(req.language)
        context, _used_hits = _build_context(
            hits,
            max_chars=MAX_CONTEXT_CHARS,
            summarize=SUMMARIZE_CONTEXT,
            lang_line=lang_line,
        )
    except Exception as e:
        def err_gen():
            yield f'data: {json.dumps({"token": f"[context 構建錯誤] {e}"})}\n\n'
            yield f'data: {json.dumps({"citations": [], "used_hits": 0, "engine": (req.engine or "auto")})}\n\n'
        return StreamingResponse(err_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","Connection":"keep-alive"})

    def event_gen():
        # 先發 keepalive（避免某些 proxy 提早關閉）
        yield ":\n\n"  # SSE 注釋行
        try:
            engine = (req.engine or "").lower()
            if engine == "ollama":
                yield from _stream_with_ollama(req.query, context, req.mode, req.language)
                final_engine = "ollama"
            elif engine == "openai":
                yield from _stream_with_openai(req.query, context, req.mode, req.language)
                final_engine = "openai"
            else:
                # 自動選擇：優先本地（可依你現有優先序調整）
                if OLLAMA_MODEL:
                    yield from _stream_with_ollama(req.query, context, req.mode, req.language)
                    final_engine = "ollama"
                elif OPENAI_API_KEY:
                    yield from _stream_with_openai(req.query, context, req.mode, req.language)
                    final_engine = "openai"
                else:
                    yield f'data: {json.dumps({"token":"[沒有可用的 LLM 後端：請設定 OLLAMA_MODEL 或 OPENAI_API_KEY] "})}\n\n'
                    final_engine = "none"
            # 正常收尾：補 citations / used_hits / engine
            cits = [{"id": h.get("id"),
                     "file_path": (h.get("metadata") or {}).get("file_path"),
                     "section": (h.get("metadata") or {}).get("section")} for h in hits]
            yield f"data: {json.dumps({'citations': cits, 'used_hits': len(hits), 'engine': final_engine})}\n\n"
        except Exception as e:
            # 串流期間例外 → 以 SSE 帶給前端而不是 500
            yield f'data: {json.dumps({"token": f"[stream error] {e}"})}\n\n'
            cits = [{"id": h.get("id"),
                     "file_path": (h.get("metadata") or {}).get("file_path"),
                     "section": (h.get("metadata") or {}).get("section")} for h in hits]
            yield f"data: {json.dumps({'citations': cits, 'used_hits': len(hits), 'engine': (req.engine or 'auto')})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","Connection":"keep-alive"})
