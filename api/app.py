# api/app.py — RAG API with SQLite conversation memory + KB FTS + multilingual context-normalization + robust SSE
from fastapi import FastAPI, HTTPException, Security, BackgroundTasks, Body
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import os, re, json, time, hashlib, logging, sqlite3, uuid, math

from .config import (
    API_KEY_ENV,
    COLLECTION_NAME,
    DB_PATH,
    DEFAULT_LANGUAGE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    FILTER_META_DEFAULT,
    HALF_LIFE_DAYS,
    KB_DB_PATH,
    MAX_CONTEXT_CHARS,
    META_TAGS,
    MODEL_PRIORITY,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_USE_CHAT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PERSIST_DIR,
    READONLY_MODE,
    RERANK_MODEL,
)

# ---------- DB (SQLite) ----------

def _db():
    conn = sqlite3.connect(DB_PATH)
    return conn

def _init_db():
    conn = _db()
    # 對話歷史
    conn.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT, ts INTEGER, role TEXT, content TEXT, lang TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS summaries(
        thread_id TEXT PRIMARY KEY, summary TEXT, lang TEXT, updated_ts INTEGER
    )""")
    # 內建知識庫（可扁平化業務資料）
    conn.execute("""CREATE TABLE IF NOT EXISTS kb_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT, ref_id TEXT, title TEXT, content TEXT, updated_ts INTEGER
    )""")
    # FTS5 索引
    conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts
        USING fts5(title, content, content='kb_items', content_rowid='id')""")
    # 觸發器同步原表→FTS
    conn.execute("""CREATE TRIGGER IF NOT EXISTS kb_ai AFTER INSERT ON kb_items BEGIN
        INSERT INTO kb_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
    END;""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS kb_ad AFTER DELETE ON kb_items BEGIN
        INSERT INTO kb_fts(kb_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
    END;""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS kb_au AFTER UPDATE ON kb_items BEGIN
        INSERT INTO kb_fts(kb_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
        INSERT INTO kb_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
    END;""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id)")

    conn.commit(); conn.close()
_init_db()

def save_message(thread_id: str, role: str, content: str, lang: Optional[str]):
    conn = _db()
    conn.execute("INSERT INTO messages(thread_id,ts,role,content,lang) VALUES(?,?,?,?,?)",
                 (thread_id, int(time.time()), role, content, (lang or "")))
    conn.commit(); conn.close()

def load_recent_messages(thread_id: str, max_turns: int = 6, max_chars: int = 1200) -> List[Tuple[str,str]]:
    conn = _db()
    cur = conn.execute("SELECT role, content FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT ?",
                       (thread_id, max_turns*2))
    rows = cur.fetchall()
    conn.close()
    rows = rows[::-1]  # old -> new
    out, size = [], 0
    for role, content in rows[::-1]:  # take newest backwards until cap
        c = content or ""
        if size + len(c) > max_chars: break
        out.append((role, c)); size += len(c)
    return out[::-1]

def get_summary(thread_id: str) -> str:
    conn = _db()
    cur = conn.execute("SELECT summary FROM summaries WHERE thread_id=?", (thread_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""

def set_summary(thread_id: str, summary: str, lang: Optional[str]):
    conn = _db()
    conn.execute("""INSERT INTO summaries(thread_id,summary,lang,updated_ts)
                    VALUES(?,?,?,?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                    summary=excluded.summary, lang=excluded.lang, updated_ts=excluded.updated_ts""",
                 (thread_id, summary, (lang or ""), int(time.time())))
    conn.commit(); conn.close()

def upsert_kb_item(source: str, ref_id: str, title: str, content: str):
    conn = _db()
    cur = conn.execute("SELECT id FROM kb_items WHERE source=? AND ref_id=?", (source, ref_id))
    row = cur.fetchone()
    if row:
        conn.execute("UPDATE kb_items SET title=?, content=?, updated_ts=? WHERE id=?",
                     (title, content, int(time.time()), row[0]))
    else:
        conn.execute("INSERT INTO kb_items(source, ref_id, title, content, updated_ts) VALUES(?,?,?,?,?)",
                     (source, ref_id, title, content, int(time.time())))
    conn.commit(); conn.close()

def search_kb_fts(query: str, limit: int = 5) -> List[Dict[str,Any]]:
    conn = _db()
    cur = conn.execute(
        "SELECT i.id, i.source, i.ref_id, i.title, i.content, i.updated_ts, "
        "bm25(kb_fts, 1.2, 0.75) as score "
        "FROM kb_fts JOIN kb_items i ON i.id = kb_fts.rowid "
        "WHERE kb_fts MATCH ? ORDER BY score LIMIT ?",
        (query, limit)
    )
    rows = cur.fetchall(); conn.close()
    hits = []
    for hid, src, rid, title, content, uts, score in rows:
        hits.append({
            "rank": 0,
            "id": f"db:{src}:{rid or hid}",
            "score": float(score) if score is not None else 0.0,
            "updated_ts": int(uts or 0),  # NEW: 帶上時間戳
            "metadata": {"source": src, "ref_id": rid, "title": title, "updated_ts": int(uts or 0)},  # NEW
            "text": content or "",
            "title": title or "",
        })
    return hits


# ---------- LLM clients ----------
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Vector store (Chroma) ----------
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
client = chromadb.PersistentClient(path=PERSIST_DIR)
embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

# ---------- Optional: cross-encoder for rerank (lazy) ----------
_CE = None
def get_cross_encoder():
    global _CE
    if _CE is None:
        try:
            from sentence_transformers import CrossEncoder
            _CE = CrossEncoder(RERANK_MODEL)
        except Exception as e:
            logger.warning(f"CrossEncoder init failed: {e}")
            _CE = None
    return _CE

# ---------- FastAPI ----------
app = FastAPI(title="Local RAG API (DB+RAG Integrated)", version="0.6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
if os.path.isdir("web"):
    app.mount("/app", StaticFiles(directory="web", html=True), name="web")

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

class SaveDocItem(BaseModel):
    title: str
    content: str
    metadata: dict = {}

# ---------- Helpers ----------
logger = logging.getLogger(__name__)

def _is_meta_hit(h: Dict[str, Any]) -> bool:  # NEW
    m = (h.get("metadata") or {})
    t = (m.get("type") or m.get("category") or "").strip().lower()
    ns = (m.get("namespace") or "").strip().lower()
    canon = (m.get("canonicality") or "").strip().lower()
    title = (m.get("title") or "").strip().lower()
    # 明確標記 meta / schema 類
    if canon == "meta" or ns == "meta":
        return True
    if t in META_TAGS:
        return True
    # 題名啟發式（避免誤殺：只判斷典型關鍵字）
    if any(k in title for k in ("curator", "schema", "prompt手冊", "prompt 指南", "系統說明", "資料庫說明")):
        return True
    return False

def _filter_meta_hits(hits: List[Dict[str, Any]], enabled: bool) -> List[Dict[str, Any]]:  # NEW
    if not enabled:
        return hits
    try:
        return [h for h in hits if not _is_meta_hit(h)]
    except Exception:
        return hits

def normalize_text(t: str) -> str:  # NEW
    return re.sub(r'\s+', ' ', (t or '')).strip()

def content_hash(t: str) -> str:  # NEW
    return hashlib.sha256(normalize_text(t).encode('utf-8')).hexdigest()

def _clamp(x, lo=0.0, hi=1.0):
    try: return max(lo, min(hi, float(x)))
    except (TypeError, ValueError): return 0.0

def _style_to_controller(style: Optional[Dict[str, Any]]) -> str:
    if not style: return ""
    tone = (style.get("tone") or "neutral").lower()
    d = _clamp(style.get("directness"), 0, 1)
    e = _clamp(style.get("empathy"), 0, 1)
    h = _clamp(style.get("hedging"), 0, 1)
    f = _clamp(style.get("formality"), 0, 1)
    return (
        "【語氣控制器】\n"
        f"- tone: {tone}\n- directness: {d:.2f}\n- empathy: {e:.2f}\n"
        f"- hedging: {h:.2f}\n- formality: {f:.2f}\n"
        "寫作規則：依使用者語言回覆；工程情境先結論後步驟；教學先共情一句再分步；"
        "拒絕時透明原因並提供至少兩個安全替代。"
    )

def _style_temperature(style: Optional[Dict[str, Any]], base: float = 0.4) -> float:
    if not style: return base
    tone = (style.get("tone") or "").lower()
    t = base
    if tone == "playful":       t = 0.8
    elif tone == "teacher":     t = 0.6
    elif tone == "expert":      t = 0.35
    elif tone == "journalistic":t = 0.45
    elif tone == "neutral":     t = base
    # 依「模糊緩衝」微調：越模糊→越有變化
    try: t = max(0.1, min(1.0, t + (float(style.get("hedging", 0)) - 0.3) * 0.2))
    except Exception: pass
    return t

def _slug(s: str) -> str:
    import re
    s = (s or "untitled").lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "-", s).strip("-")
    return s[:80] or "untitled"

def _sqlite_upsert(db_path: str, doc_id: str, title: str, text: str, meta: Dict[str, Any]):
    Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(id, title, text, metadata)")
        conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
        conn.execute(
            "INSERT INTO docs (id, title, text, metadata) VALUES (?, ?, ?, ?)",
            (doc_id, title, text, json.dumps(meta, ensure_ascii=False))
        )
        conn.commit()
    finally:
        conn.close()

def _chroma_add(doc_id: str, text: str, meta: Dict[str, Any]):
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except Exception as e:
        logger.warning(f"Chroma not available: {e}")
        return False
    try:
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
        col.add(ids=[doc_id], documents=[text], metadatas=[meta])
        return True
    except Exception as e:
        logger.warning(f"Chroma add failed: {e}")
        return False

def _index_doc_to_stores(payload: Dict[str, Any]):
    """在背景執行：寫入 SQLite FTS 與 Chroma（含去重＋版本控制）"""
    doc_id = payload.get("id")
    title = payload.get("title") or "untitled"
    text = payload.get("content") or ""
    meta = dict(payload.get("metadata") or {})
    src_key = meta.get("source_key") or _slug(title)
    upd_ts = int(meta.get("updated_ts") or time.time())
    c_hash = meta.get("content_hash") or content_hash(text)

    # --- Registry in KB_DB_PATH（記錄 content_hash 與 source_key -> doc_id） ---
    conn = sqlite3.connect(KB_DB_PATH)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS ingest_registry(content_hash TEXT PRIMARY KEY, last_doc_id TEXT, updated_ts INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS docs_registry(source_key TEXT PRIMARY KEY, doc_id TEXT, updated_ts INTEGER)")
        conn.commit()

        # 去重：同 hash 代表內容完全一致 → 直接跳過重嵌入（省資源）
        cur = conn.execute("SELECT last_doc_id FROM ingest_registry WHERE content_hash=?", (c_hash,))
        row = cur.fetchone()
        if row:
            # 已經索引過相同內容：仍更新 registry 的更新時間，但不重建索引
            conn.execute("UPDATE ingest_registry SET last_doc_id=?, updated_ts=? WHERE content_hash=?", (row[0], upd_ts, c_hash))
            conn.commit()
            return  # ← 跳過

        # 版本控制：同 source_key 僅保留最新
        cur = conn.execute("SELECT doc_id FROM docs_registry WHERE source_key=?", (src_key,))
        prev = cur.fetchone()
        if prev and prev[0] and prev[0] != doc_id:
            prev_id = prev[0]
            # 刪舊：SQLite FTS 'docs'
            try:
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(id, title, text, metadata)")
                conn.execute("DELETE FROM docs WHERE id=?", (prev_id,))
            except Exception:
                pass
            # 刪舊：Chroma（依 metadata.source_key）
            try:
                import chromadb
                client = chromadb.PersistentClient(path=PERSIST_DIR)
                col = client.get_or_create_collection(name=COLLECTION_NAME)
                col.delete(where={"source_key": src_key})
            except Exception as e:
                logger.warning(f"Chroma delete by source_key failed: {e}")

        # 寫入 SQLite FTS 與 Chroma（新版本）
        _sqlite_upsert(KB_DB_PATH, doc_id, title, text, {**meta, "title": title})
        _chroma_add(doc_id, text, {**meta, "title": title})

        # 更新 registry
        conn.execute("INSERT OR REPLACE INTO ingest_registry(content_hash, last_doc_id, updated_ts) VALUES (?,?,?)", (c_hash, doc_id, upd_ts))
        conn.execute("INSERT OR REPLACE INTO docs_registry(source_key, doc_id, updated_ts) VALUES (?,?,?)", (src_key, doc_id, upd_ts))
        conn.commit()
    finally:
        conn.close()

# --- Recency scoring（時間衰減） ---

def _hit_updated_ts(h: Dict[str, Any]) -> int:  # NEW
    m = h.get("metadata") or {}
    ts = h.get("updated_ts") or m.get("updated_ts")
    try: return int(ts)
    except Exception: return 0

def _base_similarity(h: Dict[str, Any]) -> float:  # NEW
    # 有 cross-encoder 分數優先；否則用向量距離轉相似度（1/(1+d)）
    if "rerank_score" in h:
        try: return float(h["rerank_score"])
        except Exception: return 0.0
    d = h.get("score", None)
    try:
        if d is None: return 0.0
        return 1.0 / (1.0 + float(d))
    except Exception:
        return 0.0

def _mix_with_recency(hits: List[Dict[str, Any]], alpha: float = 0.7) -> List[Dict[str, Any]]:  # NEW
    if not hits: return hits
    # 基礎相似度 0~1 正規化
    sims = [_base_similarity(h) for h in hits]
    min_s, max_s = min(sims), max(sims)
    if max_s > min_s:
        sims = [(s - min_s) / (max_s - min_s) for s in sims]
    # 時間衰減（越新越接近 1）
    now = time.time()
    recs = []
    for h in hits:
        ts = _hit_updated_ts(h)
        days = max(0.0, (now - (ts or 0)) / 86400.0)
        recency = 0.5 ** (days / HALF_LIFE_DAYS) if ts else 0.0
        recs.append(recency)
    # 混分
    mixed = []
    for h, s, r in zip(hits, sims, recs):
        h = dict(h)
        h["recency_score"] = r
        h["mixed_score"] = alpha * s + (1 - alpha) * r
        mixed.append(h)
    mixed.sort(key=lambda x: x.get("mixed_score", 0.0), reverse=True)
    # 重寫 rank
    for i, h in enumerate(mixed, 1):
        h["rank"] = i
    return mixed

def _preview_messages(engine: str, model: str, messages: list, max_len: int = 1200):
    def trunc(val) -> str:
        try:
            s = "" if val is None else (val if isinstance(val, str) else str(val))
        except Exception:
            s = repr(val)
        return s if len(s) <= max_len else s[:max_len] + f"...[+{len(s)-max_len} chars]"
    return {"engine": engine, "model": model, "messages": [{"role": m.get("role"), "content": trunc(m.get("content"))} for m in messages]}

def backend_available(name: str) -> bool:
    n = (name or "").lower()
    return (n == "openai" and bool(OPENAI_API_KEY)) or (n == "ollama" and bool(OLLAMA_MODEL))

def choose_backend(request_engine: Optional[str]) -> str:
    if request_engine:
        eng = request_engine.lower()
        if backend_available(eng): return eng
        raise HTTPException(status_code=400, detail=f"Requested engine '{request_engine}' not available.")
    for eng in MODEL_PRIORITY:
        if backend_available(eng): return eng
    raise HTTPException(status_code=500, detail="No LLM backend available (set OPENAI_API_KEY or OLLAMA_MODEL).")

# ---------- Language helpers ----------
def _norm_lang(lang: Optional[str]) -> str:
    l = (lang or "").strip().lower().replace("_", "-")
    mapping = {
        "zh": "zh", "zh-hant": "zh-tw", "zh-tw": "zh-tw", "zh-hk": "zh-tw", "zh-mo": "zh-tw", "zh-TW": "zh-tw",
        "zh-hans": "zh-cn", "zh-cn": "zh-cn", "zh-sg": "zh-cn",
        "ja": "ja", "ja-jp": "ja",
        "ko": "ko", "ko-kr": "ko",
        "en": "en", "en-us": "en", "en-gb": "en",
        "fr": "fr", "de": "de", "es": "es",
    }
    return mapping.get(l, (os.environ.get("DEFAULT_LANGUAGE") or "zh-tw").lower())

def _language_policy(lang: Optional[str]) -> tuple[str, str]:
    l = _norm_lang(lang)
    if l == "zh-tw":
        return ("【語言規則】僅能使用繁體中文（台灣）作答；可保留英文專有名詞，但不得形成完整英文句子；引用英文需以繁中轉述；程式碼/命令可英文但解說文字一律繁中。",
                "務必嚴格遵守語言規則：只用繁體中文；不得輸出英文或簡體中文句子；專有名詞可英文但不可形成英文句；引用須以繁中轉述。")
    if l == "zh-cn":
        return ("【语言规则】仅能使用简体中文作答；可保留英文专有名词，但不得形成完整英文句子；引用英文需用简体中文转述；代码/命令可英文但说明文字必须中文。",
                "严格遵守语言规则：只用简体中文；不得输出英文句子；专有名词可英文但不可形成英文句；引用需中文转述。")
    if l == "ja":
        return ("【言語ルール】日本語のみで回答してください。固有名詞の英語表記は可ですが、英語の完全な文章は作らないでください。英語の引用は日本語で要約してください。コード/コマンドは英語でも構いませんが、解説は日本語で書いてください。",
                "言語ルールを厳守：日本語のみ。英語の文章を出力しない。固有名詞の英語表記は可だが英文は禁止。引用は日本語で要約すること。")
    if l == "ko":
        return ("【언어 규칙】오직 한국어로만 답변하세요. 고유명사는 영어 표기를 허용하지만, 완전한 영어 문장은 만들지 마세요. 영어 인용은 한국어로 요약하세요. 코드/명령은 영어 가능하나 설명은 반드시 한국어로 작성하세요.",
                "언어 규칙을 엄격히 준수: 한국어만 사용. 영어 문장 출력 금지. 고유명사 영어 표기는 허용하되 문장 금지. 인용은 한국어로 요약.")
    if l == "en":
        return ("Use ONLY English. Proper nouns may remain in their original form. Summarize quotations in English. Code/commands may be English; explanatory text must be English.",
                "Strictly use English only. No sentences in other languages. Proper nouns allowed; quotes must be summarized in English.")
    # fallback
    return ("Respond ONLY in the requested language. Proper nouns may keep original form. If quoting other languages, summarize them in the requested language.",
            "Strictly respond ONLY in the requested language. Do not switch languages.")

def _wrap_context(context: str) -> str:
    return f"<<<HISTORY+RAG CONTEXT>>>\n{context}\n<<<END>>>"

def _build_user_content(query: str, context: str, lang: Optional[str],
                        target_length: Optional[str], user_guard: str) -> str:
    l = _norm_lang(lang); ctx = _wrap_context(context)
    if l == "zh-tw":
        guide = f"字數約 {target_length}。" if target_length else ""
        return f"{user_guard}\n{ctx}\n請根據上述內容完成「{query}」。{guide}\n輸出可用段落或條列，務必遵守語言規則。"
    if l == "zh-cn":
        guide = f"字数约 {target_length}。" if target_length else ""
        return f"{user_guard}\n{ctx}\n请根据上述内容完成“{query}”。{guide}\n输出可以使用段落或项目符号。"
    if l == "ja":
        guide = f"目安の長さ: {target_length}。" if target_length else ""
        return f"{user_guard}\n{ctx}\n上記の内容に基づいて「{query}」を完成してください。{guide}\n段落または箇条書き可。"
    if l == "ko":
        guide = f"분량: 약 {target_length}." if target_length else ""
        return f"{user_guard}\n{ctx}\n위 내용을 바탕으로 ‘{query}’를 완성하세요. {guide}\n단락 또는 불릿 허용."
    guide = f"Target length: {target_length}." if target_length else ""
    return f"{user_guard}\n{ctx}\nComplete “{query}” based on the context above. {guide}\nUse paragraphs and/or bullet points."

# --- 片段轉寫快取 ---
_LANG_SUM_CACHE: Dict[str, str] = {}
def _cache_key(text: str, lang: str, max_chars: int) -> str:
    return hashlib.md5((text + "|" + lang + "|" + str(max_chars)).encode("utf-8")).hexdigest()

def _summarize_to_lang_openai(text: str, lang: str, max_chars: int = 600) -> Optional[str]:
    if not OPENAI_API_KEY: return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        l = _norm_lang(lang)
        guard = {
            "zh-tw": f"只用繁體中文轉述重點，不新增資訊，不逐字抄原文。約 {max_chars} 字。",
            "zh-cn": f"只用简体中文转述要点，不新增信息，不逐字抄原文。约 {max_chars} 字。",
            "ja":    f"日本語のみで要点を要約。新情報を追加せず、逐語的な複写をしない。約{max_chars}文字以内。",
            "ko":    f"한국어로 핵심을 요약. 새로운 정보 추가 금지, 원문 베껴쓰기 금지. 약 {max_chars}자.",
        }.get(l, f"Summarize in the requested language only. ~{max_chars} chars.")
        prompt = f"{guard}\n<<<CONTEXT>>>\n{text}\n<<<END>>>"
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

def _summarize_to_lang_ollama(text: str, lang: str, max_chars: int = 600) -> Optional[str]:
    if not OLLAMA_MODEL: return None
    try:
        import requests
        l = _norm_lang(lang)
        guard = {
            "zh-tw": f"只用繁體中文轉述重點，不新增資訊，不逐字抄原文。約 {max_chars} 字。",
            "zh-cn": f"只用简体中文转述要点，不新增信息，不逐字抄原文。约 {max_chars} 字。",
            "ja":    f"日本語のみで要点を要約してください。約{max_chars}文字以内。",
            "ko":    f"한국어로만 핵심을 요약하세요. 약 {max_chars}자.",
        }.get(l, f"Summarize in the requested language only. ~{max_chars} chars.")
        prompt = f"{guard}\n<<<CONTEXT>>>\n{text}\n<<<END>>>"
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        s = (data.get("response") or "").strip()
        return s or None
    except Exception:
        return None

def _summarize_chunk_to_lang(text: str, lang: Optional[str], max_chars_per_chunk: int = 600) -> str:
    l = _norm_lang(lang)
    key = _cache_key(text, l, max_chars_per_chunk)
    if key in _LANG_SUM_CACHE: return _LANG_SUM_CACHE[key]
    out = _summarize_to_lang_openai(text, l, max_chars=max_chars_per_chunk) or _summarize_to_lang_ollama(text, l, max_chars=max_chars_per_chunk)
    result = out if (out and isinstance(out, str)) else text
    _LANG_SUM_CACHE[key] = result
    return result

def _build_context_lang(hits: list, language: Optional[str], max_chars: int = 4000, max_chars_per_chunk: int = 600) -> Tuple[str, int]:
    pieces: List[str] = []; total = 0; used = 0
    for h in hits:
        raw = h.get("text") or h.get("summary") or h.get("content") or ""
        if not raw: continue
        summarized = _summarize_chunk_to_lang(raw, language, max_chars_per_chunk=max_chars_per_chunk)
        block = f"<<<CHUNK id={h.get('id','?')}>>>\n{summarized}\n<<<END_CHUNK>>>"
        sz = len(block)
        if total + sz > max_chars: break
        pieces.append(block); total += sz; used += 1
    return ("\n\n".join(pieces), used)

def _format_history_block(thread_id: str, language: Optional[str], max_turns=6, max_chars=1200) -> str:
    recent = load_recent_messages(thread_id, max_turns=max_turns, max_chars=max_chars)
    summ = get_summary(thread_id)
    parts = []
    if summ:
        s = _summarize_chunk_to_lang(summ, language, max_chars_per_chunk=400)
        parts.append(f"<<<HISTORY_SUMMARY>>>\n{s}\n<<<END_HISTORY_SUMMARY>>>")
    if recent:
        lines = []
        for role, content in recent:
            one = _summarize_chunk_to_lang(content, language, max_chars_per_chunk=200)
            tag = "U" if role == "user" else "A"
            lines.append(f"{tag}: {one}")
        parts.append("<<<HISTORY_RECENT>>>\n" + "\n".join(lines) + "\n<<<END_HISTORY_RECENT>>>")
    return "\n\n".join(parts) if parts else ""

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

# ---------- Chroma 檢索 ----------
def _query_chroma(req: SearchRequest) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if req.namespace: where["namespace"] = req.namespace
    if req.canonicality: where["canonicality"] = req.canonicality
    try:
        res = collection.query(
            query_texts=[req.query],
            n_results=max(1, min(req.k, 20)),
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(f"chroma query failed: {e}")
        return {"hits": []}
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0] if "ids" in res else [None] * len(docs)
    hits = []
    for i in range(len(docs)):
        hits.append({
            "rank": i + 1,
            "id": ids[i],
            "score": float(dists[i]) if dists and i < len(dists) and dists[i] is not None else 0.0,
            "metadata": metas[i] if metas and i < len(metas) else {},
            "text": docs[i],
        })
    return {"hits": hits}

def _rerank(query: str, hits: List[Dict]) -> List[Dict[str, Any]]:
    if not hits: return hits
    ce = get_cross_encoder()
    if not ce:  # 無模型→略過 rerank
        return hits
    try:
        pairs = [[query, h.get("text","")] for h in hits]
        scores = ce.predict(pairs).tolist()
    except Exception as e:
        logger.warning(f"CrossEncoder predict failed: {e}")
        return hits
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    for i, h in enumerate(hits, 1): h["rank"] = i
    return hits

def _search_internal(query: str, k: int, namespace: Optional[str], canonicality: Optional[str], rerank: bool):
    req = SearchRequest(query=query, k=k, namespace=namespace, canonicality=canonicality, rerank=False)
    doc_hits = _query_chroma(req)["hits"]
    db_hits = search_kb_fts(query, limit=max(1, (k or 6)//2))

    # 預設過濾 meta/schema 類（若 canonicality 明確指定 'meta' 則不過濾）
    do_filter = FILTER_META_DEFAULT and not ((canonicality or "").strip().lower() == "meta")
    doc_hits = _filter_meta_hits(doc_hits or [], do_filter)
    db_hits  = _filter_meta_hits(db_hits or [],  do_filter)

    hits = (doc_hits) + (db_hits)
    # （若你有 rerank / recency 混合的流程，照舊處理）
    # return hits[:max(k, 1)]
    return hits[:max(k, 1)]

def _hits_signature(hits: List[Dict[str, Any]]) -> str:
    basis = [{"id":h.get("id"), "score":round(float(h.get("score", 0.0)), 6)} for h in hits[:6]]
    return hashlib.md5(json.dumps(basis, sort_keys=True).encode()).hexdigest()

# ---------- SSE 整理 ----------
def _sse_normalize(chunk: str) -> List[str]:
    if not chunk: return []
    s = chunk.replace("\r\n", "\n")
    if not s.lstrip().startswith("data:"):
        return [f"data: {json.dumps({'token': s})}\n\n"]
    out: List[str] = []; i = 0
    while True:
        start = s.find("data:", i)
        if start == -1: break
        end = s.find("\n\n", start)
        if end != -1:
            out.append(s[start:end + 2]); i = end + 2; continue
        next_start = s.find("data:", start + 5)
        if next_start != -1:
            out.append(s[start:next_start].rstrip("\n") + "\n\n"); i = next_start
        else:
            out.append(s[start:].rstrip("\n") + "\n\n"); break
    return out

# ---------- System prompts ----------
STRICT_SYS = (
    "You are the Data Curator for 'Game Fantasy Edition'. "
    "Rely only on the provided context. If information is missing, list the exact nodes/fields needed. "
    "Do not speculate or invent facts. Keep responses clear and structured."
)
CREATIVE_SYS = (
    "You are the Author's Assistant for 'Game Fantasy Edition'. "
    "Write helpful, clear content grounded in the provided context. "
    "You may extend details only when they do not contradict the context. "
    "Avoid hallucinations; prefer concise paragraphs or bullet points."
)

# ---------- LLM backends ----------
def _compose_with_openai(query: str, context: str, mode: str, language: Optional[str],
                          target_length: Optional[str] = None, max_tokens: Optional[int] = None, style=None) -> str:
    if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY 未設定")
    sys_base = STRICT_SYS if (mode or "").lower() == "strict" else CREATIVE_SYS
    user_guard, system_guard = _language_policy(language)
    user_content = _build_user_content(query, context, language, target_length, user_guard)
    tone_ctrl = _style_to_controller(style.dict() if hasattr(style, "dict") else (style or {}))
    temperature = _style_temperature(style.dict() if hasattr(style, "dict") else (style or {}), base=0.4)

    for i in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_base},
                    {"role": "system", "content": system_guard},
                    ([{"role":"system","content":tone_ctrl}] if tone_ctrl else []),
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                **({"max_tokens": max_tokens} if isinstance(max_tokens, int) and max_tokens>0 else {}),
            )
            return resp.choices[0].message.content
        except Exception:
            if i == 2: raise
            time.sleep(1.5 * (i + 1))
    raise RuntimeError("OpenAI call failed after retries.")

def _compose_with_ollama(query: str, context: str, mode: str, language: Optional[str],
                         target_length: Optional[str] = None, num_predict: Optional[int] = None, style=None) -> str:
    import requests
    if not OLLAMA_MODEL: raise RuntimeError("OLLAMA_MODEL 未設定")
    sys_base = STRICT_SYS if (mode or "").lower() == "strict" else CREATIVE_SYS
    user_guard, system_guard = _language_policy(language)
    user_content = _build_user_content(query, context, language, target_length, user_guard)
    tone_ctrl = _style_to_controller(style.dict() if hasattr(style, "dict") else (style or {}))
    temperature = _style_temperature(style.dict() if hasattr(style, "dict") else (style or {}), base=0.4)

    if OLLAMA_USE_CHAT:
        payload = {"model": OLLAMA_MODEL, "stream": False,
                   "messages": [{"role":"system","content":sys_base},
                                {"role":"system","content":system_guard},
                                ([{"role":"system","content":tone_ctrl}] if tone_ctrl else []),
                                {"role":"user","content":user_content}],
                    "options": {"temperature": temperature, **({"num_predict": num_predict} if isinstance(num_predict,int) and num_predict>0 else {})}
                    }
        if isinstance(num_predict, int) and num_predict>0:
            payload["options"] = {"num_predict": num_predict}
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
        r.raise_for_status(); data = r.json()
        return data.get("message", {}).get("content") or data.get("response", "")
    else:
        prompt = f"{sys_base}\n\n{system_guard}\n{user_guard}\n{_wrap_context(context)}"
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        if isinstance(num_predict, int) and num_predict>0:
            payload["options"] = {"num_predict": num_predict}
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
        r.raise_for_status(); data = r.json()
        return data.get("response", "")

# 注意：簽名多了 style: Optional[StyleSpec] = None
def _stream_with_openai(
    query: str,
    context: str,
    mode: str,
    language: Optional[str],
    target_length: Optional[str] = None,
    max_tokens: Optional[int] = None,
    debug_prompts: bool = False,
    style: Optional[StyleSpec] = None
):
    yield f"data: {json.dumps({'token': ''})}\n\n"
    if not OPENAI_API_KEY:
        yield f"data: {json.dumps({'token': '[OpenAI 未設定 OPENAI_API_KEY]'})}\n\n"; return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        sys_base = STRICT_SYS if (mode or "").lower() == "strict" else CREATIVE_SYS
        user_guard, system_guard = _language_policy(language)
        user_content = _build_user_content(query, context, language, target_length, user_guard)

        # 語氣控制器與溫度
        style_dict = style.dict() if hasattr(style, "dict") else (style or {})
        tone_ctrl = _style_to_controller(style_dict)
        temperature = _style_temperature(style_dict, base=0.4)

        # 正確展開 messages（避免把 list 當成單一元素）
        messages = [
            {"role": "system", "content": sys_base},
            {"role": "system", "content": system_guard},
        ]
        if tone_ctrl:
            messages.append({"role": "system", "content": tone_ctrl})
        messages.append({"role": "user", "content": user_content})

        if debug_prompts:
            preview = _preview_messages("openai", OPENAI_MODEL, messages)
            yield "data: " + json.dumps({"debug": preview}, ensure_ascii=False) + "\n\n"

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,  # ← 使用語氣控制器推導出的溫度
            stream=True,
            **({"max_tokens": max_tokens} if isinstance(max_tokens, int) and max_tokens > 0 else {}),
        )

        for chunk in stream:
            delta = ""
            try:
                # 新舊 SDK 皆相容的取法
                delta = getattr(chunk.choices[0].delta, "content", None) or chunk.choices[0].delta.get("content", "")
            except Exception:
                pass
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'token': f'[OpenAI stream error] {e}'})}\n\n"

def _stream_with_ollama(
    query: str,
    context: str,
    mode: str,
    language: Optional[str],
    target_length: Optional[str] = None,
    num_predict: Optional[int] = None,
    debug_prompts: bool = False,
    style: Optional[StyleSpec] = None
):
    yield f"data: {json.dumps({'token': ''})}\n\n"
    import requests
    if not OLLAMA_MODEL:
        yield f"data: {json.dumps({'token': '[Ollama 未設定 OLLAMA_MODEL]'})}\n\n"; return

    sys_base = STRICT_SYS if (mode or "").lower() == "strict" else CREATIVE_SYS
    user_guard, system_guard = _language_policy(language)
    user_content = _build_user_content(query, context, language, target_length, user_guard)

    # 語氣控制器與溫度
    style_dict = style.dict() if hasattr(style, "dict") else (style or {})
    tone_ctrl = _style_to_controller(style_dict)
    temperature = _style_temperature(style_dict, base=0.4)

    messages = [
        {"role": "system", "content": sys_base},
        {"role": "system", "content": system_guard},
    ]
    if tone_ctrl:
        messages.append({"role": "system", "content": tone_ctrl})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": messages,
        "options": {"temperature": temperature}
    }
    if isinstance(num_predict, int) and num_predict > 0:
        payload["options"]["num_predict"] = num_predict

    if debug_prompts:
        preview = _preview_messages("ollama", OLLAMA_MODEL, messages)
        yield "data: " + json.dumps({"debug": preview}, ensure_ascii=False) + "\n\n"

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
                # 兼容不同版本的欄位
                msg = (data.get("message") or {}).get("content") or data.get("response")
                if msg:
                    yield f"data: {json.dumps({'token': msg})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'token': f'[Ollama stream error] {e}'})}\n\n"

# ---------- Endpoints ----------
@app.get("/")
def _root_to_app():
    return RedirectResponse(url="/app/")

@app.get("/health")
def health(api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    chroma_ok, cnt = True, None
    try: cnt = collection.count()
    except Exception: chroma_ok = False
    # 簡易 Ollama 檢查
    ollama_alive = False; ollama_models: List[str] = []
    try:
        import requests
        rr = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if rr.ok:
            ollama_alive = True
            jd = rr.json(); ollama_models = [t.get("name") for t in jd.get("models", [])]
    except Exception: pass
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "persist_dir": PERSIST_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
        "docs_count": cnt, "chroma_ok": chroma_ok,
        "language_default": DEFAULT_LANGUAGE,
        "backends": {
            "priority": MODEL_PRIORITY,
            "openai": {"enabled": bool(OPENAI_API_KEY), "model": OPENAI_MODEL},
            "ollama": {"enabled": bool(OLLAMA_MODEL), "model": OLLAMA_MODEL, "host": OLLAMA_HOST, "use_chat": OLLAMA_USE_CHAT, "alive": ollama_alive, "tags": ollama_models},
        },
    }

@app.get("/threads")
def list_threads(limit: int = 200, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    conn = _db()
    cur = conn.execute("""
        SELECT thread_id, MAX(ts) AS last_ts, COUNT(*) AS cnt
        FROM messages GROUP BY thread_id ORDER BY last_ts DESC LIMIT ?
    """, (int(limit),))
    items = [{"thread_id": tid, "last_ts": ts, "count": cnt} for (tid, ts, cnt) in cur.fetchall()]
    conn.close()
    return {"threads": items}

@app.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str, limit: int = 500, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    conn = _db()
    cur = conn.execute("SELECT ts, role, content, lang FROM messages WHERE thread_id=? ORDER BY id ASC LIMIT ?",
                       (thread_id, int(limit)))
    rows = cur.fetchall(); conn.close()
    msgs = [{"ts": ts, "role": role, "content": content, "lang": lang} for (ts, role, content, lang) in rows]
    return {"thread_id": thread_id, "messages": msgs, "summary": get_summary(thread_id)}

@app.post("/docs/save")
def docs_save(item: SaveDocItem, background_tasks: BackgroundTasks, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)

    Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    doc_id = f"doc-{ts}-{uuid.uuid4().hex[:8]}"
    fname = f"{ts}_{_slug(item.title)}.json"

    # NEW: 自動補 metadata 欄位
    now_ts = int(time.time())
    src_key = item.metadata.get("source_key") if isinstance(item.metadata, dict) else None
    if not src_key:
        src_key = _slug(item.title)  # 無則用標題 slug 當來源鍵
    c_hash = content_hash(item.content)

    metadata = dict(item.metadata or {})
    metadata.update({
        "updated_ts": now_ts,       # 供 recency 使用
        "source_key": src_key,      # 供版本控制（唯一鍵）
        "content_hash": c_hash,     # 供去重
    })

    payload = {
        "id": doc_id,
        "title": item.title,
        "content": item.content,
        "metadata": metadata,
    }

    (Path(DOCS_DIR) / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    background_tasks.add_task(_index_doc_to_stores, payload)  # 背景索引
    return {"ok": True, "id": doc_id, "file": fname, "path": str(Path(DOCS_DIR) / fname)}

@app.post("/kb/upsert")
def kb_upsert(item: Dict[str, str], api_key: Optional[str] = Security(api_key_header)):
    """最簡單的 KB upsert：{source, ref_id, title, content}"""
    _auth(api_key)
    for f in ("source","ref_id","title","content"):
        if f not in item: raise HTTPException(400, f"missing field: {f}")
    upsert_kb_item(item["source"], item["ref_id"], item["title"], item["content"])
    return {"ok": True}

@app.post("/kb/search")
def kb_search(body: Dict[str, Any], api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    q = (body.get("query") or "").strip()
    if not q: raise HTTPException(400, "query required")
    k = int(body.get("k") or 5)
    return {"hits": search_kb_fts(q, limit=k)}

# ========== /search：合併 Chroma + DB FTS 並過濾 ==========
@app.post("/search")
def search(req: SearchRequest, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    out = _query_chroma(req)
    hits = out.get("hits", [])

    try:
        db_hits = search_kb_fts(req.query, limit=max(1, (req.k or 6)//2))
        for h in db_hits:
            h["metadata"] = {**h.get("metadata", {}), "title": h.get("title", ""), "source": "db"}

        # 過濾 meta/schema 類（除非 canonicality=meta）
        do_filter = FILTER_META_DEFAULT and not ((req.canonicality or "").strip().lower() == "meta")
        hits = _filter_meta_hits(hits, do_filter) + _filter_meta_hits(db_hits, do_filter)

        # 如果你有 recency/score 混合排序，這裡可再處理
        # hits = _mix_with_recency(hits, alpha=0.7)

    except Exception as e:
        out["db_error"] = str(e)

    # 可選：rerank / highlight
    if req.rerank:
        try:
            hits = _rerank(req.query, hits); out["reranked"] = True
        except Exception as e:
            out["reranked"] = False; out["rerank_error"] = str(e)

    if req.highlight:
        for h in hits:
            try:
                h["highlights"] = _highlight(req.query, h.get("text", ""))
            except Exception:
                h["highlights"] = []

    out["hits"] = hits[:max(1, req.k or 5)]
    out["source"] = "chroma+db"
    return out

@app.post("/compose")
def compose(req: ComposeRequest, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    # 檢索（合併文件＋DB）
    hits = _search_internal(req.query, req.k, req.namespace, req.canonicality, req.rerank)
    if not hits:
        return {"draft": "", "citations": [], "note": "無檢索命中；請調整 query 或新增資料。"}
    thread_id = req.thread_id or f"auto-{int(time.time()*1000)}"
    lang = _norm_lang(req.language)
    save_message(thread_id, "user", req.query, lang)

    # 組歷史＋RAG
    history_block = _format_history_block(thread_id, lang, max_turns=6, max_chars=1200)
    rag_context, used_hits = _build_context_lang(hits, language=lang, max_chars=MAX_CONTEXT_CHARS, max_chars_per_chunk=600)
    combined_ctx = (f"<<<HISTORY_START>>>\n{history_block}\n<<<HISTORY_END>>>\n\n" if history_block else "") + rag_context

    engine = choose_backend(req.engine)
    sig = _hits_signature(hits)
    cache_key = (req.query, req.mode, lang, engine, sig, MAX_CONTEXT_CHARS, req.target_length)
    # 直接生成
    if engine == "openai":
        draft = _compose_with_openai(req.query, combined_ctx, req.mode, lang, target_length=req.target_length, max_tokens=req.max_tokens)
    elif engine == "ollama":
        draft = _compose_with_ollama(req.query, combined_ctx, req.mode, lang, target_length=req.target_length, num_predict=req.num_predict)
    else:
        raise HTTPException(500, f"未知的 engine: {engine}")

    save_message(thread_id, "assistant", draft, lang)
    prev = get_summary(thread_id)
    update_src = f"Previous summary:\n{prev}\n\nNew exchange:\nU: {req.query}\nA: {draft}\n"
    new_summary = _summarize_chunk_to_lang(update_src, lang, max_chars_per_chunk=800)
    if new_summary: set_summary(thread_id, new_summary, lang)

    cits = []
    for h in hits:
        m = h.get("metadata", {}) or {}
        cits.append({"id": h.get("id"), "file_path": m.get("file_path"), "section": m.get("section")})
    return {"draft": draft, "citations": cits, "used_hits": used_hits, "engine": engine, "language": lang, "thread_id": thread_id}

@app.post("/compose_stream")
def compose_stream(req: ComposeRequest, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    try:
        hits = _search_internal(req.query, req.k, req.namespace, req.canonicality, req.rerank)
        sel = getattr(req, "selected_ids", None)
        if sel:
            idset = set(sel)
            hits = [h for h in hits if h.get("id") in idset]

        thread_id = req.thread_id or f"auto-{int(time.time()*1000)}"
        lang = _norm_lang(req.language)
        save_message(thread_id, "user", req.query, lang)
        debug_prompts = bool(getattr(req, "debug", False) or os.environ.get("DEBUG_PROMPTS") in ("1","true","True"))

        history_block = _format_history_block(thread_id, lang, max_turns=6, max_chars=1200)
        rag_context, used_hits = _build_context_lang(hits, language=lang, max_chars=MAX_CONTEXT_CHARS, max_chars_per_chunk=600)
        combined_ctx = (f"<<<HISTORY_START>>>\n{history_block}\n<<<HISTORY_END>>>\n\n" if history_block else "") + rag_context
    except Exception as e:
        # 把初始化錯誤改成 SSE 回給前端，而不是 500
        def init_fail():
            yield "data: " + json.dumps({"token": f"[compose_stream init error] {e}"}, ensure_ascii=False) + "\n\n"
        return StreamingResponse(init_fail(), media_type="text/event-stream", headers=headers)

    def event_stream():
        assistant_accum: List[str] = []
        try:
            final_engine = None
            # 先走 Ollama（若指定或預設存在）
            if req.engine == "ollama" or (not req.engine and OLLAMA_MODEL):
                final_engine = "ollama"
                for chunk in _stream_with_ollama(
                    req.query, combined_ctx, req.mode, lang,
                    target_length=req.target_length,
                    num_predict=req.num_predict,
                    debug_prompts=debug_prompts,
                    style=req.style  # NEW: 傳入 style
                ):
                    # 收集 token
                    try:
                        if chunk.startswith("data:"):
                            obj = json.loads(chunk.split("data:", 1)[1])
                            if isinstance(obj, dict) and "token" in obj:
                                assistant_accum.append(obj["token"])
                    except Exception:
                        pass
                    for ev in _sse_normalize(chunk):
                        yield ev

            # 否則走 OpenAI（若指定或有 key）
            elif req.engine == "openai" or (not req.engine and OPENAI_API_KEY):
                final_engine = "openai"
                for chunk in _stream_with_openai(
                    req.query, combined_ctx, req.mode, lang,
                    target_length=req.target_length,
                    max_tokens=req.max_tokens,
                    debug_prompts=debug_prompts,
                    style=req.style  # NEW: 傳入 style
                ):
                    try:
                        if chunk.startswith("data:"):
                            obj = json.loads(chunk.split("data:", 1)[1])
                            if isinstance(obj, dict) and "token" in obj:
                                assistant_accum.append(obj["token"])
                    except Exception:
                        pass
                    for ev in _sse_normalize(chunk):
                        yield ev
            else:
                final_engine = "none"
                yield f'data: {json.dumps({"token":"[沒有可用的 LLM 後端]"})}\n\n'

            # 串流完成：寫回答＋更新摘要
            assistant_text = "".join(assistant_accum).strip()
            if assistant_text:
                save_message(thread_id, "assistant", assistant_text, lang)
                prev = get_summary(thread_id)
                update_src = f"Previous summary:\n{prev}\n\nNew exchange:\nU: {req.query}\nA: {assistant_text}\n"
                new_summary = _summarize_chunk_to_lang(update_src, lang, max_chars_per_chunk=800)
                if new_summary:
                    set_summary(thread_id, new_summary, lang)

            tail = {
                "citations": hits,
                "used_hits": used_hits,
                "engine": final_engine,
                "thread_id": thread_id
                # 需要也可回傳 style 設定供前端顯示：
                # "style": req.style.dict() if hasattr(req.style, "dict") and req.style else req.style
            }
            yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f'data: {json.dumps({"token": f"[compose_stream error] {e}"})}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
