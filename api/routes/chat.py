from fastapi import APIRouter, Security, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json, time, os

from ..models import ComposeRequest
from ..config import MAX_CONTEXT_CHARS, OPENAI_MODEL, OLLAMA_MODEL
from ..llm import generate
from ..app import (
    _auth,
    api_key_header,
    _db,
    save_message,
    get_summary,
    set_summary,
    _search_internal,
    _norm_lang,
    _format_history_block,
    _build_context_lang,
    _prepare_messages,
    _summarize_chunk_to_lang,
    _preview_messages,
)

router = APIRouter()


@router.get("/threads")
def list_threads(limit: int = 200, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    conn = _db()
    cur = conn.execute(
        """
        SELECT thread_id, MAX(ts) AS last_ts, COUNT(*) AS cnt
        FROM messages GROUP BY thread_id ORDER BY last_ts DESC LIMIT ?
        """,
        (int(limit),),
    )
    items = [{"thread_id": tid, "last_ts": ts, "count": cnt} for (tid, ts, cnt) in cur.fetchall()]
    conn.close()
    return {"threads": items}


@router.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str, limit: int = 500, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    conn = _db()
    cur = conn.execute(
        "SELECT ts, role, content, lang FROM messages WHERE thread_id=? ORDER BY id ASC LIMIT ?",
        (thread_id, int(limit)),
    )
    rows = cur.fetchall()
    conn.close()
    msgs = [{"ts": ts, "role": role, "content": content, "lang": lang} for (ts, role, content, lang) in rows]
    return {"thread_id": thread_id, "messages": msgs, "summary": get_summary(thread_id)}


@router.post("/compose")
def compose(req: ComposeRequest, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    hits = _search_internal(req.query, req.k, req.namespace, req.canonicality, req.rerank)
    if not hits:
        return {"draft": "", "citations": [], "note": "無檢索命中；請調整 query 或新增資料。"}
    thread_id = req.thread_id or f"auto-{int(time.time()*1000)}"
    lang = _norm_lang(req.language)
    save_message(thread_id, "user", req.query, lang)

    history_block = _format_history_block(thread_id, lang, max_turns=6, max_chars=1200)
    rag_context, used_hits = _build_context_lang(hits, language=lang, max_chars=MAX_CONTEXT_CHARS, max_chars_per_chunk=600)
    combined_ctx = (f"<<<HISTORY_START>>>\n{history_block}\n<<<HISTORY_END>>>\n\n" if history_block else "") + rag_context

    messages, temperature = _prepare_messages(
        req.query,
        combined_ctx,
        req.mode,
        lang,
        target_length=req.target_length,
        style=req.style,
    )
    try:
        draft, engine = generate(
            messages,
            engine=req.engine,
            temperature=temperature,
            max_tokens=req.max_tokens,
            num_predict=req.num_predict,
        )
    except Exception as e:
        raise HTTPException(500, f"LLM generate failed: {e}")

    save_message(thread_id, "assistant", draft, lang)
    prev = get_summary(thread_id)
    update_src = f"Previous summary:\n{prev}\n\nNew exchange:\nU: {req.query}\nA: {draft}\n"
    new_summary = _summarize_chunk_to_lang(update_src, lang, max_chars_per_chunk=800)
    if new_summary:
        set_summary(thread_id, new_summary, lang)

    cits = []
    for h in hits:
        m = h.get("metadata", {}) or {}
        cits.append({"id": h.get("id"), "file_path": m.get("file_path"), "section": m.get("section")})
    return {"draft": draft, "citations": cits, "used_hits": used_hits, "engine": engine, "language": lang, "thread_id": thread_id}


@router.post("/compose_stream")
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
        def init_fail():
            yield "data: " + json.dumps({"token": f"[compose_stream init error] {e}"}, ensure_ascii=False) + "\n\n"
        return StreamingResponse(init_fail(), media_type="text/event-stream", headers=headers)

    def event_stream():
        assistant_accum: List[str] = []
        try:
            messages, temperature = _prepare_messages(
                req.query,
                combined_ctx,
                req.mode,
                lang,
                target_length=req.target_length,
                style=req.style,
            )
            yield f"data: {json.dumps({'token': ''})}\n\n"
            stream, final_engine = generate(
                messages,
                engine=req.engine,
                stream=True,
                temperature=temperature,
                max_tokens=req.max_tokens,
                num_predict=req.num_predict,
            )
            if debug_prompts:
                model = OPENAI_MODEL if final_engine == 'openai' else OLLAMA_MODEL
                preview = _preview_messages(final_engine, model, messages)
                yield "data: " + json.dumps({"debug": preview}, ensure_ascii=False) + "\n\n"
            for token in stream:
                assistant_accum.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f'data: {json.dumps({"token": f"[compose_stream error] {e}"})}\n\n'
            return

        try:
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
                "thread_id": thread_id,
            }
            yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f'data: {json.dumps({"token": f"[compose_stream error] {e}"})}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
