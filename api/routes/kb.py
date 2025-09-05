from fastapi import APIRouter, Security, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import uuid, json, time

from ..models import SaveDocItem, SearchRequest
from ..config import DOCS_DIR, FILTER_META_DEFAULT
from ..app import (
    _auth,
    api_key_header,
    content_hash,
    _slug,
    _index_doc_to_stores,
    _delete_doc_from_stores,
    upsert_kb_item,
    search_kb_fts,
    _query_chroma,
    _filter_meta_hits,
    _rerank,
    _highlight,
)

router = APIRouter()


def _find_doc(doc_id: str):
    for fp in Path(DOCS_DIR).rglob("*.json"):
        try:
            data = json.loads(fp.read_text("utf-8"))
        except Exception:
            continue
        if data.get("id") == doc_id:
            return fp, data
    return None, None


@router.get("/docs/list")
def docs_list(api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    docs = []
    if not Path(DOCS_DIR).exists():
        return {"docs": docs}
    base = Path(DOCS_DIR)
    for fp in base.rglob("*.json"):
        try:
            data = json.loads(fp.read_text("utf-8"))
            rel_path = fp.relative_to(base)
            folder = str(rel_path.parent)
            docs.append({
                "id": data.get("id"),
                "title": data.get("title"),
                "metadata": data.get("metadata") or {},
                "path": str(rel_path),
                "folder": folder,
                "file": rel_path.name,
            })
        except Exception:
            continue
    docs.sort(key=lambda x: x.get("metadata", {}).get("updated_ts", 0), reverse=True)
    return {"docs": docs}


@router.get("/docs/{doc_id}")
def docs_get(doc_id: str, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    path, data = _find_doc(doc_id)
    if not data:
        raise HTTPException(404, "doc not found")
    return data


@router.put("/docs/{doc_id}")
def docs_put(doc_id: str, item: SaveDocItem, background_tasks: BackgroundTasks, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    path, old = _find_doc(doc_id)
    if not old:
        raise HTTPException(404, "doc not found")

    now_ts = int(time.time())
    src_key = item.metadata.get("source_key") if isinstance(item.metadata, dict) else None
    if not src_key:
        src_key = _slug(item.title)
    c_hash = content_hash(item.content)

    metadata = dict(item.metadata or {})
    metadata.update({
        "updated_ts": now_ts,
        "source_key": src_key,
        "content_hash": c_hash,
    })

    payload = {
        "id": doc_id,
        "title": item.title,
        "content": item.content,
        "metadata": metadata,
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    old_src = (old.get("metadata") or {}).get("source_key")
    background_tasks.add_task(_delete_doc_from_stores, doc_id, old_src)
    background_tasks.add_task(_index_doc_to_stores, payload)
    return {"ok": True, "id": doc_id, "file": path.name, "path": str(path)}


@router.delete("/docs/{doc_id}")
def docs_delete(doc_id: str, background_tasks: BackgroundTasks, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    path, data = _find_doc(doc_id)
    if not data:
        raise HTTPException(404, "doc not found")
    path.unlink()
    src_key = (data.get("metadata") or {}).get("source_key")
    background_tasks.add_task(_delete_doc_from_stores, doc_id, src_key)
    return {"ok": True}


@router.post("/docs/save")
def docs_save(item: SaveDocItem, background_tasks: BackgroundTasks, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)

    Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    doc_id = f"doc-{ts}-{uuid.uuid4().hex[:8]}"
    fname = f"{ts}_{_slug(item.title)}.json"

    now_ts = int(time.time())
    src_key = item.metadata.get("source_key") if isinstance(item.metadata, dict) else None
    if not src_key:
        src_key = _slug(item.title)
    c_hash = content_hash(item.content)

    metadata = dict(item.metadata or {})
    metadata.update({
        "updated_ts": now_ts,
        "source_key": src_key,
        "content_hash": c_hash,
    })

    payload = {
        "id": doc_id,
        "title": item.title,
        "content": item.content,
        "metadata": metadata,
    }

    (Path(DOCS_DIR) / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    background_tasks.add_task(_index_doc_to_stores, payload)
    return {"ok": True, "id": doc_id, "file": fname, "path": str(Path(DOCS_DIR) / fname)}


@router.post("/kb/upsert")
def kb_upsert(item: Dict[str, str], api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    for f in ("source", "ref_id", "title", "content"):
        if f not in item:
            raise HTTPException(400, f"missing field: {f}")
    upsert_kb_item(item["source"], item["ref_id"], item["title"], item["content"])
    return {"ok": True}


@router.post("/kb/search")
def kb_search(body: Dict[str, Any], api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    q = (body.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "query required")
    k = int(body.get("k") or 5)
    return {"hits": search_kb_fts(q, limit=k)}


@router.post("/search")
def search(req: SearchRequest, api_key: Optional[str] = Security(api_key_header)):
    _auth(api_key)
    out = _query_chroma(req)
    hits = out.get("hits", [])

    try:
        db_hits = search_kb_fts(req.query, limit=max(1, (req.k or 6) // 2))
        for h in db_hits:
            h["metadata"] = {**h.get("metadata", {}), "title": h.get("title", ""), "source": "db"}

        do_filter = FILTER_META_DEFAULT and not ((req.canonicality or "").strip().lower() == "meta")
        hits = _filter_meta_hits(hits, do_filter) + _filter_meta_hits(db_hits, do_filter)
    except Exception as e:
        out["db_error"] = str(e)

    if req.rerank:
        try:
            hits = _rerank(req.query, hits)
            out["reranked"] = True
        except Exception as e:
            out["reranked"] = False
            out["rerank_error"] = str(e)

    if req.highlight:
        for h in hits:
            try:
                h["highlights"] = _highlight(req.query, h.get("text", ""))
            except Exception:
                h["highlights"] = []

    out["hits"] = hits[: max(1, req.k or 5)]
    out["source"] = "chroma+db"
    return out
