# scripts/bulk_import.py
# Scan docs/ recursively and upsert into Chroma. Prints a diff-like report.
import os, json, glob
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PERSIST_DIR = os.getenv("PERSIST_DIR", "./vector_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
COLLECTION_NAME = "gamefantasy"

client = chromadb.PersistentClient(path=PERSIST_DIR)
embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

def normalize(x):
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            parts.append(f"SECTION: {k}\n{normalize(v)}")
        return "\n\n".join(parts)
    return str(x)

def to_chunks(doc: dict, fp: str):
    import hashlib, os, json
    base_id = doc.get("id") or hashlib.md5((fp + json.dumps(doc, sort_keys=True)).encode("utf-8")).hexdigest()
    namespace = doc.get("namespace", os.path.basename(os.path.dirname(fp)) or "default")
    canonicality = doc.get("canonicality", "canon")
    title = doc.get("title", os.path.splitext(os.path.basename(fp))[0])
    summary = doc.get("summary", "")
    body = doc.get("body", {})

    file_path = os.path.relpath(fp, DOCS_DIR).replace("\\", "/")
    chunks = []
    chunks.append({
        "id": f"{base_id}::summary",
        "text": f"{title}\n\n{summary}",
        "metadata": {"namespace": namespace, "canonicality": canonicality, "source_id": base_id, "section": "summary", "title": title, "file_path": file_path}
    })
    if isinstance(body, dict):
        for k, v in body.items():
            txt = normalize(v)
            if txt.strip():
                chunks.append({
                    "id": f"{base_id}::{k}",
                    "text": f"{title} — {k}\n\n{txt}",
                    "metadata": {"namespace": namespace, "canonicality": canonicality, "source_id": base_id, "section": k, "title": title, "file_path": file_path}
                })
    else:
        txt = normalize(body)
        if txt.strip():
            chunks.append({
                "id": f"{base_id}::body",
                "text": f"{title}\n\n{txt}",
                "metadata": {"namespace": namespace, "canonicality": canonicality, "source_id": base_id, "section": "body", "title": title, "file_path": file_path}
            })
    return chunks

def upsert(chunks):
    if not chunks: return
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

def main():
    files = glob.glob(os.path.join(DOCS_DIR, "**", "*.json"), recursive=True)
    if not files:
        print("No JSON under docs/.")
        return
    total_docs = 0
    for fp in files:
        try:
            data = json.load(open(fp, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[SKIP] {fp} -> {e}")
            continue
        if isinstance(data, list):
            for d in data:
                upsert(to_chunks(d, fp)); total_docs += 1
        else:
            upsert(to_chunks(data, fp)); total_docs += 1
        print(f"[UPSERT] {os.path.relpath(fp, DOCS_DIR)}")
    print(f"Done. Logical docs upserted: {total_docs}")

if __name__ == "__main__":
    main()
