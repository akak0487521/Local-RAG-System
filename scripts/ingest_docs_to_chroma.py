# ingest_docs_to_chroma.py
import os, json, glob, chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PERSIST_DIR = os.getenv("PERSIST_DIR", "./vector_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gamefantasy")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=PERSIST_DIR)
ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

ids, docs, metas = [], [], []
for path in glob.glob("/app/docs/**/*.json", recursive=True):
    try:
        j = json.load(open(path, "r", encoding="utf-8"))
        text = j.get("content") or j.get("text") or j.get("body") or ""
        title = j.get("title") or os.path.basename(path)
        if not text.strip(): continue
        _id = j.get("id") or path
        meta = {
            "title": title,
            "file_path": path.replace("/app/", ""),
            "namespace": j.get("namespace") or "",
            "canonicality": j.get("canonicality") or "",
        }
        ids.append(str(_id)); docs.append(text); metas.append(meta)
    except Exception: pass

if ids:
    col.add(ids=ids, documents=docs, metadatas=metas)
    print("added:", len(ids))
else:
    print("no docs found")
