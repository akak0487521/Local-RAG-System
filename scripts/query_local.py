import argparse, os, json
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PERSIST_DIR = os.getenv("PERSIST_DIR", "./vector_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "gamefantasy"

def main():
    parser = argparse.ArgumentParser(description="Local RAG CLI query tool")
    parser.add_argument("query", type=str, help="Your query text")
    parser.add_argument("--k", type=int, default=5, help="Top-k results (1-20)")
    parser.add_argument("--namespace", type=str, default=None, help="Filter by metadata.namespace")
    parser.add_argument("--canonicality", type=str, default=None, choices=["canon","semi","non"], help="Filter by canonicality")
    parser.add_argument("--json", action="store_true", help="Output JSON for programmatic use")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

    where = {}
    if args.namespace: where["namespace"] = args.namespace
    if args.canonicality: where["canonicality"] = args.canonicality

    res = col.query(
        query_texts=[args.query],
        n_results=max(1, min(args.k, 20)),
        where=where if where else None,
        include=["documents","metadatas","distances","uris"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0]

    hits = []
    for i in range(len(docs)):
        hit = {
            "rank": i+1,
            "id": ids[i],
            "score": float(dists[i]),
            "metadata": metas[i],
            "text": docs[i]
        }
        hits.append(hit)

    if args.json:
        print(json.dumps({"hits": hits, "count": len(hits)}, ensure_ascii=False, indent=2))
    else:
        for h in hits:
            print("="*100)
            print(f"[{h['rank']}] id={h['id']}  score={h['score']:.4f}")
            meta = h.get("metadata", {})
            print(f"namespace={meta.get('namespace')}  canonicality={meta.get('canonicality')}  section={meta.get('section')}")
            print(f"title={meta.get('title')}  file_path={meta.get('file_path')}  source_id={meta.get('source_id')}")
            print("-"*100)
            print(h["text"])
            print()

if __name__ == "__main__":
    main()
