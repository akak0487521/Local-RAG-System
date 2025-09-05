import json
import pathlib
import sys
import importlib

import pytest
from fastapi.testclient import TestClient

# Ensure root path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


class DummyEmbeddingFunction:
    def __call__(self, texts):
        return [[0.0] * 3 for _ in texts]


class DummyCollection:
    def __init__(self):
        self.items = {}

    def add(self, ids, embeddings, documents, metadatas):
        for i, doc, meta in zip(ids, documents, metadatas):
            self.items[i] = {"doc": doc, "meta": meta}

    def delete(self, ids=None, where=None):
        if ids:
            for i in ids:
                self.items.pop(i, None)
        elif where and "source_key" in where:
            to_del = [i for i, v in self.items.items() if v["meta"].get("source_key") == where["source_key"]]
            for i in to_del:
                self.items.pop(i, None)

    def query(self, *args, **kwargs):
        return {"documents": []}


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.collection = DummyCollection()

    def get_or_create_collection(self, name, embedding_function=None):
        return self.collection


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    persist_dir = tmp_path / "persist"
    kb_db = tmp_path / "kb.sqlite"

    monkeypatch.setenv("DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("PERSIST_DIR", str(persist_dir))
    monkeypatch.setenv("KB_DB_PATH", str(kb_db))
    monkeypatch.setenv("API_KEY", "testkey")

    import chromadb
    from chromadb.utils import embedding_functions

    monkeypatch.setattr(chromadb, "PersistentClient", lambda *a, **kw: DummyClient())
    monkeypatch.setattr(
        embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        lambda model_name: DummyEmbeddingFunction(),
    )

    import api.config as config_module
    import api.app as app_module
    importlib.reload(config_module)
    importlib.reload(app_module)
    import api.routes.kb as kb_module
    importlib.reload(kb_module)

    store = {}
    calls = {"index": [], "delete": []}

    def fake_index(payload):
        store[payload["id"]] = payload
        calls["index"].append(payload["id"])

    def fake_delete(doc_id, src_key=None):
        store.pop(doc_id, None)
        calls["delete"].append((doc_id, src_key))

    for module in (app_module, kb_module):
        monkeypatch.setattr(module, "_index_doc_to_stores", fake_index)
        monkeypatch.setattr(module, "_delete_doc_from_stores", fake_delete)

    client = TestClient(app_module.app)
    return client, store, calls, docs_dir


def test_docs_crud(api_client):
    client, store, calls, docs_dir = api_client
    headers = {"x-api-key": "testkey"}

    # create
    resp = client.post(
        "/docs/save",
        json={"title": "t", "content": "c"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    doc_id = data["id"]
    path = docs_dir / data["file"]
    assert doc_id in store and path.exists()

    # get success
    resp = client.get(f"/docs/{doc_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["title"] == "t"

    # get not found
    resp = client.get("/docs/none", headers=headers)
    assert resp.status_code == 404

    # update
    resp = client.put(
        f"/docs/{doc_id}",
        json={"title": "t2", "content": "c2"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert doc_id in store and store[doc_id]["content"] == "c2"
    assert calls["delete"][0][0] == doc_id
    updated = json.loads(path.read_text("utf-8"))
    assert updated["title"] == "t2"

    # update not found
    resp = client.put(
        "/docs/missing",
        json={"title": "x", "content": "y"},
        headers=headers,
    )
    assert resp.status_code == 404

    # delete
    resp = client.delete(f"/docs/{doc_id}", headers=headers)
    assert resp.status_code == 200
    assert doc_id not in store and not path.exists()
    assert calls["delete"][-1][0] == doc_id

    # delete not found
    resp = client.delete("/docs/none", headers=headers)
    assert resp.status_code == 404


def test_docs_save_validation_error(api_client):
    client, store, calls, docs_dir = api_client
    headers = {"x-api-key": "testkey"}

    resp = client.post("/docs/save", json={"content": "c"}, headers=headers)
    assert resp.status_code == 422
