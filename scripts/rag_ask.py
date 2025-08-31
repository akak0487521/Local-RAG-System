#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A tiny CLI that:
# 1) Queries your local RAG API (/search)
# 2) Feeds top-k hits into OpenAI to generate an answer

import os, sys, argparse, requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

# --- Config (env with sensible defaults) ---
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/search")
RAG_API_KEY = os.getenv("API_KEY", "changeme")  # reuse .env API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if OPENAI_API_KEY is None:
    print("ERROR: Please set OPENAI_API_KEY in your environment (or .env).")
    sys.exit(1)

try:
    from openai import OpenAI
except Exception:
    print("ERROR: Missing `openai` package. Install with: pip install openai>=1.40.0")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

def query_rag(query: str, k: int = 6, namespace: Optional[str] = None, canonicality: Optional[str] = None) -> Dict[str, Any]:
    payload = {\"query\": query, \"k\": k}
    if namespace: payload[\"namespace\"] = namespace
    if canonicality: payload[\"canonicality\"] = canonicality
    headers = {}
    if RAG_API_KEY and RAG_API_KEY != \"changeme\":
        headers[\"x-api-key\"] = RAG_API_KEY
    try:
        resp = requests.post(RAG_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f\"ERROR calling RAG API: {e}\")
        sys.exit(2)
    return resp.json()

def build_context(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for h in hits:
        meta = h.get(\"metadata\", {}) or {}
        src = meta.get(\"file_path\") or meta.get(\"title\") or h.get(\"id\") or \"unknown_source\"
        section = meta.get(\"section\")
        tag = f\"{src}\" + (f\"#{section}\" if section else \"\")
        parts.append(f\"[{tag}]\\n{h.get('text','').strip()}\")
    return \"\\n\\n---\\n\\n\".join(parts)

SYS_INSTR = (
    \"你是《遊戲幻想版》資料管家。\"
    \"請僅根據提供的資料片段作答；若資料不足，明確指出缺口並說明需要的欄位。\"
    \"回答最後列出使用到的來源標籤（檔名#section）。\"
)

def ask_llm(question: str, context: str) -> str:
    prompt = (
        f\"{SYS_INSTR}\\n\\n\"
        f\"=== 資料片段開始 ===\\n{context}\\n=== 資料片段結束 ===\\n\\n\"
        f\"問題：{question}\\n\"
        f\"請用條列與小段落清楚回答。\"
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {\"role\": \"system\", \"content\": \"You are a precise retrieval-augmented assistant.\"},
            {\"role\": \"user\", \"content\": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def main():
    ap = argparse.ArgumentParser(description=\"RAG -> OpenAI answerer\")
    ap.add_argument(\"question\", type=str, help=\"你的問題（會先打本地 RAG 再送 LLM）\")
    ap.add_argument(\"--k\", type=int, default=6, help=\"Top-k 檢索數量 (1-20)\")
    ap.add_argument(\"--namespace\", type=str, default=None, help=\"過濾 metadata.namespace\")
    ap.add_argument(\"--canonicality\", type=str, default=None, choices=[\"canon\",\"semi\",\"non\"], help=\"過濾 canonicality\")
    args = ap.parse_args()

    rag = query_rag(args.question, k=max(1, min(args.k, 20)), namespace=args.namespace, canonicality=args.canonicality)
    hits = rag.get(\"hits\", [])
    if not hits:
        print(\"（RAG 無命中，請確認已建索引或調整問題/命名空間）\")
        sys.exit(0)

    context = build_context(hits)
    answer = ask_llm(args.question, context)

    print(\"\\n================= RAG x LLM 回覆 =================\\n\")
    print(answer)
    print(\"\\n================= 參考片段（前 {} 筆） =================\\n\".format(len(hits)))
    for i, h in enumerate(hits, 1):
        meta = h.get(\"metadata\", {}) or {}
        src = meta.get(\"file_path\") or meta.get(\"title\") or h.get(\"id\") or \"unknown_source\"
        sec = meta.get(\"section\")
        print(f\"[{i}] {src}\" + (f\"#{sec}\" if sec else \"\"))

if __name__ == \"__main__\":
    main()
