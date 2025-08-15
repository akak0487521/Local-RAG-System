# scripts/import_chat_history.py
# Convert past chat logs (ChatGPT export conversations.json, JSONL, or folder of conversations.json)
# into indexable JSON docs. Supports interactive selection with preview.
#
# Usage examples:
#   python scripts/import_chat_history.py --input conversations.json --interactive
#   python scripts/import_chat_history.py --input my_threads.jsonl --out-namespace history --canonicality non
#
import os, re, json, argparse, datetime, shutil, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "./docs"))

def slugify(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_").lower() or "untitled"

def ensure_text(x) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, dict):
        parts = x.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if p is not None)
        if "content" in x and isinstance(x["content"], dict) and "parts" in x["content"]:
            ps = x["content"]["parts"]
            if isinstance(ps, list):
                return "\n".join(str(p) for p in ps if p is not None)
    if isinstance(x, list):
        return "\n".join(ensure_text(i) for i in x)
    return str(x)

def parse_conversations_json(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    threads = []
    if isinstance(raw, dict) and "conversations" in raw:
        items = raw["conversations"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]

    for conv in items:
        title = conv.get("title") or conv.get("conversation_id") or "untitled"
        thread_id = conv.get("id") or conv.get("conversation_id") or slugify(title)
        messages = []
        if "messages" in conv and isinstance(conv["messages"], list):
            for m in conv["messages"]:
                role = (m.get("author") or {}).get("role") or m.get("role") or ""
                content = m.get("content")
                text = ensure_text(content if content is not None else m.get("text"))
                ts = m.get("create_time") or m.get("timestamp")
                if role in ("user","assistant","system") and text:
                    messages.append({"role": role, "content": text, "create_time": ts})
        elif "mapping" in conv and isinstance(conv["mapping"], dict):
            nodes = []
            for _, node in conv["mapping"].items():
                msg = node.get("message") or {}
                role = (msg.get("author") or {}).get("role")
                text = ensure_text((msg.get("content") or {}).get("parts", []))
                ts = msg.get("create_time")
                if role in ("user","assistant","system") and text:
                    nodes.append({"role": role, "content": text, "create_time": ts})
            nodes.sort(key=lambda x: (x.get("create_time") or 0))
            messages.extend(nodes)
        else:
            continue
        threads.append({"thread_id": str(thread_id), "title": title, "messages": messages})
    return threads

def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            thread_id = obj.get("thread_id") or "thread_" + slugify(obj.get("title",""))
            title = obj.get("title") or thread_id
            messages = obj.get("messages") or []
            norm = []
            for m in messages:
                role = m.get("role")
                text = ensure_text(m.get("content"))
                ts = m.get("create_time") or obj.get("created_at")
                if role in ("user","assistant") and text:
                    norm.append({"role": role, "content": text, "create_time": ts})
            out.append({"thread_id": thread_id, "title": title, "messages": norm})
    return out

def pair_messages(messages: List[Dict[str,Any]]):
    pairs = []
    last_user = None
    for m in messages:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if not text: continue
        if role == "user":
            last_user = m
        elif role == "assistant" and last_user is not None:
            pairs.append((last_user, m))
            last_user = None
    return pairs

def to_json_doc(thread_id: str, title: str, u: Dict[str,Any], a: Dict[str,Any], out_ns: str, canon: str, tags: List[str]) -> Dict[str,Any]:
    human = (u.get("content") or "").strip()
    ai = (a.get("content") or "").strip()
    ts = a.get("create_time") or u.get("create_time")
    updated = datetime.date.today().isoformat()
    if ts:
        try:
            if isinstance(ts, (int,float)):
                updated = datetime.datetime.fromtimestamp(ts).date().isoformat()
            elif isinstance(ts, str) and ts.isdigit():
                updated = datetime.datetime.fromtimestamp(int(ts)).date().isoformat()
        except Exception:
            pass

    title_guess = human.splitlines()[0][:80] if human else title
    file_id = slugify(f"{thread_id}__{title_guess}")
    body = {
        "prompt": human,
        "draft": ai,
        "thread_id": thread_id
    }
    return {
        "id": file_id,
        "namespace": out_ns,
        "type": "draft",
        "title": title_guess or "未命名",
        "summary": f"匯入自對話：{title}；thread_id={thread_id}",
        "body": body,
        "tags": tags,
        "canonicality": canon,
        "version": "1.0",
        "updated_at": updated
    }

def save_doc(doc: dict, out_root: Path) -> Path:
    tid = doc["body"].get("thread_id") or "thread"
    target = out_root / "history" / tid
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{doc['id']}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def import_docs():
    import subprocess
    runner = None
    if (Path("scripts") / "bulk_import.py").exists():
        runner = ["python", "scripts/bulk_import.py"]
    elif (Path("scripts") / "build_index.py").exists():
        runner = ["python", "scripts/build_index.py"]
    if runner:
        try:
            subprocess.run(runner, check=True)
        except Exception as e:
            print(f"[WARN] Index update failed: {e}")

def preview(prompt: str, draft: str, width: int = 120, pmax: int = 600, dmax: int = 1200) -> str:
    def trunc(t, n): 
        t = t.strip()
        return (t[:n] + " …(truncated)") if len(t) > n else t
    p = trunc(prompt or "", pmax)
    d = trunc(draft or "", dmax)
    lines = []
    lines.append("="*width)
    lines.append("USER PROMPT:")
    lines.append(p)
    lines.append("-"*width)
    lines.append("ASSISTANT DRAFT:")
    lines.append(d)
    lines.append("="*width)
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Import past chat prompts+generations into indexable JSON docs")
    ap.add_argument("--input", required=True, help="conversations.json | .jsonl | folder path containing conversations.json")
    ap.add_argument("--out-namespace", default="history", help="namespace for saved docs")
    ap.add_argument("--canonicality", default="non", choices=["canon","semi","non"])
    ap.add_argument("--tags", nargs="*", default=["imported","chat"])
    ap.add_argument("--no-import", action="store_true", help="skip updating vector index after saving")
    ap.add_argument("--interactive", action="store_true", help="preview each pair and confirm import [y/N]")
    args = ap.parse_args()

    inp = Path(args.input)
    threads = []
    if inp.is_file() and inp.name.endswith(".jsonl"):
        threads = parse_jsonl(inp)
    elif inp.is_file() and inp.name.endswith(".json"):
        threads = parse_conversations_json(inp)
    elif inp.is_dir():
        if (inp / "conversations.json").exists():
            threads = parse_conversations_json(inp / "conversations.json")
        else:
            print("[WARN] Folder mode: please provide conversations.json; skipping.")
            return
    else:
        print("[ERROR] Unknown input. Provide ChatGPT conversations.json, a JSONL, or a folder containing conversations.json.")
        return

    saved = 0
    total_pairs = sum(len(pair_messages(t.get("messages", []))) for t in threads)
    idx = 0

    for t in threads:
        pairs = pair_messages(t.get("messages", []))
        for (u, a) in pairs:
            idx += 1
            prompt = (u.get("content") or "").strip()
            draft = (a.get("content") or "").strip()

            if args.interactive:
                print(preview(prompt, draft))
                ans = input(f"[{idx}/{total_pairs}] 匯入這一組嗎？ [y/N/a=all/q=quit] ").strip().lower()
                if ans == "q":
                    print("中止。")
                    if saved and not args.no_import:
                        import_docs()
                    return
                if ans == "a":
                    # turn off interactive for the rest
                    args.interactive = False
                elif ans != "y":
                    continue

            doc = to_json_doc(t["thread_id"], t.get("title",""), u, a, args.out_namespace, args.canonicality, args.tags)
            path = save_doc(doc, DOCS_DIR)
            print(f"[OK] {path}")
            saved += 1

    print(f"Saved {saved} docs.")
    if not args.no_import:
        import_docs()

if __name__ == "__main__":
    main()
