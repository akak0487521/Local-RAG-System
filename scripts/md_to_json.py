# scripts/md_to_json.py
import os, re, json, argparse, hashlib, datetime
from pathlib import Path

def slugify(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_").lower()

def parse_markdown(md: str):
    lines = md.splitlines()
    title = None
    summary_lines = []
    sections = {}
    current = None
    for line in lines:
        if re.match(r"^#\s+", line) and title is None:
            title = re.sub(r"^#\s+", "", line).strip()
            continue
        if re.match(r"^##\s+", line):
            current = re.sub(r"^##\s+", "", line).strip()
            sections[current] = []
            continue
        if current:
            sections[current].append(line)
        else:
            summary_lines.append(line)
    summary = "\n".join(l for l in summary_lines).strip()
    body = {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}
    return title or "未命名", summary, body

def main():
    ap = argparse.ArgumentParser(description="Convert Markdown in docs_raw/ to JSON schema in docs/")
    ap.add_argument("--src", default="docs_raw", help="source folder with .md files")
    ap.add_argument("--dst", default="docs", help="destination folder for .json files")
    ap.add_argument("--namespace_from_parent", action="store_true", help="set namespace from parent folder name if not provided")
    ap.add_argument("--canonicality", default="canon")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    count = 0
    for path in src.rglob("*.md"):
        md = path.read_text(encoding="utf-8")
        title, summary, body = parse_markdown(md)
        parent_ns = path.parent.name
        file_stem = slugify(path.stem)
        data = {
            "id": file_stem,
            "namespace": parent_ns if args.namespace_from_parent else "default",
            "type": "note",
            "title": title,
            "summary": summary,
            "body": body,
            "tags": [],
            "canonicality": args.canonicality,
            "version": "1.0",
            "updated_at": datetime.date.today().isoformat()
        }
        dst_path = dst / parent_ns / f"{file_stem}.json" if args.namespace_from_parent else dst / f"{file_stem}.json"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Converted: {path} -> {dst_path}")
        count += 1
    print(f"Done. Converted {count} file(s).")

if __name__ == "__main__":
    main()
