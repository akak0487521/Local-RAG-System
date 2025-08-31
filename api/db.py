"""SQLite conversation database utilities."""

from __future__ import annotations

import sqlite3
import time
from typing import List, Optional, Tuple

from .config import DB_PATH

__all__ = [
    "_db",
    "_init_db",
    "save_message",
    "load_recent_messages",
    "get_summary",
    "set_summary",
]


def _db() -> sqlite3.Connection:
    """Return a new connection to the conversation database."""
    return sqlite3.connect(DB_PATH)


def _init_db() -> None:
    """Create required tables and indexes if they do not exist."""
    conn = _db()
    # conversation history
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT, ts INTEGER, role TEXT, content TEXT, lang TEXT
    )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS summaries(
        thread_id TEXT PRIMARY KEY, summary TEXT, lang TEXT, updated_ts INTEGER
    )"""
    )
    # built-in knowledge base
    conn.execute(
        """CREATE TABLE IF NOT EXISTS kb_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT, ref_id TEXT, title TEXT, content TEXT, updated_ts INTEGER
    )"""
    )
    # FTS5 index
    conn.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts
        USING fts5(title, content, content='kb_items', content_rowid='id')"""
    )
    # triggers to sync base table to FTS
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS kb_ai AFTER INSERT ON kb_items BEGIN
        INSERT INTO kb_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
    END;"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS kb_ad AFTER DELETE ON kb_items BEGIN
        INSERT INTO kb_fts(kb_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
    END;"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS kb_au AFTER UPDATE ON kb_items BEGIN
        INSERT INTO kb_fts(kb_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
        INSERT INTO kb_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
    END;"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id)"
    )
    conn.commit()
    conn.close()


def save_message(thread_id: str, role: str, content: str, lang: Optional[str]) -> None:
    """Persist a message belonging to a conversation thread."""
    conn = _db()
    conn.execute(
        "INSERT INTO messages(thread_id,ts,role,content,lang) VALUES(?,?,?,?,?)",
        (thread_id, int(time.time()), role, content, (lang or "")),
    )
    conn.commit()
    conn.close()


def load_recent_messages(
    thread_id: str, max_turns: int = 6, max_chars: int = 1200
) -> List[Tuple[str, str]]:
    """Retrieve recent messages for a thread limited by turns and characters."""
    conn = _db()
    cur = conn.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT ?",
        (thread_id, max_turns * 2),
    )
    rows = cur.fetchall()
    conn.close()
    rows = rows[::-1]  # old -> new
    out, size = [], 0
    for role, content in rows[::-1]:  # take newest backwards until cap
        c = content or ""
        if size + len(c) > max_chars:
            break
        out.append((role, c))
        size += len(c)
    return out[::-1]


def get_summary(thread_id: str) -> str:
    """Fetch the stored summary for a conversation thread."""
    conn = _db()
    cur = conn.execute(
        "SELECT summary FROM summaries WHERE thread_id=?", (thread_id,)
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""


def set_summary(thread_id: str, summary: str, lang: Optional[str]) -> None:
    """Upsert the summary for a conversation thread."""
    conn = _db()
    conn.execute(
        """INSERT INTO summaries(thread_id,summary,lang,updated_ts)
                    VALUES(?,?,?,?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                    summary=excluded.summary, lang=excluded.lang, updated_ts=excluded.updated_ts""",
        (thread_id, summary, (lang or ""), int(time.time())),
    )
    conn.commit()
    conn.close()
