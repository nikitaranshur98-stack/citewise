import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

DB_PATH = "chat_history.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    with closing(_connect()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                briefing_json TEXT,
                checks_json TEXT,
                tools_json TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        
        
def create_chat(title):
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO chats (title, created_at) VALUES (?, ?)",
            (title[:80], _now()),
        )
        return cursor.lastrowid


def add_message(chat_id, role, content, briefing=None, checks=None, tools=None):
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            """INSERT INTO messages
               (chat_id, role, content, briefing_json, checks_json, tools_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                chat_id,
                role,
                content,
                briefing.model_dump_json() if briefing else None,
                json.dumps([c.model_dump() for c in checks]) if checks else None,
                json.dumps(tools) if tools else None,
                _now(),
            ),
        )
        return cursor.lastrowid
    
def list_chats():
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM chats ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_messages(chat_id):
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def delete_chat(chat_id):
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

