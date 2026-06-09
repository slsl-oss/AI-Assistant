import sqlite3
import time
import os
from typing import List
from utils.path_tool import get_abs_path

DB_PATH = get_abs_path("memory_db", "memory_index.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init():
    with _conn() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS memory_scores (
                memory_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_user ON memory_scores(user_id)")
        db.commit()


def upsert(memory_hash: str, user_id: str, importance: float = 0.5):
    with _conn() as db:
        db.execute("""
            INSERT INTO memory_scores (memory_hash, user_id, importance, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_hash) DO UPDATE SET importance = excluded.importance
        """, (memory_hash, user_id, importance, time.time()))
        db.commit()


def batch_upsert(items: List[dict]):
    now = time.time()
    with _conn() as db:
        db.executemany("""
            INSERT INTO memory_scores (memory_hash, user_id, importance, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_hash) DO UPDATE SET importance = excluded.importance
        """, [(it["memory_hash"], it["user_id"], it.get("importance", 0.5), now) for it in items])
        db.commit()


def remove(memory_hash: str):
    with _conn() as db:
        db.execute("DELETE FROM memory_scores WHERE memory_hash = ?", (memory_hash,))
        db.commit()


def remove_batch(hashes: List[str]):
    with _conn() as db:
        db.executemany("DELETE FROM memory_scores WHERE memory_hash = ?", [(h,) for h in hashes])
        db.commit()


def get_all(user_id: str) -> List[dict]:
    with _conn() as db:
        rows = db.execute(
            "SELECT memory_hash, user_id, importance, created_at, access_count "
            "FROM memory_scores WHERE user_id = ?", (user_id,)
        ).fetchall()
    return [{"memory_hash": r[0], "user_id": r[1], "importance": r[2],
             "created_at": r[3], "access_count": r[4]} for r in rows]


def get_by_importance(user_id: str, threshold: float) -> List[str]:
    with _conn() as db:
        rows = db.execute(
            "SELECT memory_hash FROM memory_scores WHERE user_id = ? AND importance < ?",
            (user_id, threshold)
        ).fetchall()
    return [r[0] for r in rows]


def get_by_time(user_id: str, max_age_days: int) -> List[str]:
    cutoff = time.time() - max_age_days * 86400
    with _conn() as db:
        rows = db.execute(
            "SELECT memory_hash FROM memory_scores WHERE user_id = ? AND created_at < ?",
            (user_id, cutoff)
        ).fetchall()
    return [r[0] for r in rows]


def get_sorted_by_score(user_id: str) -> List[dict]:
    now = time.time()
    with _conn() as db:
        rows = db.execute(
            "SELECT memory_hash, user_id, importance, created_at, access_count "
            "FROM memory_scores WHERE user_id = ?", (user_id,)
        ).fetchall()
    items = []
    for r in rows:
        age_days = (now - r[3]) / 86400
        recency = max(0, 1 - age_days / 365)
        score = r[2] * 0.7 + recency * 0.3
        items.append({"memory_hash": r[0], "user_id": r[1],
                       "importance": r[2], "created_at": r[3],
                       "access_count": r[4], "score": score})
    items.sort(key=lambda x: x["score"])
    return items


def count(user_id: str) -> int:
    with _conn() as db:
        r = db.execute("SELECT COUNT(*) FROM memory_scores WHERE user_id = ?", (user_id,)).fetchone()
    return r[0] if r else 0


def has_hash(memory_hash: str) -> bool:
    with _conn() as db:
        r = db.execute("SELECT 1 FROM memory_scores WHERE memory_hash = ?", (memory_hash,)).fetchone()
    return r is not None


init()
