"""基于 SQLite 的去重：记录见过的文章 hash。"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .fetch import Article

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    hash TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    first_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_first_seen ON seen(first_seen);
"""


class SeenStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def filter_new(self, articles: list[Article]) -> list[Article]:
        """过滤掉已经见过的文章，同一批次内也去重。"""
        if not articles:
            return []
        hashes = [a.hash_key for a in articles]
        placeholders = ",".join("?" for _ in hashes)
        cur = self.conn.execute(
            f"SELECT hash FROM seen WHERE hash IN ({placeholders})", hashes
        )
        seen = {row[0] for row in cur.fetchall()}

        fresh: list[Article] = []
        batch_seen: set[str] = set()
        for a in articles:
            if a.hash_key in seen or a.hash_key in batch_seen:
                continue
            batch_seen.add(a.hash_key)
            fresh.append(a)
        log.info(
            "dedupe: %d in -> %d new (filtered %d)",
            len(articles), len(fresh), len(articles) - len(fresh),
        )
        return fresh

    def mark_seen(self, articles: list[Article]) -> None:
        rows = [a.to_row() for a in articles]
        self.conn.executemany(
            "INSERT OR IGNORE INTO seen(hash, url, title, first_seen) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        log.info("dedupe: marked %d articles as seen", len(rows))

    def total(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
