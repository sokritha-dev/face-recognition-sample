# db/logger.py
import sqlite3
from datetime import datetime


def setup_logger(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        confidence REAL,
        timestamp TEXT
    )""")


def log_match(conn, name, confidence):
    timestamp = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO logs (name, confidence, timestamp) VALUES (?, ?, ?)",
        (name, confidence, timestamp),
    )
    conn.commit()
