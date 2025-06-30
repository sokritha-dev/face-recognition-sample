# db/logger.py
from datetime import datetime
from sqlite3 import Connection


class MatchLoggerDB:
    def __init__(self, conn: Connection):
        self.conn = conn
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                confidence REAL,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def log(self, name, confidence):
        timestamp = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO logs (name, confidence, timestamp) VALUES (?, ?, ?)",
            (name, confidence, timestamp),
        )
        self.conn.commit()
