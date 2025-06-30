import json
import numpy as np
from sqlite3 import Connection


class FaceDatabase:
    def __init__(self, conn: Connection):
        self.conn = conn
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                name TEXT,
                embedding TEXT
            )
        """)
        self.conn.commit()

    def insert_face(self, name: str, embedding: np.ndarray):
        embedding_json = json.dumps(embedding.tolist())
        self.conn.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding_json)
        )
        self.conn.commit()

    def load_embeddings(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, embedding FROM faces")
        rows = cur.fetchall()
        return [
            (id, name, np.array(json.loads(embedding))) for id, name, embedding in rows
        ]
