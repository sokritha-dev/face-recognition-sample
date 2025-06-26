import sqlite3
import numpy as np
import json

def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY,
        name TEXT,
        embedding TEXT
    )''')
    return conn

def load_embeddings(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, name, embedding FROM faces")
    return [(id, name, np.array(json.loads(embedding))) for id, name, embedding in cur.fetchall()]

def insert_face(conn, name, embedding):
    cur = conn.cursor()
    cur.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, json.dumps(embedding.tolist())))
    conn.commit()
