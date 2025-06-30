import sqlite3


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    if db is None:
        raise ValueError(f"Could not connect to database at {db_path}")

    return db
