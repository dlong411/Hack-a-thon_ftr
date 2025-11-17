import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in environment")
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Create pgvector extension and base tables (documents, telegram_groups, users).

    Note: this will attempt to create the `vector` extension (pgvector). If the DB user
    doesn't have permission to create extensions, run the extension installation as a superuser.
    """
    conn = get_conn()
    cur = conn.cursor()
    # Create extension if not exists
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Documents table with embedding vector. Embedding dimension is flexible; using 1536 by default
    # (OpenAI embeddings use 1536 for some models). Adjust if you use a different model.
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        metadata JSONB,
        embedding vector(1536),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    )

    # Telegram groups table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS telegram_groups (
        id SERIAL PRIMARY KEY,
        tg_id BIGINT,
        name TEXT,
        description TEXT,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    )

    # Users table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE,
        full_name TEXT,
        is_admin BOOLEAN DEFAULT FALSE,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    )

    conn.commit()
    cur.close()
    conn.close()


def insert_document(title: str, content: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None) -> int:
    """Insert a document with optional embedding. Returns the inserted row id."""
    conn = get_conn()
    cur = conn.cursor()
    metadata_json = json.dumps(metadata or {})
    if embedding:
        # Insert embedding as PostgreSQL array - cast to vector in SQL for pgvector
        cur.execute(
            "INSERT INTO documents (title, content, metadata, embedding) VALUES (%s, %s, %s, %s) RETURNING id",
            (title, content, metadata_json, embedding),
        )
    else:
        cur.execute(
            "INSERT INTO documents (title, content, metadata) VALUES (%s, %s, %s) RETURNING id",
            (title, content, metadata_json),
        )
    _id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return _id


def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, title, content, metadata, created_at FROM documents WHERE id = %s", (doc_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def update_document(doc_id: int, title: Optional[str] = None, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    updates = []
    params = []
    if title is not None:
        updates.append("title = %s")
        params.append(title)
    if content is not None:
        updates.append("content = %s")
        params.append(content)
    if metadata is not None:
        updates.append("metadata = %s")
        params.append(json.dumps(metadata))
    if not updates:
        cur.close()
        conn.close()
        return False
    params.append(doc_id)
    sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = %s"
    cur.execute(sql, tuple(params))
    conn.commit()
    cur.close()
    conn.close()
    return True


def delete_document(doc_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    changed = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return changed > 0


def list_documents(limit: int = 20):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, title, metadata, created_at FROM documents ORDER BY created_at DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def search_similar_by_embedding(embedding: List[float], k: int = 5):
    """Return top-k similar documents using pgvector <-> operator (Euclidean distance).

    This function expects the `embedding` to be a Python list of floats. The SQL performs a cast
    to `vector` to compare against the stored embeddings.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    # Build SQL that casts the passed array to vector. The exact cast syntax may vary; psycopg2
    # will send the list as a PostgreSQL array and pgvector accepts array to vector casting.
    cur.execute(
        "SELECT id, title, content, metadata, created_at FROM documents ORDER BY embedding <-> %s::vector LIMIT %s",
        (embedding, k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def add_telegram_group(tg_id: int, name: str, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    metadata_json = json.dumps(metadata or {})
    cur.execute(
        "INSERT INTO telegram_groups (tg_id, name, description, metadata) VALUES (%s, %s, %s, %s) RETURNING id",
        (tg_id, name, description, metadata_json),
    )
    _id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return _id


def list_telegram_groups(limit: int = 50):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, tg_id, name, description, metadata, created_at FROM telegram_groups ORDER BY created_at DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def delete_telegram_group(group_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM telegram_groups WHERE id = %s", (group_id,))
    changed = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return changed > 0


if __name__ == "__main__":
    print("Initializing DB (creating tables and extension).")
    init_db()
    print("Done.")
