"""
Database module for session and message management using SQLite
"""
import os
import sqlite3
import uuid
from datetime import datetime
from contextlib import contextmanager
from typing import List, Optional, Dict

# SQLite Database Setup
DB_PATH = "/workspace/data/chat_history.db"


def init_database():
    """Initialize SQLite database with sessions and messages tables"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            engine TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized")


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_session(title: Optional[str] = None, engine: Optional[str] = None, model: Optional[str] = None) -> str:
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title, engine, model) VALUES (?, ?, ?, ?)",
            (session_id, title or "New Chat", engine, model)
        )
        conn.commit()
    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    """Get session by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, engine, model, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "title": row["title"],
                "engine": row["engine"],
                "model": row["model"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
    return None


def list_sessions(limit: int = 50) -> List[Dict]:
    """List all sessions ordered by updated_at"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, engine, model, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]


def delete_session(session_id: str) -> bool:
    """Delete a session and its messages"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0


def add_message(session_id: str, role: str, content: str) -> int:
    """Add a message to a session"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        # Update session's updated_at
        cursor.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )
        conn.commit()
        return cursor.lastrowid


def get_session_messages(session_id: str) -> List[Dict]:
    """Get all messages for a session"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def update_session_model(session_id: str, engine: str, model: str):
    """Update session's engine and model"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET engine = ?, model = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (engine, model, session_id)
        )
        conn.commit()

