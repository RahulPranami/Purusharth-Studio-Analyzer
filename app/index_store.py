"""
Index Store module providing a replaceable abstraction for face embeddings storage.
Initial implementation uses SQLite, architected for straightforward migration
to Supabase PostgreSQL with pgvector.
"""

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.config import settings
from app.face_engine import FaceDetectionResult


class IndexStore(ABC):
    """Abstract interface for storing and querying face embeddings."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize database schema and indexes."""
        pass

    @abstractmethod
    def should_index(self, image_key: str, etag: Optional[str], force_reindex: bool = False) -> bool:
        """Check if an image should be indexed based on key, etag, and status."""
        pass

    @abstractmethod
    def record_indexed_image(
        self,
        image_key: str,
        collection_scope: str,
        etag: Optional[str],
        size_bytes: int,
        status: str,
        faces: List[FaceDetectionResult],
        model_version: str,
    ) -> None:
        """Save image processing status and all detected face embeddings."""
        pass

    @abstractmethod
    def record_failed_image(
        self,
        image_key: str,
        collection_scope: str,
        error_message: str,
    ) -> None:
        """Record image failure status."""
        pass

    @abstractmethod
    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        collection_scope: Optional[str] = None,
        threshold: float = 0.60,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search for matching face embeddings using cosine similarity."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return counts of indexed images, faces, and collections."""
        pass

    @abstractmethod
    def get_indexed_image(self, image_key: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a single indexed image."""
        pass


class SQLiteIndexStore(IndexStore):
    """SQLite implementation of IndexStore."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.sqlite_db_path
        self._ensure_dir()
        self.initialize()

    def _ensure_dir(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS indexed_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_key TEXT UNIQUE NOT NULL,
                    collection_scope TEXT NOT NULL DEFAULT 'default',
                    etag TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    status TEXT NOT NULL, -- 'indexed', 'failed', 'no_faces'
                    face_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_key TEXT NOT NULL,
                    collection_scope TEXT NOT NULL DEFAULT 'default',
                    face_index INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    model_version TEXT NOT NULL,
                    bounding_box TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(image_key) REFERENCES indexed_images(image_key) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_face_collection ON face_embeddings(collection_scope);
                CREATE INDEX IF NOT EXISTS idx_face_image_key ON face_embeddings(image_key);
                CREATE INDEX IF NOT EXISTS idx_img_collection ON indexed_images(collection_scope);
                """
            )

    def should_index(self, image_key: str, etag: Optional[str], force_reindex: bool = False) -> bool:
        if force_reindex:
            return True

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT etag, status FROM indexed_images WHERE image_key = ?",
                (image_key,),
            ).fetchone()

            if not row:
                return True

            # If it failed previously, allow retry
            if row["status"] == "failed":
                return True

            # If ETag is available, check if it matches
            if etag and row["etag"] == etag and row["status"] in ("indexed", "no_faces"):
                return False

            # If already successfully indexed without etag difference
            if not etag and row["status"] in ("indexed", "no_faces"):
                return False

            return True

    def record_indexed_image(
        self,
        image_key: str,
        collection_scope: str,
        etag: Optional[str],
        size_bytes: int,
        status: str,
        faces: List[FaceDetectionResult],
        model_version: str,
    ) -> None:
        with self._get_connection() as conn:
            # Upsert into indexed_images
            conn.execute(
                """
                INSERT INTO indexed_images (image_key, collection_scope, etag, size_bytes, status, face_count, error_message, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(image_key) DO UPDATE SET
                    collection_scope=excluded.collection_scope,
                    etag=excluded.etag,
                    size_bytes=excluded.size_bytes,
                    status=excluded.status,
                    face_count=excluded.face_count,
                    error_message=NULL,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (image_key, collection_scope, etag, size_bytes, status, len(faces)),
            )

            # Clear previous embeddings for this image
            conn.execute("DELETE FROM face_embeddings WHERE image_key = ?", (image_key,))

            # Insert new face embeddings
            for f in faces:
                emb_bytes = f.embedding.astype(np.float32).tobytes()
                box_json = json.dumps(f.bounding_box) if f.bounding_box else None
                conn.execute(
                    """
                    INSERT INTO face_embeddings (image_key, collection_scope, face_index, embedding, model_version, bounding_box, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (image_key, collection_scope, f.face_index, emb_bytes, model_version, box_json, f.confidence),
                )

    def record_failed_image(
        self,
        image_key: str,
        collection_scope: str,
        error_message: str,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO indexed_images (image_key, collection_scope, status, face_count, error_message, updated_at)
                VALUES (?, ?, 'failed', 0, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(image_key) DO UPDATE SET
                    status='failed',
                    error_message=excluded.error_message,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (image_key, collection_scope, error_message),
            )

    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        collection_scope: Optional[str] = None,
        threshold: float = 0.60,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity across stored face embeddings using vectorized matrix operations.
        """
        # Ensure query is unit normalized
        query = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 1e-6:
            query = query / norm

        query_sql = "SELECT id, image_key, face_index, embedding, bounding_box, confidence FROM face_embeddings"
        params: List[Any] = []
        if collection_scope and collection_scope != "*":
            query_sql += " WHERE collection_scope = ?"
            params.append(collection_scope)

        with self._get_connection() as conn:
            rows = conn.execute(query_sql, params).fetchall()

        if not rows:
            return []

        # Vectorized similarity computation
        matches: List[Dict[str, Any]] = []
        for row in rows:
            blob = row["embedding"]
            candidate_vec = np.frombuffer(blob, dtype=np.float32)
            if len(candidate_vec) != len(query):
                continue  # Skip mismatched vector dimensions

            # Cosine similarity for unit vectors is dot product
            similarity = float(np.dot(query, candidate_vec))
            if similarity >= threshold:
                distance = float(max(0.0, 1.0 - similarity))
                bbox = json.loads(row["bounding_box"]) if row["bounding_box"] else None
                matches.append({
                    "image_key": row["image_key"],
                    "face_index": row["face_index"],
                    "similarity": round(similarity, 4),
                    "distance": round(distance, 4),
                    "bounding_box": bbox,
                })

        # Sort descending by similarity
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            img_row = conn.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN status='indexed' THEN 1 ELSE 0 END) AS indexed, "
                "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed FROM indexed_images"
            ).fetchone()
            face_row = conn.execute("SELECT COUNT(*) AS total FROM face_embeddings").fetchone()

            return {
                "total_images": img_row["total"] or 0,
                "indexed_images": img_row["indexed"] or 0,
                "failed_images": img_row["failed"] or 0,
                "total_faces": face_row["total"] or 0,
            }

    def get_indexed_image(self, image_key: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM indexed_images WHERE image_key = ?", (image_key,)).fetchone()
            if not row:
                return None
            return dict(row)


# Global index store instance
index_store = SQLiteIndexStore()
