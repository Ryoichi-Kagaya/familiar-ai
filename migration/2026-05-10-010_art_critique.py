"""Art critique long-term memory: artworks, critiques, motivation scores."""

from __future__ import annotations
import sqlite3


def upgrade(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS artworks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist_name TEXT NOT NULL,
            artist_name_normalized TEXT,
            year TEXT,
            medium TEXT,
            dimensions TEXT,
            collection TEXT,
            description TEXT,
            image_ref TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_artworks_artist_norm
            ON artworks(artist_name_normalized)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS art_critiques (
            id TEXT PRIMARY KEY,
            artwork_id TEXT NOT NULL REFERENCES artworks(id),
            critique_text TEXT,
            personal_resonance INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_art_critiques_artwork
            ON art_critiques(artwork_id)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS art_motivation_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            critique_id TEXT NOT NULL REFERENCES art_critiques(id),
            motivation_index INTEGER NOT NULL CHECK (motivation_index BETWEEN 1 AND 7),
            motivation_degree INTEGER NOT NULL CHECK (motivation_degree BETWEEN 1 AND 10),
            achievement INTEGER NOT NULL CHECK (achievement BETWEEN 1 AND 10),
            notes TEXT,
            UNIQUE (critique_id, motivation_index)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_art_scores_critique
            ON art_motivation_scores(critique_id)
    """)
