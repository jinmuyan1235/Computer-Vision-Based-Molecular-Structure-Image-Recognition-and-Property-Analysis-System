"""SQLite database bootstrap for local analysis history."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import config


SCHEMA_VERSION = 1


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the local application database and ensure the schema exists."""
    path = Path(db_path or config.APP_DB_PATH).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    initialize(connection)
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    """Create all history tables and indexes idempotently."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            input_type TEXT,
            filename TEXT,
            input_path TEXT,
            image_sha256 TEXT,
            backend TEXT,
            decision TEXT,
            status TEXT,
            final_smiles TEXT,
            inchikey TEXT,
            report_path TEXT,
            is_favorite INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS corrections (
            correction_id TEXT PRIMARY KEY,
            analysis_id TEXT NOT NULL,
            previous_smiles TEXT,
            new_smiles TEXT,
            source TEXT,
            created_at TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT,
            status TEXT,
            progress TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            result_path TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analyses_filename ON analyses(filename);
        CREATE INDEX IF NOT EXISTS idx_analyses_final_smiles ON analyses(final_smiles);
        CREATE INDEX IF NOT EXISTS idx_analyses_inchikey ON analyses(inchikey);
        CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
        CREATE INDEX IF NOT EXISTS idx_analyses_decision ON analyses(decision);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        PRAGMA user_version = 1;
        """
    )
    connection.commit()
