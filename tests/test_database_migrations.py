from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.storage import database


def test_database_initializes_schema_version_with_migrations(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    connection = database.connect(db_path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert version == database.SCHEMA_VERSION
    assert {"analyses", "corrections", "jobs"}.issubset(tables)


def test_database_migrates_v1_history_delete_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    try:
        database.MIGRATIONS[1](connection)
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    upgraded = database.connect(db_path)
    try:
        version = upgraded.execute("PRAGMA user_version").fetchone()[0]
        columns = {row[1] for row in upgraded.execute("PRAGMA table_info(analyses)").fetchall()}
    finally:
        upgraded.close()

    assert version == database.SCHEMA_VERSION
    assert {"delete_status", "delete_errors", "delete_requested_at", "delete_updated_at"}.issubset(columns)


def test_database_refuses_newer_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "future.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(f"PRAGMA user_version = {database.SCHEMA_VERSION + 1}")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="schema"):
        database.connect(db_path)
