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
