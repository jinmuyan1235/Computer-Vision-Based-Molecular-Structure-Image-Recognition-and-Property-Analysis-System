from __future__ import annotations

import sys
import time
from pathlib import Path

from src.runtime.batch_job_store import BatchJobStore
from src.runtime.job_manager import run_process
from src.runtime.job_registry import load_batch_job_result, refresh_batch_job, start_batch_job


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_batch_job_store_persists_progress_and_control_flags(tmp_path: Path) -> None:
    store = BatchJobStore(tmp_path / "jobs")
    state = store.create(
        "job1",
        backend="demo",
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        total=3,
        source="test",
    )
    assert state["status"] == "queued"

    updated = store.mark_running("job1", 1234)
    assert updated["status"] == "running"
    assert updated["pid"] == 1234

    progress = store.update_progress("job1", {
        "status": "running",
        "total": 3,
        "completed": 1,
        "current_file": "compound_001.png",
        "summary": {"total": 3, "completed": 1, "accepted": 1, "review_needed": 0, "rejected": 0, "failed": 0},
    })
    assert progress["completed"] == 1
    assert progress["current_file"] == "compound_001.png"
    assert progress["accepted"] == 1

    store.request_skip_current("job1")
    assert store.consume_skip_request("job1") is True
    assert store.consume_skip_request("job1") is False

    cancelling = store.request_cancel("job1")
    assert cancelling["status"] == "cancelling"
    assert store.cancel_requested("job1") is True


def test_process_batch_writes_background_job_result(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sample = PROJECT_ROOT / "data" / "samples" / "aspirin.png"
    (input_dir / "aspirin.png").write_bytes(sample.read_bytes())

    store = BatchJobStore(tmp_path / "jobs")
    output_dir = tmp_path / "out"
    job_id = "job_process"
    store.create(job_id, backend="demo", input_dir=input_dir, output_dir=output_dir, total=1)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "process_batch.py"),
        "--input",
        str(input_dir),
        "--backend",
        "demo",
        "--output",
        str(output_dir),
        "--job-id",
        job_id,
        "--job-store-dir",
        str(store.root),
    ]
    result = run_process(command, cwd=PROJECT_ROOT, timeout=60)

    assert result.returncode == 0, result.stderr or result.stdout
    state = store.read(job_id)
    assert state["status"] == "completed"
    assert state["completed"] == 1
    assert Path(state["result_path"]).is_file()
    payload = store.load_result(job_id)
    assert payload is not None
    assert payload["summary"]["total"] == 1
    assert payload["exports"]["csv"]


def test_job_registry_starts_and_recovers_batch_result(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sample = PROJECT_ROOT / "data" / "samples" / "aspirin.png"
    (input_dir / "aspirin.png").write_bytes(sample.read_bytes())
    store = BatchJobStore(tmp_path / "jobs")

    state = start_batch_job(input_dir, "demo", {}, store=store, source="test")
    job_id = state["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        state = refresh_batch_job(job_id, store)
        if state["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.2)

    assert state["status"] == "completed", state
    result = load_batch_job_result(job_id, store)
    assert result is not None
    assert result["summary"]["completed"] == 1
    assert Path(result["exports"]["json"]).is_file()
