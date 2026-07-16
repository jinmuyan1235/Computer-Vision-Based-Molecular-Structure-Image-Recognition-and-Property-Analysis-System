"""Runtime registry for resumable background jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping
from uuid import uuid4

import config
from src.runtime.batch_job_store import BatchJobStore
from src.runtime.job_manager import (
    extract_json_object,
    is_process_alive,
    start_logged_process,
    terminate_process_tree_by_pid,
)
from src.utils.file_utils import ensure_directory, iter_image_files, safe_stem


def start_batch_job(
    input_dir: str | Path,
    backend: str,
    runtime_config: Mapping[str, Any] | None = None,
    *,
    store: BatchJobStore | None = None,
    source: str = "folder",
    parent_job_id: str | None = None,
    retry_mode: str | None = None,
) -> dict[str, Any]:
    """Start a background batch job and return its persisted state."""
    active_store = store or BatchJobStore()
    job_id = _new_job_id()
    return _start_prepared_batch_job(
        job_id,
        Path(input_dir).expanduser().resolve(),
        backend,
        runtime_config or {},
        active_store,
        source=source,
        parent_job_id=parent_job_id,
        retry_mode=retry_mode,
    )


def start_batch_job_from_uploads(
    uploads: Iterable[tuple[str, bytes]],
    backend: str,
    runtime_config: Mapping[str, Any] | None = None,
    *,
    store: BatchJobStore | None = None,
) -> dict[str, Any]:
    """Persist uploaded files and start a background batch job."""
    active_store = store or BatchJobStore()
    job_id = _new_job_id()
    input_dir = ensure_directory(active_store.job_dir(job_id) / "input")
    for index, (name, content) in enumerate(uploads, start=1):
        suffix = Path(name).suffix.lower()
        stem = safe_stem(Path(name).stem, f"upload_{index:03d}")
        destination = input_dir / f"{index:03d}_{stem}{suffix}"
        destination.write_bytes(content)
    return _start_prepared_batch_job(job_id, input_dir, backend, runtime_config or {}, active_store, source="upload")


def start_batch_retry_job(
    result: Mapping[str, Any],
    backend: str,
    mode: str,
    runtime_config: Mapping[str, Any] | None = None,
    *,
    store: BatchJobStore | None = None,
    parent_job_id: str | None = None,
) -> dict[str, Any]:
    """Start a retry job from failed or review-needed reports in an existing result."""
    active_store = store or BatchJobStore()
    selected = _retry_source_paths(result, mode)
    if not selected:
        raise ValueError("没有可重试的图片。")
    job_id = _new_job_id()
    input_dir = ensure_directory(active_store.job_dir(job_id) / "input")
    for index, source in enumerate(selected, start=1):
        suffix = source.suffix.lower()
        destination = input_dir / f"{index:03d}_{safe_stem(source.stem, 'retry')}{suffix}"
        shutil.copy2(source, destination)
    return _start_prepared_batch_job(
        job_id,
        input_dir,
        backend,
        runtime_config or {},
        active_store,
        source="retry",
        parent_job_id=parent_job_id,
        retry_mode=mode,
    )


def refresh_batch_job(job_id: str, store: BatchJobStore | None = None) -> dict[str, Any]:
    """Refresh status for a job that may have outlived the current UI session."""
    active_store = store or BatchJobStore()
    state = active_store.read(job_id)
    status = state.get("status")
    if status not in {"queued", "running", "cancelling"}:
        return state
    pid = _int_or_none(state.get("pid"))
    if is_process_alive(pid):
        return state

    payload = _payload_from_logs(active_store.stdout_path(job_id))
    if payload and payload.get("result_path") and Path(str(payload["result_path"])).is_file():
        result = active_store.load_result(job_id)
        if result:
            summary = result.get("summary") or {}
            if payload.get("status") == "cancelled" or summary.get("cancelled"):
                return active_store.mark_cancelled(job_id, "任务已取消。")
            return active_store.complete(job_id, payload["result_path"], result.get("exports") or {}, summary)
    if status == "cancelling" or active_store.cancel_requested(job_id):
        return active_store.mark_cancelled(job_id, "任务已取消。")
    stderr = active_store.stderr_path(job_id).read_text(encoding="utf-8") if active_store.stderr_path(job_id).is_file() else ""
    message = (stderr.strip().splitlines()[-1] if stderr.strip() else "后台任务已退出但未写出结果。")
    return active_store.fail(job_id, message)


def cancel_batch_job(job_id: str, store: BatchJobStore | None = None, force: bool = True) -> dict[str, Any]:
    """Request cancellation and optionally terminate the process immediately."""
    active_store = store or BatchJobStore()
    state = active_store.request_cancel(job_id)
    pid = _int_or_none(state.get("pid"))
    if force and is_process_alive(pid):
        terminate_process_tree_by_pid(pid)
        state = active_store.mark_cancelled(job_id, "任务已取消。")
    return state


def request_skip_current(job_id: str, store: BatchJobStore | None = None) -> dict[str, Any]:
    """Ask the worker to skip the next file boundary."""
    return (store or BatchJobStore()).request_skip_current(job_id)


def clear_batch_job(job_id: str, store: BatchJobStore | None = None) -> None:
    """Remove a non-running job from the registry."""
    active_store = store or BatchJobStore()
    state = active_store.read(job_id)
    if state.get("status") in {"queued", "running", "cancelling"}:
        cancel_batch_job(job_id, active_store, force=True)
    active_store.clear(job_id)


def load_batch_job_result(job_id: str, store: BatchJobStore | None = None) -> dict[str, Any] | None:
    """Load the completed UI result for a job."""
    return (store or BatchJobStore()).load_result(job_id)


def _start_prepared_batch_job(
    job_id: str,
    input_dir: Path,
    backend: str,
    runtime_config: Mapping[str, Any],
    store: BatchJobStore,
    *,
    source: str,
    parent_job_id: str | None = None,
    retry_mode: str | None = None,
) -> dict[str, Any]:
    output_dir = ensure_directory(store.job_dir(job_id) / "outputs")
    total = len(list(iter_image_files(input_dir)))
    command = _batch_command(job_id, input_dir, output_dir, backend, runtime_config, store)
    store.create(
        job_id,
        backend=backend,
        input_dir=input_dir,
        output_dir=output_dir,
        total=total,
        source=source,
        command=command,
        runtime_config=dict(runtime_config),
        parent_job_id=parent_job_id,
        retry_mode=retry_mode,
    )
    process = start_logged_process(
        command,
        cwd=config.PROJECT_ROOT,
        env=_job_environment(),
        stdout_path=store.stdout_path(job_id),
        stderr_path=store.stderr_path(job_id),
    )
    return store.mark_running(job_id, process.pid, command)


def _batch_command(
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    backend: str,
    runtime_config: Mapping[str, Any],
    store: BatchJobStore,
) -> list[str]:
    command = [
        sys.executable,
        str(config.PROJECT_ROOT / "scripts" / "process_batch.py"),
        "--input",
        str(input_dir),
        "--backend",
        backend,
        "--output",
        str(output_dir),
        "--job-id",
        job_id,
        "--job-store-dir",
        str(store.root),
    ]
    if runtime_config.get("molscribe_device"):
        command.extend(["--molscribe-device", str(runtime_config["molscribe_device"])])
    if runtime_config.get("decimer_device"):
        command.extend(["--decimer-device", str(runtime_config["decimer_device"])])
    if runtime_config.get("visible_gpu_index") is not None:
        command.extend(["--visible-gpu-index", str(runtime_config["visible_gpu_index"])])
    return command


def _job_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MOLSCRIBE_ISOLATED_SUBPROCESS", "true")
    env.setdefault("DECIMER_ISOLATED_SUBPROCESS", "true")
    return env


def _retry_source_paths(result: Mapping[str, Any], mode: str) -> list[Path]:
    reports = list(result.get("reports") or [])
    paths: list[Path] = []
    for report in reports:
        if not isinstance(report, Mapping):
            continue
        if mode == "failed" and report.get("status") == "success":
            continue
        if mode == "review" and not _needs_review(report):
            continue
        input_data = report.get("input") if isinstance(report.get("input"), Mapping) else {}
        path = Path(str(input_data.get("path") or ""))
        if path.is_file():
            paths.append(path)
    return paths


def _needs_review(report: Mapping[str, Any]) -> bool:
    decision = report.get("recognition_decision") if isinstance(report.get("recognition_decision"), Mapping) else {}
    ocsr = report.get("ocsr") if isinstance(report.get("ocsr"), Mapping) else {}
    consensus = ocsr.get("consensus") if isinstance(ocsr.get("consensus"), Mapping) else {}
    return bool(
        decision.get("manual_review_recommended")
        or decision.get("decision") in {"review_needed", "accepted_with_warning"}
        or consensus.get("decision") == "review_needed"
        or consensus.get("status") == "disagreement"
    )


def _payload_from_logs(stdout_path: Path) -> dict[str, Any] | None:
    if not stdout_path.is_file():
        return None
    return extract_json_object(stdout_path.read_text(encoding="utf-8"))


def _new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"batch_{stamp}_{uuid4().hex[:8]}"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
