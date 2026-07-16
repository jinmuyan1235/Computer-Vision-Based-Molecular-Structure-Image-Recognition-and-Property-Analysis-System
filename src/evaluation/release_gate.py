"""Release acceptance gates for fixed OCSR benchmark runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.metrics import is_automatic_accept


DEFAULT_THRESHOLDS = {
    "valid_smiles_rate_min": 0.95,
    "canonical_exact_match_rate_min": 0.80,
    "false_accept_rate_max": 0.05,
    "high_risk_error_review_needed_rate_min": 1.0,
    "p95_latency_ms_max": 15000.0,
}


def _overall(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    if "metrics" in metrics_payload and isinstance(metrics_payload["metrics"], dict):
        metrics_payload = metrics_payload["metrics"]
    return dict(metrics_payload.get("overall") or metrics_payload)


def evaluate_release_gates(
    metrics_payload: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed release gates against benchmark metrics."""
    active = dict(DEFAULT_THRESHOLDS)
    active.update(thresholds or {})
    overall = _overall(metrics_payload)
    high_risk_error_count = int(overall.get("high_risk_error_count") or 0)
    high_risk_review_rate = (
        1.0 if high_risk_error_count == 0 else float(overall.get("high_risk_error_review_needed_rate") or 0.0)
    )
    checks = [
        _min_check("valid_smiles_rate", overall.get("valid_smiles_rate"), active["valid_smiles_rate_min"]),
        _min_check(
            "canonical_exact_match_rate",
            overall.get("canonical_exact_match_rate"),
            active["canonical_exact_match_rate_min"],
        ),
        _max_check("false_accept_rate", overall.get("false_accept_rate"), active["false_accept_rate_max"]),
        _min_check(
            "high_risk_error_review_needed_rate",
            high_risk_review_rate,
            active["high_risk_error_review_needed_rate_min"],
            denominator=high_risk_error_count,
        ),
        _max_check("p95_latency_ms", overall.get("p95_latency_ms"), active["p95_latency_ms_max"]),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "thresholds": active,
        "checks": checks,
    }


def collect_release_error_rows(rows: Iterable[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    """Return rows that should be inspected in a release report."""
    errors: list[dict[str, Any]] = []
    for row in rows:
        expected_action = str(row.get("expected_action") or "recognize").lower()
        negative_hallucination = expected_action == "reject" and bool(row.get("recognition_success"))
        false_accept = negative_hallucination and is_automatic_accept(row)
        mismatch = expected_action != "reject" and not bool(row.get("canonical_exact_match"))
        if row.get("failure_reason") or negative_hallucination or mismatch:
            errors.append({
                "backend": backend,
                "sample_id": row.get("sample_id"),
                "expected_action": row.get("expected_action"),
                "category": row.get("category"),
                "source": row.get("source"),
                "image_quality": row.get("image_quality"),
                "ground_truth_smiles": row.get("ground_truth_smiles"),
                "predicted_smiles": row.get("predicted_smiles"),
                "recognition_status": row.get("recognition_status"),
                "recognition_decision": row.get("recognition_decision"),
                "canonical_exact_match": row.get("canonical_exact_match"),
                "failure_reason": row.get("failure_reason")
                or (
                    "false_accept"
                    if false_accept
                    else "negative_hallucination_review_required"
                    if negative_hallucination
                    else "mismatch"
                ),
                "inference_time_ms": row.get("inference_time_ms"),
            })
    return errors


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV even when there are no rows."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = [
            "backend",
            "sample_id",
            "expected_action",
            "category",
            "source",
            "image_quality",
            "ground_truth_smiles",
            "predicted_smiles",
            "recognition_status",
            "recognition_decision",
            "canonical_exact_match",
            "failure_reason",
            "inference_time_ms",
        ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_release_report(
    path: str | Path,
    release_version: str,
    backend_payloads: dict[str, dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    """Write a Markdown release acceptance summary."""
    lines = [
        f"# OCSR Release Acceptance Report {release_version}",
        "",
        "This report is generated from a fixed, reviewed acceptance manifest. Generated demo datasets are not evidence of real-world OCSR accuracy.",
        "",
        "## Gate Summary",
        "",
        "| Backend | Gates | Valid SMILES | Canonical exact | False accept | Negative hallucination | High-risk review | P50 ms | P95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for backend, payload in backend_payloads.items():
        overall = _overall(payload["metrics"])
        gates = payload["gates"]
        lines.append(
            "| "
            + " | ".join([
                backend,
                "PASS" if gates["passed"] else "FAIL",
                _fmt(overall.get("valid_smiles_rate")),
                _fmt(overall.get("canonical_exact_match_rate")),
                _fmt(overall.get("false_accept_rate")),
                _fmt(overall.get("negative_hallucination_rate")),
                _fmt(
                    1.0
                    if int(overall.get("high_risk_error_count") or 0) == 0
                    else overall.get("high_risk_error_review_needed_rate")
                ),
                _fmt(overall.get("p50_latency_ms")),
                _fmt(overall.get("p95_latency_ms")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Gate Details",
        "",
    ])
    for backend, payload in backend_payloads.items():
        lines.extend([f"### {backend}", ""])
        lines.append("```json")
        lines.append(json.dumps(payload["gates"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines.extend([
        "## Error Rows",
        "",
        f"- Error row count: {len(errors)}",
        "- See `errors.csv` for row-level details.",
        "",
        "## Release Policy",
        "",
        "- Test/acceptance manifests must not be used for training or threshold tuning.",
        "- Private images may remain local; publish only metadata that is allowed by the source license.",
        "- Metrics are project phase targets, not industry-wide claims.",
        "",
    ])
    Path(path).expanduser().resolve().write_text("\n".join(lines), encoding="utf-8")


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _min_check(name: str, value: Any, threshold: float, denominator: int | None = None) -> dict[str, Any]:
    metric = _coerce_float(value)
    passed = metric is not None and metric >= threshold
    return {
        "metric": name,
        "operator": ">=",
        "threshold": threshold,
        "value": metric,
        "denominator": denominator,
        "passed": passed,
    }


def _max_check(name: str, value: Any, threshold: float) -> dict[str, Any]:
    metric = _coerce_float(value)
    passed = metric is not None and metric <= threshold
    return {
        "metric": name,
        "operator": "<=",
        "threshold": threshold,
        "value": metric,
        "passed": passed,
    }


def _fmt(value: Any) -> str:
    metric = _coerce_float(value)
    if metric is None:
        return "-"
    return f"{metric:.6g}"
