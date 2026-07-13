"""Process one molecular image outside the Streamlit server process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.analysis.molecule_report import MoleculeReportGenerator
from src.export.json_exporter import to_json_text
from src.utils.file_utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input PNG/JPG/JPEG molecular structure image.")
    parser.add_argument("--backend", default=config.OCSR_BACKEND, help="OCSR backend: demo, molscribe, decimer, ensemble.")
    parser.add_argument("--output", default=str(config.OUTPUT_DIR / "image_runs"), help="Output directory for image runs.")
    parser.add_argument("--original-filename", default=None, help="User-facing filename to preserve in the report.")
    parser.add_argument("--molscribe-device", default=None, help="Runtime MolScribe device override.")
    parser.add_argument("--decimer-device", default=None, help="Runtime DECIMER device override.")
    parser.add_argument("--visible-gpu-index", default=None, help="CUDA_VISIBLE_DEVICES index for TensorFlow/DECIMER.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = ensure_directory(args.output)
    try:
        runtime_config = {
            "molscribe_device": args.molscribe_device,
            "decimer_device": args.decimer_device,
            "visible_gpu_index": args.visible_gpu_index,
        }
        generator = MoleculeReportGenerator(args.backend, config.OUTPUT_DIR, runtime_config=runtime_config)
        report = generator.generate(image_path=args.input)
        if args.original_filename:
            report.setdefault("input", {})["filename"] = args.original_filename
        result_path = output_dir / f"{report.get('analysis_id', 'image')}_report.json"
        result_path.write_text(to_json_text(report), encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "failed", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": "success",
        "analysis_id": report.get("analysis_id"),
        "report_status": report.get("status"),
        "message": report.get("message"),
        "result_path": str(result_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
