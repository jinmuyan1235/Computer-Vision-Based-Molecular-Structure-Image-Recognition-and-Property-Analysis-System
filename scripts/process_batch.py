"""Run batch OCSR outside the Streamlit server process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.analysis.batch_analyzer import BatchAnalyzer
from src.export.json_exporter import to_json_text
from src.utils.file_utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Folder containing PNG/JPG/JPEG images.")
    parser.add_argument("--backend", default=config.OCSR_BACKEND, help="OCSR backend: demo, molscribe, decimer, ensemble.")
    parser.add_argument("--output", default=str(config.OUTPUT_DIR / "batch_runs"), help="Output directory for batch runs.")
    parser.add_argument("--molscribe-device", default=None, help="Runtime MolScribe device override.")
    parser.add_argument("--decimer-device", default=None, help="Runtime DECIMER device override.")
    parser.add_argument("--visible-gpu-index", default=None, help="CUDA_VISIBLE_DEVICES index for TensorFlow/DECIMER.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        runtime_config = {
            "molscribe_device": args.molscribe_device,
            "decimer_device": args.decimer_device,
            "visible_gpu_index": args.visible_gpu_index,
        }
        output_dir = ensure_directory(args.output)
        result = BatchAnalyzer(args.backend, output_dir, runtime_config=runtime_config).analyze_folder(args.input)
        ui_result = {
            "summary": result["summary"],
            "rows": result["rows"],
            "reports": result["reports"],
            "exports": result["exports"],
        }
        result_path = output_dir / "batch_ui_result.json"
        result_path.write_text(to_json_text(ui_result), encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "failed", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": "success",
        "summary": ui_result["summary"],
        "exports": ui_result["exports"],
        "result_path": str(result_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
