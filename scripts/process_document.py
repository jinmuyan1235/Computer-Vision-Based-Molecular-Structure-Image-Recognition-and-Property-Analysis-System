"""Process a PDF, page image, or ZIP image collection through document OCSR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.documents.input_loader import DocumentInputError, OptionalDependencyError
from src.documents.processor import DocumentOCSRProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PDF, PNG/JPG/JPEG page image, or ZIP of page images.")
    parser.add_argument("--backend", default=config.OCSR_BACKEND, help="OCSR backend: demo, molscribe, decimer, ensemble.")
    parser.add_argument("--output", default=str(config.DOCUMENT_OUTPUT_DIR), help="Output directory for document runs.")
    parser.add_argument("--detect-only", action="store_true", help="Detect and crop regions without running OCSR.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        processor = DocumentOCSRProcessor(backend=args.backend, output_dir=args.output)
        result = processor.process(args.input, run_ocsr=not args.detect_only)
    except OptionalDependencyError as exc:
        print(json.dumps({"status": "unavailable", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except (DocumentInputError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "failed", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": "success",
        "document_id": result["document_id"],
        "summary": result["summary"],
        "exports": result["exports"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
