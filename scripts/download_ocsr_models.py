"""Download official optional OCSR model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.gpu_manager import sha256_file


MOLSCRIBE_REPO_ID = "yujieq/MolScribe"
MOLSCRIBE_DEFAULT_FILE = "swin_base_char_aux_1m.pth"


def download_molscribe(filename: str = MOLSCRIBE_DEFAULT_FILE) -> dict[str, str | None]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("缺少 huggingface_hub，请先运行 setup_gpu_environment.sh 或 pip install huggingface_hub。") from exc
    model_dir = PROJECT_ROOT / "models" / "molscribe"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo_id=MOLSCRIBE_REPO_ID, filename=filename, local_dir=model_dir))
    digest = sha256_file(path)
    if digest:
        path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {
        "backend": "molscribe",
        "source": f"https://huggingface.co/{MOLSCRIBE_REPO_ID}",
        "filename": filename,
        "path": str(path.resolve()),
        "sha256": digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official OCSR model weights.")
    parser.add_argument("--molscribe-file", default=MOLSCRIBE_DEFAULT_FILE)
    args = parser.parse_args()
    result = {"molscribe": download_molscribe(args.molscribe_file)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
