"""Download official optional OCSR model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.gpu_manager import sha256_file


MOLSCRIBE_REPO_ID = "yujieq/MolScribe"
MOLSCRIBE_DEFAULT_FILE = "swin_base_char_aux_1m.pth"
DECIMER_MODEL_URLS = {
    "DECIMER": "https://zenodo.org/record/8300489/files/models.zip",
    "DECIMER_HandDrawn": "https://zenodo.org/records/10781330/files/DECIMER_HandDrawn_model.zip",
}


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


def _download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def download_decimer_models(force: bool = False) -> dict[str, object]:
    cache_dir = Path.home() / ".data" / "DECIMER-V2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"cache_dir": str(cache_dir), "models": {}}
    model_results: dict[str, object] = {}
    for model_name, url in DECIMER_MODEL_URLS.items():
        target_dir = cache_dir / f"{model_name}_model"
        zip_name = "models.zip" if model_name == "DECIMER" else f"{model_name}_model.zip"
        zip_path = cache_dir / zip_name
        if force or not (target_dir / "saved_model.pb").is_file():
            _download_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(cache_dir)
            zip_path.unlink(missing_ok=True)
        saved_model = target_dir / "saved_model.pb"
        model_results[model_name] = {
            "source": url,
            "path": str(target_dir),
            "exists": saved_model.is_file(),
            "saved_model_size": saved_model.stat().st_size if saved_model.is_file() else None,
        }
    results["models"] = model_results
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official OCSR model weights.")
    parser.add_argument("--molscribe-file", default=MOLSCRIBE_DEFAULT_FILE)
    parser.add_argument("--skip-molscribe", action="store_true")
    parser.add_argument("--skip-decimer", action="store_true")
    parser.add_argument("--force-decimer", action="store_true")
    args = parser.parse_args()
    result: dict[str, object] = {}
    if not args.skip_molscribe:
        result["molscribe"] = download_molscribe(args.molscribe_file)
    if not args.skip_decimer:
        result["decimer"] = download_decimer_models(force=args.force_decimer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
