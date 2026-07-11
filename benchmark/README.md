# OCSR Benchmark

This folder contains the manifest format and a small demo manifest for exercising the benchmark framework. The bundled `example_manifest.csv` references local demo images already present in `data/samples`; it is not a scientific benchmark and must not be reported as real OCSR accuracy.

## Manifest Format

Required CSV columns:

```csv
sample_id,image_path,ground_truth_smiles,category,source,notes
aspirin_001,images/aspirin_001.png,CC(=O)Oc1ccccc1C(=O)O,clean_generated,local_demo,
```

Rules enforced by the loader:

- `sample_id`, `image_path`, `ground_truth_smiles`, `category`, and `source` must be non-empty.
- `sample_id` values must be unique.
- `image_path` must exist and remain inside `--dataset-root`.
- `ground_truth_smiles` must be parseable by RDKit.
- Invalid rows are reported explicitly; they are not silently skipped.

By default, `--dataset-root` is the project root, so paths like `data/samples/aspirin.png` work. For a separate dataset directory, pass `--dataset-root C:\path\to\dataset` and keep image paths relative to that root.

## Run

```bash
python scripts/evaluate_ocsr.py \
  --manifest benchmark/example_manifest.csv \
  --backend demo \
  --output data/outputs/benchmark
```

Windows PowerShell:

```powershell
python scripts/evaluate_ocsr.py `
  --manifest benchmark/example_manifest.csv `
  --backend demo `
  --output data/outputs/benchmark
```

Each run creates a new directory such as:

```text
data/outputs/benchmark/20260711_153000_demo/
├── config.json
├── predictions.csv
├── metrics.json
├── report.md
├── failure_cases.csv
└── charts/
```

Historical runs are never overwritten.

## Real Datasets

To benchmark a real backend:

1. Prepare images from a source you are allowed to use.
2. Create a manifest with verified ground-truth SMILES.
3. Install and configure the selected backend, for example MolScribe with `MOLSCRIBE_MODEL_PATH`.
4. Run the CLI with the backend and preprocessing strategy you want to compare.
5. Compare reports by `backend` and `preprocessing_strategy`.

Do not commit model weights, proprietary datasets, or benchmark results that cannot be reproduced from documented inputs.
