# OCSR Benchmark

This folder contains the manifest format and a small demo manifest for exercising the benchmark framework. The bundled `example_manifest.csv` references local demo images already present in `data/samples`; it is not a scientific benchmark and must not be reported as real OCSR accuracy.

## Manifest Format

Required CSV columns:

```csv
sample_id,image_path,ground_truth_smiles,category,source,notes
aspirin_001,images/aspirin_001.png,CC(=O)Oc1ccccc1C(=O)O,clean_generated,local_demo,
```

Recommended columns for acceptance runs:

```csv
split,scaffold_key,source_document,image_quality,complexity,perturbation,structure_features
test,benzene_carboxylate,patent_us_xxx,scanned,medium,low_resolution,ester;acid;aromatic
```

Rules enforced by the loader:

- `sample_id`, `image_path`, `ground_truth_smiles`, `category`, and `source` must be non-empty.
- `sample_id` values must be unique.
- `image_path` must exist and remain inside `--dataset-root`.
- `ground_truth_smiles` must be parseable by RDKit.
- Invalid rows are reported explicitly; they are not silently skipped.

Recommended fields are optional for compatibility, but real acceptance reports should include them. The evaluator stratifies metrics by `source`, `image_quality`, `complexity`, `perturbation`, `structure_features`, `split`, backend and preprocessing strategy.

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

## Build a Local Acceptance Pack

Generate a deterministic local acceptance dataset with clean RDKit renders, common image perturbations and reject/distractor controls:

```bash
python scripts/build_ocsr_acceptance_set.py
python scripts/evaluate_ocsr.py \
  --manifest data/ocsr_acceptance/manifest.csv \
  --dataset-root data/ocsr_acceptance \
  --backend molscribe \
  --output data/outputs/benchmark
```

To ingest manually cropped and labeled real-world images, fill `benchmark/manual_labels_template.csv` and run:

```bash
python scripts/ingest_ocsr_labeled_images.py \
  --labels data/raw_ocsr_real/labels.csv \
  --image-root data/raw_ocsr_real \
  --output-root data/ocsr_manual_labeled
```

See `docs/ocsr_dataset_curation.md` for the full labeling guide.

## Real Datasets

To benchmark a real backend:

1. Prepare images from a source you are allowed to use.
2. Create a manifest with verified ground-truth SMILES.
3. Split by molecule identity, scaffold, document/source family, or vendor batch rather than random image rows.
4. Install and configure the selected backend, for example MolScribe with `MOLSCRIBE_MODEL_PATH`.
5. Run the CLI with the backend and preprocessing strategy you want to compare.
6. Compare reports by `backend`, `preprocessing_strategy`, `source`, `image_quality`, `complexity` and `perturbation`.

Minimum acceptance coverage should include clean RDKit/CDK drawings, scanned literature figures, screenshots or compressed images, low-resolution/rotated/blurred/noisy/non-white backgrounds, hand-drawn structures, stereochemistry and E/Z bonds, isotopes and charges, salts/fragments/metal complexes, long chains, macrocycles and fused rings, R-groups/abbreviations/polymers, pages containing multiple structures, and reaction/text/table distractors. Use the `category`, `image_quality`, `complexity`, `perturbation` and `structure_features` fields to label these cases explicitly.

The core report separates valid-SMILES rate from exact-match rate, and adds stereochemistry exact rate, atom-count error rate, formal-charge error rate, bond-type profile error rate, confidence calibration error and rejection coverage for distractor-like rows.

Do not commit model weights, proprietary datasets, or benchmark results that cannot be reproduced from documented inputs.
