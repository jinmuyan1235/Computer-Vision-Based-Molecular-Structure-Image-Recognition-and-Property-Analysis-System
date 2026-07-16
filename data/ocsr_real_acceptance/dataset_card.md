# OCSR Real Acceptance Starter v1

## What is included

This starter contains **14 image rows from 2 independent source images**:

- **1 genuine patent/document crop** from the public OCMR repository. It contains `Formula II`, atom numbering, and surrounding document content. The molecule is benzene-1,2-dicarbonitrile (`N#Cc1ccccc1C#N`).
- **1 independent official MolScribe example image** with the ground-truth SMILES published in the MolScribe README. Its upstream provenance is not stated, so it is classified as an external official example rather than a real literature crop.
- Derived low-resolution, JPEG, rotated, thresholded, and scan-like variants.
- Two real-document negative controls: text-only content and an incomplete molecular crop.

## Important limitation

This package fixes the repository's “zero real sample” problem, but it is **not a statistically meaningful benchmark**. It contains only one independently sourced real-document molecule. Perturbations of the same source must remain in the same evaluation group and must not be counted as independent evidence.

Use `scripts/build_full_real_subset.py` to download a larger deterministic subset from CLEF, JPO, UOB, and USPTO benchmark archives.

## Verification policy

Rows in this starter are marked `verified` because:

- the OCMR patent image was visually checked against the SMILES printed by the upstream OCMR example;
- the MolScribe image uses the ground-truth SMILES shown by the upstream MolScribe README;
- negative controls were manually cropped and assigned `expected_action=reject`.

For a release benchmark, manually review newly downloaded samples before changing their status from `pending` to `verified`.

## Leakage policy

All variants sharing a `source_document` belong to the same source group. Do not place variants of the same image in different train/dev/test splits.

## Licensing

- MolScribe repository: MIT license.
- OCMR `test.png`: publicly accessible GitHub source, but no explicit repository license was found during packaging. Treat this item as local research/evaluation material and verify reuse terms before republishing it in a public release.
- The full benchmark downloader preserves upstream source and license notes; users are responsible for complying with CLEF/UOB/JPO/USPTO terms.
