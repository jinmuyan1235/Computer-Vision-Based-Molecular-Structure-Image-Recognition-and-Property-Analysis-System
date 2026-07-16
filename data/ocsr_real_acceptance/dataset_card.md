# Real OCSR Acceptance Dataset Card

This directory is the fixed, reviewed acceptance set used for release gates.

Images in `images/` may be private or license-restricted and are ignored by Git by default. Commit only files that are legal to share. If images cannot be public, keep them locally and commit only this dataset card, the manifest schema, and reproducible release reports that do not expose restricted image content.

## Required Manifest Fields

- `dataset_version`
- `image_sha256`
- `source_document`
- `source_license`
- `annotator`
- `reviewer`
- `review_status`
- `ground_truth_smiles`
- `ground_truth_inchikey`
- `expected_action`
- `supported_scope`

Release acceptance requires `review_status=verified`, matching image SHA-256, and matching `ground_truth_inchikey` for recognisable molecule rows. Reject rows may leave `ground_truth_smiles` and `ground_truth_inchikey` empty.

## Split Policy

This acceptance set is release-only. Do not use it for training, prompt tuning, threshold tuning, or repeated manual model selection. Keep `split=test` unless a row is intentionally excluded from release gates.

Avoid leakage by separating rows across:

- molecule identity and InChIKey;
- scaffold family;
- source document or patent family;
- perturbation variants of the same image;
- highly similar structure families.

## First-Phase Coverage Target

Aim for roughly 300 reviewed rows:

- 100-120 clear single-molecule images;
- 70-80 real paper or patent crops;
- 40-50 scanned, low-resolution, compressed, or rotated images;
- 25-35 handwritten, charged, salt, or multi-fragment molecules;
- 25-35 non-molecule and reaction distractors.

These are project-phase targets, not general industry claims.
