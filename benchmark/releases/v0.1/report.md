# OCSR Release Acceptance Report v0.1

This report is generated from a fixed, reviewed acceptance manifest. Generated demo datasets are not evidence of real-world OCSR accuracy.

## Gate Summary

| Backend | Gates | Valid SMILES | Canonical exact | False accept | Negative hallucination | High-risk review | P50 ms | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| molscribe | FAIL | 0.857143 | 0.785714 | 0 | 0.5 | 1 | 9097.11 | 9267.75 |
| decimer | FAIL | 1 | 0.428571 | 0 | 1 | 1 | 33589.7 | 33843.6 |
| ensemble | FAIL | 0.5 | 0.428571 | 0 | 0.5 | 1 | 42106.8 | 43149.7 |

## Gate Details

### molscribe

```json
{
  "passed": false,
  "thresholds": {
    "valid_smiles_rate_min": 0.95,
    "canonical_exact_match_rate_min": 0.8,
    "false_accept_rate_max": 0.05,
    "high_risk_error_review_needed_rate_min": 1.0,
    "p95_latency_ms_max": 15000.0
  },
  "checks": [
    {
      "metric": "valid_smiles_rate",
      "operator": ">=",
      "threshold": 0.95,
      "value": 0.857143,
      "denominator": null,
      "passed": false
    },
    {
      "metric": "canonical_exact_match_rate",
      "operator": ">=",
      "threshold": 0.8,
      "value": 0.785714,
      "denominator": null,
      "passed": false
    },
    {
      "metric": "false_accept_rate",
      "operator": "<=",
      "threshold": 0.05,
      "value": 0.0,
      "passed": true
    },
    {
      "metric": "high_risk_error_review_needed_rate",
      "operator": ">=",
      "threshold": 1.0,
      "value": 1.0,
      "denominator": 1,
      "passed": true
    },
    {
      "metric": "p95_latency_ms",
      "operator": "<=",
      "threshold": 15000.0,
      "value": 9267.754,
      "passed": true
    }
  ]
}
```

### decimer

```json
{
  "passed": false,
  "thresholds": {
    "valid_smiles_rate_min": 0.95,
    "canonical_exact_match_rate_min": 0.8,
    "false_accept_rate_max": 0.05,
    "high_risk_error_review_needed_rate_min": 1.0,
    "p95_latency_ms_max": 15000.0
  },
  "checks": [
    {
      "metric": "valid_smiles_rate",
      "operator": ">=",
      "threshold": 0.95,
      "value": 1.0,
      "denominator": null,
      "passed": true
    },
    {
      "metric": "canonical_exact_match_rate",
      "operator": ">=",
      "threshold": 0.8,
      "value": 0.428571,
      "denominator": null,
      "passed": false
    },
    {
      "metric": "false_accept_rate",
      "operator": "<=",
      "threshold": 0.05,
      "value": 0.0,
      "passed": true
    },
    {
      "metric": "high_risk_error_review_needed_rate",
      "operator": ">=",
      "threshold": 1.0,
      "value": 1.0,
      "denominator": 7,
      "passed": true
    },
    {
      "metric": "p95_latency_ms",
      "operator": "<=",
      "threshold": 15000.0,
      "value": 33843.62,
      "passed": false
    }
  ]
}
```

### ensemble

```json
{
  "passed": false,
  "thresholds": {
    "valid_smiles_rate_min": 0.95,
    "canonical_exact_match_rate_min": 0.8,
    "false_accept_rate_max": 0.05,
    "high_risk_error_review_needed_rate_min": 1.0,
    "p95_latency_ms_max": 15000.0
  },
  "checks": [
    {
      "metric": "valid_smiles_rate",
      "operator": ">=",
      "threshold": 0.95,
      "value": 0.5,
      "denominator": null,
      "passed": false
    },
    {
      "metric": "canonical_exact_match_rate",
      "operator": ">=",
      "threshold": 0.8,
      "value": 0.428571,
      "denominator": null,
      "passed": false
    },
    {
      "metric": "false_accept_rate",
      "operator": "<=",
      "threshold": 0.05,
      "value": 0.0,
      "passed": true
    },
    {
      "metric": "high_risk_error_review_needed_rate",
      "operator": ">=",
      "threshold": 1.0,
      "value": 1.0,
      "denominator": 6,
      "passed": true
    },
    {
      "metric": "p95_latency_ms",
      "operator": "<=",
      "threshold": 15000.0,
      "value": 43149.661,
      "passed": false
    }
  ]
}
```

## Error Rows

- Error row count: 19
- See `errors.csv` for row-level details.

## Release Policy

- Test/acceptance manifests must not be used for training or threshold tuning.
- Private images may remain local; publish only metadata that is allowed by the source license.
- Metrics are project phase targets, not industry-wide claims.
