# API Documentation — Breast Cancer Diagnosis Support System

Base URL (local development): `http://localhost:8000`

Interactive, auto-generated OpenAPI docs are also available at `/docs` (Swagger UI) and
`/redoc` (ReDoc) whenever the server is running.

---

## Authentication

None. This is a local/academic deployment with no authentication layer. **Do not deploy
this as-is to a public endpoint without adding authentication, rate limiting, and TLS.**

---

## Endpoints

### `GET /health`
Basic liveness/readiness check.

**Response `200`**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `GET /model-info`
Returns metadata about the currently loaded model, including its held-out test-set
performance and the exact feature order it expects.

**Response `200`**
```json
{
  "model_name": "Logistic Regression",
  "n_train_samples": 455,
  "n_test_samples": 114,
  "test_metrics": {
    "model": "Logistic Regression",
    "accuracy": 0.9825,
    "precision": 0.9861,
    "recall": 0.9861,
    "f1_score": 0.9861,
    "roc_auc": 0.9954,
    "cv_roc_auc_mean": 0.9957,
    "cv_roc_auc_std": 0.0048,
    "train_time_sec": 0.0108
  },
  "malignant_class_recall": 0.9762,
  "feature_order": ["mean radius", "mean texture", "..."]
}
```

---

### `POST /predict`
Runs a diagnosis prediction from 30 cell-nuclei measurements.

**Request body** (`application/json`) — all 30 fields are required, all numeric:

| Field | Type | Constraint |
|---|---|---|
| `mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_smoothness`, `mean_symmetry`, `mean_fractal_dimension` | float | > 0 |
| `mean_compactness`, `mean_concavity`, `mean_concave_points` | float | ≥ 0 |
| `radius_error`, `texture_error`, `perimeter_error`, `area_error`, `smoothness_error`, `symmetry_error`, `fractal_dimension_error` | float | > 0 |
| `compactness_error`, `concavity_error`, `concave_points_error` | float | ≥ 0 |
| `worst_radius`, `worst_texture`, `worst_perimeter`, `worst_area`, `worst_smoothness`, `worst_symmetry`, `worst_fractal_dimension` | float | > 0 |
| `worst_compactness`, `worst_concavity`, `worst_concave_points` | float | ≥ 0 |

**Example request**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 12.3, "mean_texture": 15.2, "mean_perimeter": 78.5, "mean_area": 464.1,
    "mean_smoothness": 0.089, "mean_compactness": 0.078, "mean_concavity": 0.041,
    "mean_concave_points": 0.026, "mean_symmetry": 0.171, "mean_fractal_dimension": 0.059,
    "radius_error": 0.28, "texture_error": 0.95, "perimeter_error": 1.98, "area_error": 21.3,
    "smoothness_error": 0.006, "compactness_error": 0.018, "concavity_error": 0.021,
    "concave_points_error": 0.009, "symmetry_error": 0.017, "fractal_dimension_error": 0.0028,
    "worst_radius": 13.5, "worst_texture": 19.8, "worst_perimeter": 87.2, "worst_area": 561.3,
    "worst_smoothness": 0.121, "worst_compactness": 0.152, "worst_concavity": 0.113,
    "worst_concave_points": 0.061, "worst_symmetry": 0.251, "worst_fractal_dimension": 0.0742
  }'
```

**Response `200`**
```json
{
  "diagnosis": "benign",
  "malignant_probability": 0.0004,
  "benign_probability": 0.9996,
  "risk_level": "Low",
  "confidence": 0.9996,
  "disclaimer": "This tool provides decision support only and does not replace professional medical diagnosis. All results must be reviewed by a qualified clinician."
}
```

**Response `422` — validation error** (e.g. missing or negative field)
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "mean_radius"],
      "msg": "Input should be greater than 0",
      "input": -5
    }
  ]
}
```

**Response `503`** — model not yet loaded (only possible in the brief startup window).

---

## Risk Level Bucketing

`risk_level` is derived from `malignant_probability`:

| Malignant probability | Risk level |
|---|---|
| < 0.20 | Low |
| 0.20 – 0.60 | Moderate |
| > 0.60 | High |

This threshold scheme is a UX/triage convenience layered on top of the raw probability —
the raw `malignant_probability` should be used for any downstream clinical decisioning,
not the bucket label alone.

---

## Tested Example Outcomes

These two examples were run against the live server during development and are also
available as "Load sample" options in the frontend:

| Sample | Predicted diagnosis | Malignant probability | Risk level |
|---|---|---|---|
| Reference sample A (small, regular measurements) | benign | 0.04% | Low |
| Reference sample B (large, irregular measurements) | malignant | ~100% | High |

## CORS

The API allows all origins (`allow_origins=["*"]`) so the static frontend can call it
from `file://` or any local port during development. **Restrict this before any
production deployment.**
