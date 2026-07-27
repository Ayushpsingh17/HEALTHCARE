# Breast Cancer Diagnosis Support System

An end-to-end AI application that predicts malignant vs. benign classification from
digitized fine-needle aspirate (FNA) cell-nuclei measurements, built as a Major Project
for the **Healthcare & Digital Health** track.

> **Disclaimer:** This is a decision-support and educational tool only. It does not
> diagnose disease and must never replace evaluation by a qualified pathologist or
> clinician.

---

## 1. Problem Statement

Breast cancer screening via fine-needle aspiration biopsy requires a pathologist to
visually assess cell nuclei characteristics (size, shape, texture, concavity, etc.) from
a digitized image and classify the mass as malignant or benign. This process is:

- **Time-consuming** at scale, in high-volume or resource-constrained labs.
- **Subject to inter-observer variability** — different pathologists can reach different
  conclusions on borderline cases.
- **Difficult to triage** — every case gets equal turnaround time regardless of how
  clear-cut it is.

**Why AI is an appropriate tool here:** the diagnostic decision is ultimately a pattern
match across ~30 quantitative measurements already computed from the image — a
supervised classification problem with a well-defined ground truth (later confirmed by
biopsy/surgery). A statistical model can learn this mapping from historical, confirmed
cases and provide a fast, consistent, second opinion that flags high-risk cases for
priority review. It does not replace the pathologist's judgment, particularly on
borderline or atypical presentations.

**Existing approaches and their limits:** most public tutorials solve this dataset as a
pure classification exercise in a notebook and stop there. That doesn't reflect how such
a tool would actually be used — a clinician needs an interface to enter values and get a
result, not a Jupyter cell. This project closes that gap with a served model behind a
real API and a usable frontend.

---

## 2. Dataset

- **Source:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset, originally from the UCI
  Machine Learning Repository (Dr. William H. Wolberg, University of Wisconsin
  Hospitals). Accessed via `sklearn.datasets.load_breast_cancer` — no external download
  required, which also avoids any data-provenance or licensing ambiguity.
- **Size:** 569 samples, 30 numeric features, binary label (malignant / benign).
- **Features:** 10 base measurements of cell nuclei (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each
  reported as a mean, standard error, and "worst" (largest) value — hence 30 columns.
- **Class balance:** 357 benign, 212 malignant (moderately imbalanced; addressed via
  stratified splitting and by monitoring recall on the malignant class specifically,
  not just overall accuracy).

A reference copy is exported to `model/wdbc_dataset_reference.csv` for inspection.

---

## 3. Engineering Justification

| Decision | Choice | Why |
|---|---|---|
| **Dataset** | WDBC via scikit-learn | Real, peer-reviewed clinical data; bundled with the library (reproducible, no download/licensing risk) |
| **Preprocessing** | `StandardScaler` on all 30 features | Feature scales vary by orders of magnitude (e.g. area vs. smoothness); scaling is required for SVM, MLP, and regularized Logistic Regression to converge properly and weight features fairly |
| **Split** | Stratified 80/20 train/test | Preserves the malignant/benign ratio in both sets given class imbalance |
| **Model selection** | Compared 5 families: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting, MLP | Avoids picking a model arbitrarily; demonstrates the trade-offs across linear, kernel, ensemble, and neural approaches |
| **Selection metric** | ROC-AUC (primary), malignant-class recall (secondary check) | ROC-AUC is threshold-independent and balances both classes; recall on malignant cases is checked separately because a false negative (missed cancer) is the costliest error type in this domain |
| **Final model** | **Logistic Regression** | Tied for best ROC-AUC (0.9954) with SVM, but chosen over SVM for interpretability (inspectable coefficients — see Model Performance tab) and ~2x faster inference, both valuable in a clinical tool |
| **Backend** | FastAPI | Async-capable, automatic request validation via Pydantic, and free interactive OpenAPI docs at `/docs` |
| **Frontend** | Static HTML/CSS/JS | No build step, fully portable, easy for a reviewer to open directly; calls the backend over a documented REST API |
| **Evaluation metrics** | Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold CV ROC-AUC | Accuracy alone is insufficient under class imbalance; the full set gives a rounded view and the CV score checks that performance isn't a lucky split |

---

## 4. Results (Held-Out Test Set, 114 samples)

| Metric | Value |
|---|---|
| Accuracy | 98.25% |
| Precision | 98.61% |
| Recall (overall) | 98.61% |
| F1 Score | 0.9861 |
| ROC-AUC | 0.9954 |
| 5-fold CV ROC-AUC | 0.9957 ± 0.0048 |
| **Recall on malignant class specifically** | **97.62%** (41/42 correctly caught) |

Full comparison across all 5 candidate models is in
`model/model_comparison_results.csv`, and supporting figures are in `model/figures/`.

---

## 5. Project Structure

```
breast-cancer-diagnosis-ai/
├── README.md                        <- you are here
├── requirements.txt
├── .gitignore
├── model/
│   ├── train_model.py               <- data loading, preprocessing, model comparison, training
│   ├── wdbc_dataset_reference.csv   <- exported dataset for inspection
│   ├── model_comparison_results.csv <- all 5 candidate models' metrics
│   ├── top_feature_importances.csv
│   ├── figures/                     <- charts used in the technical report & frontend
│   └── saved_models/
│       ├── best_model.joblib        <- trained Logistic Regression model
│       ├── scaler.joblib            <- fitted StandardScaler
│       └── model_metadata.json      <- model name, metrics, feature order
├── backend/
│   ├── main.py                      <- FastAPI app (health, model-info, predict endpoints)
│   ├── schemas.py                   <- Pydantic request/response models
│   └── requirements.txt
├── frontend/
│   └── index.html                   <- single-file clinical UI (Assessment + Model Performance tabs)
└── docs/
    ├── API_DOCUMENTATION.md
    ├── Technical_Report.docx
    └── Presentation.pptx
```

---

## 6. Running It Locally

### Prerequisites
Python 3.10+, pip.

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Retrain the model
A trained model is already included under `model/saved_models/`. To retrain from
scratch and regenerate the comparison figures:
```bash
cd model
python train_model.py
```

### Step 3 — Start the backend API
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Verify it's running: open `http://localhost:8000/health` — you should see
`{"status":"ok","model_loaded":true}`. Interactive API docs (Swagger UI) are at
`http://localhost:8000/docs`.

### Step 4 — Open the frontend
Open `frontend/index.html` directly in a browser (double-click it, or
`open frontend/index.html` / `start frontend/index.html`). Use **Load sample** to try a
pre-filled example, or enter your own 30 values, then click **Run Assessment**.

> The frontend expects the backend at `http://localhost:8000`. If you run the backend on
> a different host/port, update the `API_BASE` constant near the top of the `<script>`
> block in `frontend/index.html`.

---

## 7. Academic Integrity Note

This project uses a well-known public dataset (WDBC, bundled with scikit-learn) rather
than a scraped or ambiguously-licensed source. All model comparison numbers in this
README and the technical report were produced by actually running `train_model.py` in
this repository — they are not estimated or copied from another source. No external
GitHub repository or Kaggle notebook was copied; the backend, frontend, and training
pipeline were written for this project.

## 8. References

- Wolberg, W.H., Street, W.N., Mangasarian, O.L. — *Wisconsin Diagnostic Breast Cancer
  (WDBC)* dataset, UCI Machine Learning Repository.
- scikit-learn documentation: https://scikit-learn.org
- FastAPI documentation: https://fastapi.tiangolo.com
