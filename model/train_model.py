"""
train_model.py
================
Trains and compares multiple ML models on the Wisconsin Diagnostic Breast
Cancer (WDBC) dataset, then persists the best-performing model + scaler +
metadata for the backend API to serve.

Dataset: sklearn.datasets.load_breast_cancer
  - 569 samples, 30 numeric features computed from digitized images of a
    fine needle aspirate (FNA) of a breast mass.
  - Features describe characteristics of the cell nuclei present in the
    image (radius, texture, perimeter, area, smoothness, compactness,
    concavity, symmetry, fractal dimension - each as mean, standard error,
    and "worst"/largest value).
  - Target: 0 = malignant, 1 = benign (as provided by sklearn; we invert
    this in the API layer so "1 = malignant risk" reads naturally for a
    clinical UI - see backend/main.py).
  - Original source: UCI Machine Learning Repository / Dr. William H.
    Wolberg, University of Wisconsin Hospitals.

Run:
    python train_model.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
SAVED = HERE / "saved_models"
SAVED.mkdir(exist_ok=True)
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# sklearn encodes 0=malignant, 1=benign. We keep that convention internally
# and re-label at the API boundary for clinical readability.
y = pd.Series(data.target, name="benign")

print(f"Dataset shape: {X.shape}")
print(f"Class balance -> benign: {(y==1).sum()}, malignant: {(y==0).sum()}")

# Save a reference copy of the dataset (for the report / reproducibility)
df_full = X.copy()
df_full["diagnosis"] = np.where(y == 0, "malignant", "benign")
df_full.to_csv(HERE / "wdbc_dataset_reference.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified to preserve class balance)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Preprocessing: standardize features (important for SVM, MLP, LR)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Model comparison
#    We deliberately compare a spread of model families to justify the
#    final choice rather than picking one arbitrarily:
#      - Logistic Regression: interpretable linear baseline
#      - SVM (RBF): strong on small, high-dimensional tabular data
#      - Random Forest: robust, handles nonlinearity, gives feature importance
#      - Gradient Boosting: often top performer on tabular data
#      - MLP (small neural net): deep-learning representative
# ---------------------------------------------------------------------------
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "MLP (Neural Network)": MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=2000, random_state=RANDOM_STATE
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []

for name, model in candidates.items():
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        "cv_roc_auc_std": round(cv_scores.std(), 4),
        "train_time_sec": round(train_time, 4),
    }
    results.append(metrics)
    print(f"\n{name}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v}")

results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
results_df.to_csv(HERE / "model_comparison_results.csv", index=False)
print("\n=== Model comparison (sorted by ROC-AUC) ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Select best model
#    In a clinical screening context, RECALL on malignant cases matters a
#    great deal (a false negative = a missed cancer diagnosis). We therefore
#    use ROC-AUC as primary ranking (threshold-independent, balances both
#    classes) but explicitly report recall-on-malignant as a secondary check
#    before finalizing.
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["model"]
best_model = candidates[best_name]
print(f"\nSelected model: {best_name}")

# Recall specifically on the malignant class (label 0)
y_pred_best = best_model.predict(X_test_scaled)
malignant_recall = recall_score(y_test, y_pred_best, pos_label=0)
print(f"Recall on malignant class specifically: {malignant_recall:.4f}")

report = classification_report(
    y_test, y_pred_best, target_names=["malignant", "benign"]
)
print("\nClassification report:\n", report)

# ---------------------------------------------------------------------------
# 6. Figures for the technical report
# ---------------------------------------------------------------------------
# 6a. Model comparison bar chart
plt.figure(figsize=(8, 5))
plt.barh(results_df["model"], results_df["roc_auc"], color="#2E5E8C")
plt.xlabel("ROC-AUC (test set)")
plt.title("Model Comparison - ROC-AUC on Held-Out Test Set")
plt.xlim(0.9, 1.0)
plt.tight_layout()
plt.savefig(FIGS / "model_comparison.png", dpi=150)
plt.close()

# 6b. Confusion matrix for best model
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
labels = ["Malignant", "Benign"]
plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(FIGS / "confusion_matrix.png", dpi=150)
plt.close()

# 6c. ROC curve for best model
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, color="#2E5E8C", linewidth=2,
         label=f"{best_name} (AUC = {roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:,1]):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIGS / "roc_curve.png", dpi=150)
plt.close()

# 6d. Feature importance (if available) - helps explain model to clinicians
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    plt.barh(top10.index[::-1], top10.values[::-1], color="#2E5E8C")
    plt.title(f"Top 10 Feature Importances - {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGS / "feature_importance.png", dpi=150)
    plt.close()
    top10.to_csv(HERE / "top_feature_importances.csv")

# ---------------------------------------------------------------------------
# 7. Persist model, scaler, and metadata for the backend
# ---------------------------------------------------------------------------
joblib.dump(best_model, SAVED / "best_model.joblib")
joblib.dump(scaler, SAVED / "scaler.joblib")

metadata = {
    "model_name": best_name,
    "feature_order": list(X.columns),
    "class_mapping": {"0": "malignant", "1": "benign"},
    "test_metrics": results_df.iloc[0].to_dict(),
    "malignant_class_recall": round(malignant_recall, 4),
    "n_train_samples": int(len(X_train)),
    "n_test_samples": int(len(X_test)),
    "sklearn_dataset": "load_breast_cancer (UCI WDBC)",
    "random_state": RANDOM_STATE,
}
with open(SAVED / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model, scaler, and metadata to", SAVED)
print("Saved comparison figures to", FIGS)
