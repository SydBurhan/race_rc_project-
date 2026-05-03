#Since model A goves an accuracy of 31,70, we use SVM to see compariosn to fulfil >2
"""
Model A — Supervised: Linear SVM Classifier
Imports the data pipeline directly from model_a_train.py for a 1:1 fair comparison.
Baseline LR Accuracy: 31.70%
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

#use model A;s data
from model_a_train import (
    load_csv, load_vectorizer, expand_to_option_rows, build_features,
    evaluate_binary, evaluate_answer_selection,
    TRAIN_CSV, VAL_CSV, MODEL_DIR, VECTORIZER_PATH
)

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/processed")   
OUTPUT_DIR  = Path("models/model_a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT    = OUTPUT_DIR / "svm_model.joblib"
LR_BASELINE  = 0.3170                 
CLASS_NAMES  = ["A", "B", "C", "D"]


print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)
train_df = load_csv(DATA_DIR / "train.csv")
val_df   = load_csv(DATA_DIR / "val.csv")
print(f"  Train rows (questions): {len(train_df):,}")
print(f"  Val   rows (questions): {len(val_df):,}")


print("\nSTEP 2 — Expanding to option rows")
train_exp = expand_to_option_rows(train_df)
val_exp   = expand_to_option_rows(val_df)
print(f"  Train option-rows: {len(train_exp):,}")
print(f"  Val   option-rows: {len(val_exp):,}")


print("\nSTEP 3 — Building TF-IDF features")
vectorizer = load_vectorizer(VECTORIZER_PATH)
X_train, y_train    = build_features(train_exp, vectorizer)
X_val,   y_val      = build_features(val_exp,   vectorizer)
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val   shape: {X_val.shape}")


print("\nSTEP 4 — Training LinearSVC (calibrated)")
svc   = LinearSVC(C=1.0, max_iter=2000, random_state=42)
model = CalibratedClassifierCV(svc, cv=3)
model.fit(X_train, y_train)
print("  Training complete.")


print("\n" + "=" * 60)
print("BINARY EVALUATION — Option-row level (is_correct label)")
print("=" * 60)

evaluate_binary(model, X_val, y_val, split_name="val")


print("\n" + "=" * 60)
print("ANSWER-SELECTION EVALUATION — Question level")
print("=" * 60)
svm_accuracy = evaluate_answer_selection(model, val_exp, X_val, split_name="val")
print(f"  SVM Answer-Selection Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT — Option-row binary labels")
print("=" * 60)
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, zero_division=0))


print("\nCONFUSION MATRIX (binary: 0 = wrong option, 1 = correct option)")
print("-" * 60)
cm = confusion_matrix(y_val, y_pred)
print(pd.DataFrame(cm,
                   index=  ["Actual 0",  "Actual 1"],
                   columns=["Pred 0",    "Pred 1"]).to_string())
print()


print("=" * 60)
print("MODEL COMPARISON TABLE — Answer-Selection Accuracy")
print("=" * 60)
delta = (svm_accuracy - LR_BASELINE) * 100
comparison = pd.DataFrame({
    "Model":         ["Logistic Regression (baseline)", "Linear SVM (calibrated)"],
    "Accuracy (%)":  [round(LR_BASELINE * 100, 2),      round(svm_accuracy * 100, 2)],
    "Δ vs Baseline": ["—",                              f"{delta:+.2f} pp"],
})
print(comparison.to_string(index=False))
print()


joblib.dump(model, MODEL_OUT)
print(f"Model saved → {MODEL_OUT}")
print("\nDone. Screenshot the comparison table and classification report for your submission.")