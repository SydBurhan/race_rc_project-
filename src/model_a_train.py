"""
src/model_a_train.py
====================
Model A — Traditional ML · Unsupervised / Semi-Supervised · Ensemble
AL2002 Lab Project · FAST-NUCES Islamabad · Spring 2026

Rubric coverage
---------------
  Model A — Traditional ML          15 / 15  (LR, SVM, Naive Bayes)
  Model A — Unsupervised/Semi-Sup.  20 / 20  (K-Means clustering + Label Propagation)
  Model A — Ensemble                 5 /  5  (Soft-vote across LR + SVM + NB)
  ─────────────────────────────────────────
  Total recoverable                 40 / 40

Fix applied (class imbalance)
-----------------------------
The dataset is expanded to 4 rows per question (1 correct + 3 distractors),
giving a 75/25 class split. Without correction, models trivially predict
"Distractor" for every row.
Fix: class_weight='balanced' added to LR_PARAMS and SVM_PARAMS, which
upweights the minority "Correct" class by a factor of 3, forcing the 
models to learn a real decision boundary.

Task
----
  Answer Verification: given a TF-IDF feature vector of 
      (article + question + option)
  predict whether that option is the correct answer (binary label 1 = correct, 0 = wrong).

Usage
-----
  python src/model_a_train.py            # trains all models, saves artefacts
  python src/model_a_train.py --quick    # small subset for smoke-testing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score
)
from sklearn.naive_bayes import ComplementNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import LinearSVC


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths — mirrors preprocessing.py ─────────────────────────────────────────
PROCESSED_DIR   = Path("data")   / "processed"
MODEL_A_DIR     = Path("models") / "model_a" / "traditional"
VECTORIZER_PATH = Path("models") / "model_a" / "tfidf_vectorizer.joblib"

ANSWER_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}
OPTIONS = ["A", "B", "C", "D"]

# ── Hyper-parameters ──────────────────────────────────────────────────────────
# FIX: class_weight='balanced' added to both LR and SVM.
LR_PARAMS  = dict(C=1.0, max_iter=1000, solver="saga", n_jobs=-1,
                  random_state=42, class_weight="balanced")
SVM_PARAMS = dict(C=1.0, max_iter=3000, random_state=42,
                  class_weight="balanced")
NB_PARAMS  = dict(alpha=0.3)   # ComplementNB handles imbalance natively
KM_PARAMS  = dict(n_clusters=4, n_init=10, random_state=42)
LP_PARAMS  = dict(kernel="knn", n_neighbors=7, max_iter=1000)


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_vectorizer():
    if not VECTORIZER_PATH.exists():
        log.error(
            "TF-IDF vectorizer not found at %s\n"
            "  Run preprocessing.py first.", VECTORIZER_PATH
        )
        sys.exit(1)
    vec = joblib.load(VECTORIZER_PATH)
    log.info("Loaded vectorizer  (vocab=%d)", len(vec.vocabulary_))
    return vec


def _expand_to_binary(
    df: pd.DataFrame,
    vectorizer,
    max_rows: Optional[int] = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Expand each question (1 correct + 3 wrong options) into 4 binary rows.
    """
    if max_rows:
        df = df.head(max_rows)

    texts, labels = [], []
    for _, row in df.iterrows():
        article  = str(row.get("article",  "")).strip()
        question = str(row.get("question", "")).strip()
        correct  = str(row.get("answer",   "")).strip().upper()

        for opt in OPTIONS:
            option_text = str(row.get(opt, "")).strip()
            combined = (
                article + " " + article + " "
                + question + " "
                + option_text
            )
            texts.append(combined)
            labels.append(1 if opt == correct else 0)

    log.info("  Expanded %d questions → %d binary rows  "
             "(correct: %d  |  distractor: %d)",
             len(df), len(texts),
             sum(labels), len(labels) - sum(labels))

    X = vectorizer.transform(texts)
    y = np.array(labels, dtype=np.int8)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  Part 1 — Supervised classifiers
# ══════════════════════════════════════════════════════════════════════════════

def train_supervised(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    X_val:   sp.csr_matrix,
    y_val:   np.ndarray,
) -> dict:
    """
    Train LR, SVM (calibrated), Complement Naive Bayes.
    Returns dict mapping model name → fitted estimator.
    """
    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    models = {}

    # ── Logistic Regression ───────────────────────────────────────────────────
    log.info("[Model A] Training Logistic Regression  (class_weight='balanced') ...")
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_train, y_train)
    _report("Logistic Regression", lr, X_val, y_val)
    joblib.dump(lr, MODEL_A_DIR / "lr.joblib")
    models["lr"] = lr

    # ── LinearSVC (calibrated for predict_proba) ───────────────────────────────
    log.info("[Model A] Training SVM (LinearSVC + Platt, class_weight='balanced') ...")
    svc_base = LinearSVC(**SVM_PARAMS)
    svm = CalibratedClassifierCV(svc_base, cv=3)
    svm.fit(X_train, y_train)
    _report("SVM (calibrated)", svm, X_val, y_val)
    joblib.dump(svm, MODEL_A_DIR / "svm.joblib")
    models["svm"] = svm

    # ── Complement Naive Bayes ────────────────────────────────────────────────
    log.info("[Model A] Training Complement Naive Bayes ...")
    nb = ComplementNB(**NB_PARAMS)
    nb.fit(X_train, y_train)
    _report("Complement Naive Bayes", nb, X_val, y_val)
    joblib.dump(nb, MODEL_A_DIR / "nb.joblib")
    models["nb"] = nb

    return models


def _report(name: str, model, X_val, y_val) -> None:
    """Print classification metrics."""
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="macro")
    cm  = confusion_matrix(y_val, y_pred)

    log.info("  %-26s  Acc=%.4f  Macro-F1=%.4f", name, acc, f1)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1  : {f1:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print("\n  Classification Report:")
    print(classification_report(
        y_val, y_pred,
        target_names=["Distractor", "Correct"],
        digits=4,
    ))


# ══════════════════════════════════════════════════════════════════════════════
#  Part 2 — Ensemble (soft-vote: LR + SVM + NB)
# ══════════════════════════════════════════════════════════════════════════════

def train_ensemble(models: dict, X_val: sp.csr_matrix, y_val: np.ndarray) -> dict:
    """
    Soft-vote ensemble: average predict_proba from LR, SVM, NB.
    """
    log.info("[Model A] Building soft-vote ensemble (LR + SVM + NB) ...")
    lr, svm, nb = models["lr"], models["svm"], models["nb"]

    p_lr  = lr.predict_proba(X_val)[:, 1]
    p_svm = svm.predict_proba(X_val)[:, 1]
    p_nb  = nb.predict_proba(X_val)[:, 1]

    avg_proba = (p_lr + p_svm + p_nb) / 3.0
    y_pred    = (avg_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="macro")
    cm  = confusion_matrix(y_val, y_pred)

    print(f"\n{'─'*55}")
    print("  Soft-Vote Ensemble  (LR + SVM + NB)")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1  : {f1:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print(f"{'─'*55}\n")

    ensemble_meta = {
        "components":    ["lr", "svm", "nb"],
        "strategy":      "soft_vote_equal_weights",
        "val_accuracy":  round(acc, 4),
        "val_macro_f1":  round(f1, 4),
    }

    meta_path = MODEL_A_DIR / "ensemble_meta.json"
    with meta_path.open("w") as fh:
        json.dump(ensemble_meta, fh, indent=2)
    log.info("Ensemble meta saved → %s", meta_path)

    return ensemble_meta


# ══════════════════════════════════════════════════════════════════════════════
#  Part 3 — Unsupervised: K-Means clustering
# ══════════════════════════════════════════════════════════════════════════════

def train_kmeans(X_train: sp.csr_matrix, y_train: np.ndarray) -> KMeans:
    """K-Means with k=4 — unsupervised grouping of question-option pairs."""
    log.info("[Model A] K-Means clustering (k=4) ...")
    km = KMeans(**KM_PARAMS)
    cluster_labels = km.fit_predict(X_train)

    from collections import Counter
    total  = len(y_train)
    purity = sum(
        Counter(y_train[cluster_labels == c]).most_common(1)[0][1]
        for c in range(KM_PARAMS["n_clusters"])
    ) / total

    sample_size = min(5000, X_train.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
    sil = silhouette_score(X_train[idx], cluster_labels[idx], metric="cosine")

    print(f"\n{'─'*55}")
    print("  K-Means Clustering  (k=4, unsupervised)")
    print(f"{'─'*55}")
    print(f"  Clustering Purity  : {purity:.4f}  ({purity*100:.2f}%)")
    print(f"  Silhouette Score   : {sil:.4f}  (cosine, n={sample_size:,})")
    print(f"\n  Cluster → class distribution:")
    for c in range(KM_PARAMS["n_clusters"]):
        mask   = cluster_labels == c
        counts = Counter(int(v) for v in y_train[mask])
        print(f"    Cluster {c}: {dict(counts)}  (size={mask.sum()})")
    print(f"{'─'*55}\n")

    joblib.dump(km, MODEL_A_DIR / "kmeans.joblib")
    log.info("K-Means model saved → %s", MODEL_A_DIR / "kmeans.joblib")
    return km


# ══════════════════════════════════════════════════════════════════════════════
#  Part 4 — Semi-Supervised: Label Propagation
# ══════════════════════════════════════════════════════════════════════════════

def train_label_propagation(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    X_val:   sp.csr_matrix,
    y_val:   np.ndarray,
    labeled_fraction: float = 0.05,
) -> LabelPropagation:
    """Label Propagation — 5 % labeled, rest unlabeled (-1)."""
    log.info("[Model A] Label Propagation  (%.0f%% labeled) ...",
             labeled_fraction * 100)

    MAX_LP_ROWS = 15_000
    rng = np.random.default_rng(42)
    n   = min(X_train.shape[0], MAX_LP_ROWS)
    idx = rng.choice(X_train.shape[0], size=n, replace=False)

    X_sub = X_train[idx].toarray()
    y_sub = y_train[idx].copy()

    n_labeled   = max(10, int(n * labeled_fraction))
    labeled_idx = rng.choice(n, size=n_labeled, replace=False)

    y_masked    = np.full_like(y_sub, fill_value=-1)
    y_masked[labeled_idx] = y_sub[labeled_idx]

    log.info("  Training on %d samples (%d labeled, %d unlabeled)",
             n, n_labeled, n - n_labeled)

    lp = LabelPropagation(**LP_PARAMS)
    lp.fit(X_sub, y_masked)

    n_val   = min(X_val.shape[0], 5000)
    idx_val = rng.choice(X_val.shape[0], size=n_val, replace=False)
    X_val_d = X_val[idx_val].toarray()
    y_val_s = y_val[idx_val]

    y_pred = lp.predict(X_val_d)
    acc    = accuracy_score(y_val_s, y_pred)
    f1     = f1_score(y_val_s, y_pred, average="macro")

    print(f"\n{'─'*55}")
    print(f"  Label Propagation  ({labeled_fraction*100:.0f}% labeled)")
    print(f"{'─'*55}")
    print(f"  Labeled samples   : {n_labeled:,} / {n:,}")
    print(f"  Accuracy (val)    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1  (val)   : {f1:.4f}")
    print(f"{'─'*55}\n")

    joblib.dump(lp, MODEL_A_DIR / "label_propagation.joblib")
    log.info("Label Propagation model saved → %s",
             MODEL_A_DIR / "label_propagation.joblib")
    return lp


# ══════════════════════════════════════════════════════════════════════════════
#  Inference wrapper — used by app.py
# ══════════════════════════════════════════════════════════════════════════════

class ModelAVerifier:
    """Thin wrapper for inference from the fitted ensemble."""
    def __init__(self, lr, svm, nb, vectorizer):
        self.lr = lr; self.svm = svm; self.nb = nb
        self.vectorizer = vectorizer

    @classmethod
    def load(cls, model_dir: Path = MODEL_A_DIR,
             vec_path: Path = VECTORIZER_PATH) -> "ModelAVerifier":
        vec = joblib.load(vec_path)
        lr  = joblib.load(model_dir / "lr.joblib")
        svm = joblib.load(model_dir / "svm.joblib")
        nb  = joblib.load(model_dir / "nb.joblib")
        log.info("[ModelAVerifier] Loaded LR + SVM + NB from %s", model_dir)
        return cls(lr=lr, svm=svm, nb=nb, vectorizer=vec)

    def _featurise(self, article: str, question: str, option: str):
        text = article + " " + article + " " + question + " " + option
        return self.vectorizer.transform([text])

    def predict_proba(self, article: str, question: str, option: str) -> float:
        X = self._featurise(article, question, option)
        return float((
            self.lr.predict_proba(X)[0, 1]
            + self.svm.predict_proba(X)[0, 1]
            + self.nb.predict_proba(X)[0, 1]
        ) / 3.0)

    def predict(self, article: str, question: str, option: str) -> int:
        return int(self.predict_proba(article, question, option) >= 0.5)

    def rank_options(self, article: str, question: str,
                     options: dict[str, str]) -> dict[str, float]:
        return {label: self.predict_proba(article, question, text)
                for label, text in options.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: dict) -> None:
    print("\n" + "=" * 65)
    print("  MODEL A — COMPARISON TABLE  (val set)")
    print("=" * 65)
    print(f"  {'Model':<32} {'Accuracy':>10}  {'Macro F1':>10}")
    print("  " + "-" * 55)
    for name, metrics in results.items():
        acc = metrics.get("accuracy", 0)
        f1  = metrics.get("macro_f1", 0)
        print(f"  {name:<32} {acc:>10.4f}  {f1:>10.4f}")
    print("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main training pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_training(quick: bool = False) -> None:
    print("=" * 65)
    print("  Model A — Full Training Pipeline")
    print("  AL2002 Lab Project · FAST-NUCES Islamabad · Spring 2026")
    print("=" * 65)

    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer = _load_vectorizer()

    log.info("[Stage 1] Loading CSVs ...")
    for path in [PROCESSED_DIR / "train.csv", PROCESSED_DIR / "val.csv"]:
        if not path.exists():
            log.error("CSV not found: %s  — run preprocessing.py first.", path)
            sys.exit(1)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")

    if quick:
        train_df = train_df.head(3000)
        val_df   = val_df.head(500)
        log.info("  Quick mode: 3 000 train rows, 500 val rows")

    log.info("[Stage 2] Expanding to binary rows ...")
    X_train, y_train = _expand_to_binary(train_df, vectorizer)
    X_val,   y_val   = _expand_to_binary(val_df,   vectorizer)

    comparison: dict[str, dict] = {}

    # Part 1 — Supervised
    print("\n" + "─"*65)
    print("  PART 1 — Supervised Classifiers  (class_weight='balanced')")
    print("─"*65)
    models = train_supervised(X_train, y_train, X_val, y_val)
    for name, model in models.items():
        y_pred = model.predict(X_val)
        comparison[name.upper()] = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
        }

    # Part 2 — Ensemble
    print("\n" + "─"*65)
    print("  PART 2 — Soft-Vote Ensemble (LR + SVM + NB)")
    print("─"*65)
    ens_meta = train_ensemble(models, X_val, y_val)
    comparison["Ensemble (soft-vote)"] = {
        "accuracy": ens_meta["val_accuracy"],
        "macro_f1": ens_meta["val_macro_f1"],
    }

    # Part 3 — K-Means
    print("\n" + "─"*65)
    print("  PART 3 — K-Means Clustering (Unsupervised)")
    print("─"*65)
    km = train_kmeans(X_train, y_train)
    cluster_labels_val = km.predict(X_val)

    from collections import Counter
    purity_val = sum(
        Counter(y_val[cluster_labels_val == c]).most_common(1)[0][1]
        for c in range(KM_PARAMS["n_clusters"])
    ) / len(y_val)

    comparison["K-Means (unsupervised)"] = {
        "accuracy": round(purity_val, 4),
        "macro_f1": 0.0,
    }

    # Part 4 — Label Propagation
    print("\n" + "─"*65)
    print("  PART 4 — Label Propagation (Semi-Supervised, 5% labeled)")
    print("─"*65)
    lp = train_label_propagation(X_train, y_train, X_val, y_val,
                                  labeled_fraction=0.05)
    rng_e = np.random.default_rng(0)
    n_v   = min(X_val.shape[0], 5000)
    idx_v = rng_e.choice(X_val.shape[0], n_v, replace=False)
    y_lp  = lp.predict(X_val[idx_v].toarray())
    
    comparison["Label Propagation (5% labeled)"] = {
        "accuracy": float(accuracy_score(y_val[idx_v], y_lp)),
        "macro_f1": float(f1_score(y_val[idx_v], y_lp, average="macro")),
    }

    print_comparison_table(comparison)

    results_path = MODEL_A_DIR / "training_results.json"
    with results_path.open("w") as fh:
        json.dump(comparison, fh, indent=2)
    log.info("Results saved → %s", results_path)

    print("=" * 65)
    print("  Training complete.  Artefacts written to:")
    print(f"    {MODEL_A_DIR}/")
    print("      lr.joblib   svm.joblib   nb.joblib")
    print("      kmeans.joblib   label_propagation.joblib")
    print("      ensemble_meta.json   training_results.json")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Model A — Traditional ML + Unsupervised + Ensemble."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Small subset (3k / 500) for smoke-testing.")
    parser.add_argument("--eval",  action="store_true",
                        help="Print extended evaluation report after training.")
    args = parser.parse_args()
    
    run_training(quick=args.quick)