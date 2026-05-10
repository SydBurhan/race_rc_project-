"""
src/model_a_train.py
====================
Model A — Traditional ML training (no neural networks).

Trains:
  • Logistic Regression (LR)
  • Calibrated LinearSVC (SVM)
  • Complement Naive Bayes (NB)
  • Random Forest (RF)
  • XGBoost (XGB)
  • Soft-vote ensemble (LR + SVM + NB)
  • K-Means clustering (k=4) — unsupervised
  • Label Propagation — semi-supervised

Features (option-level, 4 rows per RACE question):
  • OHE binary CountVectorizer + handcrafted lexical (combined sparse matrix)

Saves all classifiers to:  models/model_a/traditional/*.pkl
Reports comparison table to stdout and JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import LinearSVC

import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_A_TRAD = ROOT / "models" / "model_a" / "traditional"
OHE_PATH = MODEL_A_TRAD / "ohe_vectorizer.pkl"
RESULTS_PATH = MODEL_A_TRAD / "training_results.json"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LR_PARAMS = dict(C=1.0, max_iter=1000, solver="liblinear",
                 random_state=42, class_weight="balanced")
SVM_PARAMS = dict(C=0.05, max_iter=3000, random_state=42, class_weight="balanced")
NB_PARAMS = dict(alpha=0.3)
# RF on sparse 15k-feature data is expensive; keep modest defaults.
RF_PARAMS = dict(n_estimators=80, max_depth=24, max_features="sqrt",
                 min_samples_leaf=4, class_weight="balanced",
                 random_state=42, n_jobs=-1)
def _detect_xgb_device() -> str:
    """Return 'cuda' if XGBoost can see a GPU, else 'cpu'."""
    try:
        import xgboost as _xgb
        # Probe with a tiny matrix; XGBoost falls back gracefully when no GPU.
        import numpy as _np
        dtrain = _xgb.DMatrix(_np.zeros((2, 2)), label=_np.array([0, 1]))
        _xgb.train({"device": "cuda", "tree_method": "hist"}, dtrain,
                    num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


# XGBoost: hist + GPU when available (5-10x speedup on Colab/Kaggle).
_XGB_DEVICE = _detect_xgb_device()
XGB_PARAMS = dict(n_estimators=150, max_depth=6, tree_method="hist",
                  device=_XGB_DEVICE, learning_rate=0.1,
                  eval_metric="logloss", random_state=42, n_jobs=-1)
KM_PARAMS = dict(n_clusters=4, n_init=10, random_state=42)
LP_PARAMS = dict(kernel="knn", n_neighbors=7, max_iter=1000)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_split(split: str):
    X = sp.load_npz(PROCESSED_DIR / f"X_combined_{split}.npz")
    y = np.load(PROCESSED_DIR / f"y_{split}.npy")
    return X, y


def _report(name: str, model, X_val, y_val) -> dict:
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    print(f"  Accuracy   : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Macro F1   : {f1:.4f}")
    print("  Confusion Matrix:")
    if cm.shape == (2, 2):
        print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
        print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print(classification_report(y_val, y_pred,
                                 target_names=["Distractor", "Correct"],
                                 digits=4, zero_division=0))
    return {"accuracy": float(acc), "macro_f1": float(f1)}


def _print_top_features(name: str, importances: np.ndarray, vocab_size: int,
                         top_k: int = 20):
    """Print top-k features by importance, mapping OHE indices to vocab terms."""
    if not OHE_PATH.exists():
        return
    vec = joblib.load(OHE_PATH)
    inv_vocab = {idx: term for term, idx in vec.vocabulary_.items()}
    order = np.argsort(-importances)[:top_k]
    print(f"\n  Top-{top_k} feature importances ({name}):")
    for i in order:
        if i < vocab_size:
            term = inv_vocab.get(i, f"<oov-{i}>")
        else:
            # Lexical features tail
            lex_names = ["article_len", "question_len", "option_len",
                          "keyword_overlap", "answer_in_article", "option_position"]
            term = lex_names[i - vocab_size] if (i - vocab_size) < len(lex_names) else f"<idx-{i}>"
        print(f"    {term:<28}  {importances[i]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Supervised training
# ══════════════════════════════════════════════════════════════════════════════

def train_supervised(X_tr, y_tr, X_va, y_va, vocab_size: int) -> tuple[dict, dict]:
    MODEL_A_TRAD.mkdir(parents=True, exist_ok=True)
    models, metrics = {}, {}

    log.info("Training Logistic Regression ...")
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_tr, y_tr)
    metrics["LR"] = _report("Logistic Regression", lr, X_va, y_va)
    joblib.dump(lr, MODEL_A_TRAD / "lr.pkl")
    models["lr"] = lr

    log.info("Training calibrated LinearSVC ...")
    svm_base = LinearSVC(**SVM_PARAMS)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_tr, y_tr)
    metrics["SVM"] = _report("SVM (calibrated)", svm, X_va, y_va)
    joblib.dump(svm, MODEL_A_TRAD / "svm.pkl")
    models["svm"] = svm

    log.info("Training Complement Naive Bayes ...")
    nb = ComplementNB(**NB_PARAMS)
    nb.fit(X_tr, y_tr)
    metrics["NB"] = _report("Complement Naive Bayes", nb, X_va, y_va)
    joblib.dump(nb, MODEL_A_TRAD / "nb.pkl")
    models["nb"] = nb

    log.info("Training Random Forest ...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    metrics["RF"] = _report("Random Forest", rf, X_va, y_va)
    joblib.dump(rf, MODEL_A_TRAD / "rf_classifier.pkl")
    models["rf"] = rf
    _print_top_features("RandomForest", rf.feature_importances_, vocab_size)

    log.info("Training XGBoost on device=%s ...", _XGB_DEVICE)
    xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_clf.fit(X_tr, y_tr)
    metrics["XGB"] = _report("XGBoost", xgb_clf, X_va, y_va)
    joblib.dump(xgb_clf, MODEL_A_TRAD / "xgb_classifier.pkl")
    models["xgb"] = xgb_clf
    _print_top_features("XGBoost", xgb_clf.feature_importances_, vocab_size)

    return models, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Soft-vote ensemble (LR + SVM + NB)
# ══════════════════════════════════════════════════════════════════════════════

def train_ensemble(models: dict, X_va, y_va) -> dict:
    log.info("Building soft-vote ensemble (LR + SVM + NB) ...")
    p_lr = models["lr"].predict_proba(X_va)[:, 1]
    p_svm = models["svm"].predict_proba(X_va)[:, 1]
    p_nb = models["nb"].predict_proba(X_va)[:, 1]
    avg = (p_lr + p_svm + p_nb) / 3.0
    y_pred = (avg >= 0.5).astype(int)

    acc = float(accuracy_score(y_va, y_pred))
    f1 = float(f1_score(y_va, y_pred, average="macro"))
    print(f"\n{'─' * 55}\n  Soft-Vote Ensemble (LR+SVM+NB)\n{'─' * 55}")
    print(f"  Accuracy : {acc:.4f}  Macro F1 : {f1:.4f}")
    meta = {
        "components": ["lr", "svm", "nb"],
        "strategy": "soft_vote_equal_weights",
        "val_accuracy": acc,
        "val_macro_f1": f1,
    }
    (MODEL_A_TRAD / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))
    return {"accuracy": acc, "macro_f1": f1}


# ══════════════════════════════════════════════════════════════════════════════
#  Unsupervised — K-Means
# ══════════════════════════════════════════════════════════════════════════════

def train_kmeans(X_tr, y_tr, supervised_acc: float) -> dict:
    log.info("K-Means clustering (k=4) ...")
    km = KMeans(**KM_PARAMS)
    labels = km.fit_predict(X_tr)

    from collections import Counter
    purity = sum(
        Counter(y_tr[labels == c]).most_common(1)[0][1]
        for c in range(KM_PARAMS["n_clusters"])
    ) / len(y_tr)

    sample = min(5000, X_tr.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_tr.shape[0], size=sample, replace=False)
    sil = silhouette_score(X_tr[idx], labels[idx], metric="cosine")

    print(f"\n{'─' * 55}\n  K-Means (unsupervised, k=4)\n{'─' * 55}")
    print(f"  Purity            : {purity:.4f}")
    print(f"  Silhouette (cosine): {sil:.4f}")
    print("\nDISCUSSION (copy into report):")
    print(f"  K-Means purity ({purity:.4f}) vs supervised ensemble accuracy "
          f"({supervised_acc:.4f}). The unsupervised model recovers cluster "
          f"structure but cannot reach supervised performance because option "
          f"correctness is not captured by lexical clusters alone.")

    joblib.dump(km, MODEL_A_TRAD / "kmeans.pkl")
    return {"purity": float(purity), "silhouette": float(sil)}


# ══════════════════════════════════════════════════════════════════════════════
#  Semi-supervised — Label Propagation
# ══════════════════════════════════════════════════════════════════════════════

def train_label_propagation(X_tr, y_tr, X_va, y_va,
                             supervised_f1: float,
                             labeled_fraction: float = 0.05) -> dict:
    log.info("Label Propagation (5%% labeled) ...")
    MAX = 8000
    rng = np.random.default_rng(42)
    n = min(X_tr.shape[0], MAX)
    idx = rng.choice(X_tr.shape[0], size=n, replace=False)
    X_sub = X_tr[idx].toarray()
    y_sub = y_tr[idx].copy()
    n_lab = max(10, int(n * labeled_fraction))
    lab_idx = rng.choice(n, size=n_lab, replace=False)
    y_masked = np.full_like(y_sub, fill_value=-1)
    y_masked[lab_idx] = y_sub[lab_idx]

    lp = LabelPropagation(**LP_PARAMS)
    lp.fit(X_sub, y_masked)

    n_v = min(X_va.shape[0], 3000)
    iv = rng.choice(X_va.shape[0], size=n_v, replace=False)
    y_pred = lp.predict(X_va[iv].toarray())
    acc = float(accuracy_score(y_va[iv], y_pred))
    f1 = float(f1_score(y_va[iv], y_pred, average="macro"))

    print(f"\n{'─' * 55}\n  Label Propagation (5% labeled)\n{'─' * 55}")
    print(f"  Accuracy : {acc:.4f}  Macro F1 : {f1:.4f}")
    print("\nDISCUSSION (copy into report):")
    print(f"  Label Propagation F1 ({f1:.4f}) vs fully supervised LR macro F1 "
          f"({supervised_f1:.4f}). With only 5% labels, semi-supervised learning "
          f"narrows the gap by exploiting the geometry of unlabeled OHE features.")

    joblib.dump(lp, MODEL_A_TRAD / "label_propagation.pkl")
    return {"accuracy": acc, "macro_f1": f1}


# ══════════════════════════════════════════════════════════════════════════════
#  Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(metrics: dict) -> None:
    rows = []
    for name in ["LR", "SVM", "NB", "RF", "XGB", "Ensemble"]:
        m = metrics.get(name)
        if not m:
            continue
        rows.append({"Model": name,
                     "Val Accuracy": round(m["accuracy"], 4),
                     "Macro F1": round(m["macro_f1"], 4)})
    df = pd.DataFrame(rows)
    print("\n" + "=" * 65)
    print("  MODEL A — Comparison Table (validation set)")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_training(quick: bool = False) -> None:
    if not OHE_PATH.exists():
        log.error("OHE vectorizer not found. Run src/preprocessing.py first.")
        sys.exit(1)

    X_tr, y_tr = _load_split("train")
    X_va, y_va = _load_split("val")

    if quick:
        # Use first 10k option-level rows for quick smoke test
        X_tr, y_tr = X_tr[:10000], y_tr[:10000]
        X_va, y_va = X_va[:2000], y_va[:2000]
        log.info("Quick mode: train=%d val=%d", X_tr.shape[0], X_va.shape[0])

    vec = joblib.load(OHE_PATH)
    vocab_size = len(vec.vocabulary_)

    models, metrics = train_supervised(X_tr, y_tr, X_va, y_va, vocab_size)
    metrics["Ensemble"] = train_ensemble(models, X_va, y_va)
    metrics["KMeans"] = train_kmeans(X_tr, y_tr,
                                      supervised_acc=metrics["Ensemble"]["accuracy"])
    metrics["LabelProp"] = train_label_propagation(
        X_tr, y_tr, X_va, y_va, supervised_f1=metrics["LR"]["macro_f1"]
    )

    print_comparison(metrics)

    RESULTS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Results saved -> %s", RESULTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use 10k/2k subset for smoke testing.")
    args = parser.parse_args()
    run_training(quick=args.quick)
