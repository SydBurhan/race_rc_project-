"""
model_a_train.py  —  RACE Reading Comprehension System  (AL2002 Lab Project)
=============================================================================
Model A: Answer Verifier — Logistic Regression on TF-IDF features.

How the verifier works
-----------------------
The RACE dataset has 4 candidate options (A, B, C, D) per question.
Rather than building a 4-class classifier directly, we reframe the task
as a binary relevance problem:

    For each (article, question, option) triple, predict:
        label = 1  if this option is the correct answer
        label = 0  otherwise

During inference we score all 4 options and pick argmax(probability of 1).

Why this formulation?
    - Naturally handles the ~75 % / 25 % class imbalance of RACE (3 wrong
      options for every 1 correct one).
    - class_weight='balanced' in Logistic Regression adjusts for this.
    - Produces a confidence score we can use for ranking and hint generation.

Pipeline
--------
1. Load data/processed/train.csv  and  data/processed/val.csv
2. Load models/model_a/tfidf_vectorizer.joblib  (already fitted on train)
3. Feature engineering: expand each row into 4 option rows, build corpus
       combined = article + article + question + option
   and call vectorizer.transform() — never fit_transform()
4. Create binary labels  (1 if option == answer, else 0)
5. Train LogisticRegression with class_weight='balanced', solver='saga',
   max_iter=1000 (saga converges reliably on sparse TF-IDF matrices)
6. Evaluate on validation set: Accuracy, Macro F1, Confusion Matrix
7. Evaluate answer-selection accuracy: pick the option with highest P(correct)
   per question and compare to ground truth
8. Save models/model_a/lr_classifier.joblib
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR   = Path("data") / "processed"
MODEL_DIR       = Path("models") / "model_a"

TRAIN_CSV       = PROCESSED_DIR / "train.csv"
VAL_CSV         = PROCESSED_DIR / "val.csv"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "lr_classifier.joblib"

# Option columns in the schema order
OPTION_COLS: list[str] = ["A", "B", "C", "D"]

# Answer letter -> integer index
ANSWER_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}

# ---------------------------------------------------------------------------
# Logistic Regression hyper-parameters
# ---------------------------------------------------------------------------
LR_PARAMS: dict = dict(
    C=1.0,                    # inverse regularisation strength
    class_weight="balanced",  # compensates for 3:1 wrong:correct imbalance
    solver="saga",            # best solver for large sparse matrices
    max_iter=1000,            # saga needs more iterations than lbfgs
    n_jobs=-1,                # parallelise across all CPU cores
    random_state=42,
)


# ===========================================================================
# 1. Data loading
# ===========================================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Load a processed CSV, validate required columns, return DataFrame."""
    if not path.exists():
        log.error("File not found: %s", path)
        sys.exit(1)

    df = pd.read_csv(path)
    log.info("Loaded %s  (%d rows)", path, len(df))

    required = {"article", "question", "answer"} | set(OPTION_COLS)
    missing  = required - set(df.columns)
    if missing:
        log.error("Missing columns in %s: %s", path, missing)
        sys.exit(1)

    # Drop rows where critical fields are null
    before = len(df)
    df = df.dropna(subset=["article", "question", "answer"])
    if len(df) < before:
        log.warning("Dropped %d rows with null article/question/answer.",
                    before - len(df))

    return df.reset_index(drop=True)


def load_vectorizer(path: Path) -> object:
    """Load the pre-fitted TfidfVectorizer from disk."""
    if not path.exists():
        log.error(
            "Vectorizer not found: %s\n"
            "  Run preprocessing.py first to generate it.",
            path,
        )
        sys.exit(1)

    vectorizer = joblib.load(path)
    log.info(
        "Loaded vectorizer from %s  (vocab size: %d)",
        path,
        len(vectorizer.vocabulary_),
    )
    return vectorizer


# ===========================================================================
# 2 & 3. Feature engineering — expand + transform
# ===========================================================================

def expand_to_option_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each question row into 4 option rows (one per candidate answer).

    Input row columns : id, article, question, A, B, C, D, answer
    Output row columns: id, article, question, option, option_letter,
                        answer, is_correct

    This is the standard "pointwise" ranking formulation used in reading-
    comprehension retrieval: score each option independently, pick the best.
    """
    records = []
    for _, row in df.iterrows():
        answer_letter = str(row["answer"]).strip().upper()
        for letter in OPTION_COLS:
            records.append({
                "id":            row["id"],
                "article":       str(row["article"]).strip(),
                "question":      str(row["question"]).strip(),
                "option":        str(row[letter]).strip(),
                "option_letter": letter,
                "answer":        answer_letter,
                # Binary label: 1 if this option is the correct answer
                "is_correct":    1 if letter == answer_letter else 0,
            })

    expanded = pd.DataFrame(records)
    log.info(
        "  Expanded %d question-rows -> %d option-rows  "
        "(positive rate: %.1f %%)",
        len(df),
        len(expanded),
        100.0 * expanded["is_correct"].mean(),
    )
    return expanded


def build_corpus(df: pd.DataFrame) -> pd.Series:
    """
    Build the text corpus for TF-IDF transformation.

    Formula (TF-IDF Manual §6 best practice):
        combined = article + article + question + option

    The article is repeated to give passage content twice the term-frequency
    weight compared to the shorter question and option strings.
    """
    return (
        df["article"] + " "
        + df["article"] + " "
        + df["question"] + " "
        + df["option"]
    )


def build_features(
    df_expanded: pd.DataFrame,
    vectorizer,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Transform the expanded option-level DataFrame into a sparse TF-IDF matrix
    and extract binary labels.

    CRITICAL: Only .transform() is called here — the vectorizer vocabulary
    and IDF weights were fixed during preprocessing. Re-fitting on train or
    val data would constitute data leakage into future splits.
    """
    corpus = build_corpus(df_expanded)

    # transform() only — never fit_transform()
    X: sp.csr_matrix = vectorizer.transform(corpus)
    y: np.ndarray    = df_expanded["is_correct"].to_numpy(dtype=np.int32)

    log.info("  Feature matrix: %s  positives: %d / %d",
             X.shape, y.sum(), len(y))
    return X, y


# ===========================================================================
# 4 & 5. Training
# ===========================================================================

def train_classifier(X_train: sp.csr_matrix, y_train: np.ndarray) -> LogisticRegression:
    """
    Train a Logistic Regression answer verifier.

    Solver choice — 'saga':
        SAGA is the recommended solver for large sparse feature matrices
        (scikit-learn docs). It handles L2 regularisation with class weights
        efficiently without converting the sparse matrix to dense.

    class_weight='balanced':
        RACE has 3 wrong options per 1 correct option (~75 / 25 split).
        Balanced weighting prevents the model from collapsing to "always
        predict wrong".
    """
    log.info("Training LogisticRegression  params=%s", LR_PARAMS)
    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train, y_train)
    log.info("Training complete.")
    return clf


# ===========================================================================
# 6 & 7. Evaluation
# ===========================================================================

def evaluate_binary(
    clf: LogisticRegression,
    X: sp.csr_matrix,
    y: np.ndarray,
    split_name: str,
) -> None:
    """
    Print binary classification metrics (option-level).

    This tells us how well the model distinguishes correct from wrong options
    when each option is judged independently.
    """
    y_pred    = clf.predict(X)
    accuracy  = accuracy_score(y, y_pred)
    macro_f1  = f1_score(y, y_pred, average="macro")
    cm        = confusion_matrix(y, y_pred)

    log.info("")
    log.info("── Binary (option-level) evaluation on %s ──", split_name)
    log.info("  Accuracy  : %.4f", accuracy)
    log.info("  Macro F1  : %.4f", macro_f1)
    log.info("  Confusion matrix (rows=actual, cols=predicted):")
    log.info("              Pred-0   Pred-1")
    log.info("  Actual-0   %6d   %6d", cm[0, 0], cm[0, 1])
    log.info("  Actual-1   %6d   %6d", cm[1, 0], cm[1, 1])
    log.info("")
    log.info("  Full classification report:")
    print(classification_report(y, y_pred, target_names=["Wrong (0)", "Correct (1)"]))


def evaluate_answer_selection(
    clf: LogisticRegression,
    df_expanded: pd.DataFrame,
    X: sp.csr_matrix,
    split_name: str,
) -> float:
    """
    Evaluate question-level answer selection accuracy.

    For each question we score all 4 options, pick the one with the highest
    P(is_correct=1), and check if it matches the ground-truth answer.

    This is the metric that matters for the UI: "did the model pick the right
    answer for this question?"
    """
    # P(is_correct=1) for each option row
    proba = clf.predict_proba(X)[:, 1]

    df_eval = df_expanded[["id", "option_letter", "answer", "is_correct"]].copy()
    df_eval["score"] = proba

    # For each question (id), pick the option with the highest score
    best_idx = df_eval.groupby("id")["score"].idxmax()
    selected = df_eval.loc[best_idx, ["id", "option_letter", "answer"]].copy()
    selected["correct"] = (
        selected["option_letter"] == selected["answer"]
    )

    selection_accuracy = selected["correct"].mean()

    log.info("── Answer-selection accuracy on %s ──", split_name)
    log.info(
        "  Selected the correct answer for %d / %d questions  (%.2f %%)",
        selected["correct"].sum(),
        len(selected),
        100.0 * selection_accuracy,
    )
    log.info("")
    return selection_accuracy


# ===========================================================================
# 8. Persist model
# ===========================================================================

def save_classifier(clf: LogisticRegression) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    log.info("Classifier saved -> %s", CLASSIFIER_PATH)


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_training() -> None:
    log.info("=" * 65)
    log.info("Model A Training — Answer Verifier (Logistic Regression)")
    log.info("=" * 65)

    # ── 1. Load data and vectorizer ───────────────────────────────────────
    log.info("[Step 1] Loading data and vectorizer ...")
    train_df   = load_csv(TRAIN_CSV)
    val_df     = load_csv(VAL_CSV)
    vectorizer = load_vectorizer(VECTORIZER_PATH)

    # ── 2. Expand rows to option level ────────────────────────────────────
    log.info("[Step 2] Expanding to option-level rows ...")
    log.info("  train:")
    train_exp = expand_to_option_rows(train_df)
    log.info("  val:")
    val_exp   = expand_to_option_rows(val_df)

    # ── 3. Build TF-IDF feature matrices ─────────────────────────────────
    log.info("[Step 3] Building TF-IDF feature matrices ...")
    log.info("  train:")
    X_train, y_train = build_features(train_exp, vectorizer)
    log.info("  val:")
    X_val,   y_val   = build_features(val_exp,   vectorizer)

    # ── 4 & 5. Train ──────────────────────────────────────────────────────
    log.info("[Step 4] Training classifier ...")
    clf = train_classifier(X_train, y_train)

    # ── 6 & 7. Evaluate ───────────────────────────────────────────────────
    log.info("[Step 5] Evaluating on validation set ...")
    evaluate_binary(clf, X_val, y_val, split_name="val")
    evaluate_answer_selection(clf, val_exp, X_val, split_name="val")

    # Quick sanity-check on training set (should be higher — checks for bugs)
    log.info("[Step 5b] Sanity check on training set ...")
    evaluate_answer_selection(clf, train_exp, X_train, split_name="train")

    # ── 8. Save ───────────────────────────────────────────────────────────
    log.info("[Step 6] Saving classifier ...")
    save_classifier(clf)

    log.info("=" * 65)
    log.info("model_a_train.py complete.")
    log.info("  Vectorizer : %s", VECTORIZER_PATH)
    log.info("  Classifier : %s", CLASSIFIER_PATH)
    log.info("=" * 65)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_training()