#Model A (Verify Answer), Logistic Regression 
# for each question label 1 if c, else 0
# for 4 options, 3 bad, so imbalance 
# 1) Load csvsz
# 2) Load joblib
# 3) Build corpus
# 4) Create labels
# 5) Train LR
# 6) Evaluate -> Acc, Macro , CMatrix
# 7) Highest P Correct 
# 8) Save

#23i0757 #23i0541 Model A Code :)
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

#log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

#path
PROCESSED_DIR   = Path("data") / "processed"
MODEL_DIR       = Path("models") / "model_a"

TRAIN_CSV       = PROCESSED_DIR / "train.csv"
VAL_CSV         = PROCESSED_DIR / "val.csv"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "lr_classifier.joblib"


OPTION_COLS: list[str] = ["A", "B", "C", "D"]


ANSWER_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}


LR_PARAMS: dict = dict(
    C=1.0,                     
    class_weight="balanced",  
    solver="saga",            
    max_iter=1000,            
    n_jobs=-1,                
    random_state=42,
)


#1)
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

    # Drop row if null
    before = len(df)
    df = df.dropna(subset=["article", "question", "answer"])
    if len(df) < before:
        log.warning("Dropped %d rows with null article/question/answer.",
                    before - len(df))

    return df.reset_index(drop=True)


def load_vectorizer(path: Path) -> object:
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


#2+3

def expand_to_option_rows(df: pd.DataFrame) -> pd.DataFrame:

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
    
    corpus = build_corpus(df_expanded)

    # transform() only — never fit_transform()
    X: sp.csr_matrix = vectorizer.transform(corpus)
    y: np.ndarray    = df_expanded["is_correct"].to_numpy(dtype=np.int32)

    log.info("  Feature matrix: %s  positives: %d / %d",
             X.shape, y.sum(), len(y))
    return X, y

#4 5
def train_classifier(X_train: sp.csr_matrix, y_train: np.ndarray) -> LogisticRegression:

    log.info("Training LogisticRegression  params=%s", LR_PARAMS)
    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train, y_train)
    log.info("Training complete.")
    return clf


# 67 67 67
def evaluate_binary(
    clf: LogisticRegression,
    X: sp.csr_matrix,
    y: np.ndarray,
    split_name: str,
) -> None:
    
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


#8
def save_classifier(clf: LogisticRegression) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    log.info("Classifier saved -> %s", CLASSIFIER_PATH)



def run_training() -> None:
    log.info("=" * 65)
    log.info("Model A Training — Answer Verifier (Logistic Regression)")
    log.info("=" * 65)

   
    log.info("[Step 1] Loading data and vectorizer ...")
    train_df   = load_csv(TRAIN_CSV)
    val_df     = load_csv(VAL_CSV)
    vectorizer = load_vectorizer(VECTORIZER_PATH)

   
    log.info("[Step 2] Expanding to option-level rows ...")
    log.info("  train:")
    train_exp = expand_to_option_rows(train_df)
    log.info("  val:")
    val_exp   = expand_to_option_rows(val_df)

    
    log.info("[Step 3] Building TF-IDF feature matrices ...")
    log.info("  train:")
    X_train, y_train = build_features(train_exp, vectorizer)
    log.info("  val:")
    X_val,   y_val   = build_features(val_exp,   vectorizer)

    
    log.info("[Step 4] Training classifier ...")
    clf = train_classifier(X_train, y_train)

    
    log.info("[Step 5] Evaluating on validation set ...")
    evaluate_binary(clf, X_val, y_val, split_name="val")
    evaluate_answer_selection(clf, val_exp, X_val, split_name="val")

   
    log.info("[Step 5b] Sanity check on training set ...")
    evaluate_answer_selection(clf, train_exp, X_train, split_name="train")

   
    log.info("[Step 6] Saving classifier ...")
    save_classifier(clf)

    log.info("=" * 65)
    log.info("model_a_train.py complete.")
    log.info("  Vectorizer : %s", VECTORIZER_PATH)
    log.info("  Classifier : %s", CLASSIFIER_PATH)
    log.info("=" * 65)



if __name__ == "__main__":
    run_training()