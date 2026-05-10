"""
src/preprocessing.py
====================
RACE preprocessing pipeline (traditional ML only, no neural nets).

Reads CSVs from data/raw/{train,val,test}.csv with columns
[id, article, question, A, B, C, D, answer], and produces:

  models/model_a/tfidf_vectorizer.joblib
  models/model_a/traditional/ohe_vectorizer.pkl
  data/processed/{train,val,test}.csv          (cleaned copies)
  data/processed/X_tfidf_{train,val,test}.npz  (TF-IDF, doc-level)
  data/processed/X_ohe_{train,val,test}.npz    (OHE, option-level expanded)
  data/processed/X_combined_{train,val,test}.npz  (OHE + lexical, option-level)
  data/processed/y_{train,val,test}.npy        (binary labels, option-level)

Reproducibility: all random_state=42.
No fit_transform on val/test.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_A_DIR = ROOT / "models" / "model_a"
MODEL_A_TRAD_DIR = MODEL_A_DIR / "traditional"

TFIDF_PATH = MODEL_A_DIR / "tfidf_vectorizer.joblib"
OHE_PATH = MODEL_A_TRAD_DIR / "ohe_vectorizer.pkl"

CSV_PATHS = {s: PROCESSED_DIR / f"{s}.csv" for s in ("train", "val", "test")}

ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
OPTIONS = ["A", "B", "C", "D"]

TFIDF_PARAMS = dict(stop_words="english", sublinear_tf=True, max_features=15_000)
OHE_PARAMS = dict(binary=True, max_features=15_000, stop_words="english")


# ══════════════════════════════════════════════════════════════════════════════
#  Loading & cleaning
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_csv(split: str) -> pd.DataFrame:
    path = RAW_DIR / f"{split}.csv"
    if not path.exists():
        log.error("Missing raw CSV: %s", path)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"id", "article", "question", "A", "B", "C", "D", "answer"}
    missing = required - set(df.columns)
    if missing:
        log.error("CSV %s missing columns: %s", path, missing)
        sys.exit(1)
    df = df.dropna(subset=["article", "question", "answer"]).reset_index(drop=True)
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()
    df = df[df["answer"].isin(OPTIONS)].reset_index(drop=True)
    log.info("Loaded %s: %d rows", split, len(df))
    return df


def save_processed_csv(df: pd.DataFrame, split: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATHS[split], index=False, encoding="utf-8")
    log.info("Saved %s -> %s", split, CSV_PATHS[split])


# ══════════════════════════════════════════════════════════════════════════════
#  Doc-level TF-IDF (one vector per question, used for hints/distractors)
# ══════════════════════════════════════════════════════════════════════════════

def _build_doc_corpus(df: pd.DataFrame) -> pd.Series:
    a = df["article"].fillna("").astype(str).map(_clean_text)
    q = df["question"].fillna("").astype(str).map(_clean_text)
    A = df["A"].fillna("").astype(str).map(_clean_text)
    B = df["B"].fillna("").astype(str).map(_clean_text)
    C = df["C"].fillna("").astype(str).map(_clean_text)
    D = df["D"].fillna("").astype(str).map(_clean_text)
    return a + " " + a + " " + q + " " + A + " " + B + " " + C + " " + D


def build_tfidf(train_df, val_df, test_df):
    log.info("Building TF-IDF (fit on train only) ...")
    train_corpus = _build_doc_corpus(train_df)
    val_corpus = _build_doc_corpus(val_df)
    test_corpus = _build_doc_corpus(test_df)

    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X_tr = vec.fit_transform(train_corpus)
    X_va = vec.transform(val_corpus)
    X_te = vec.transform(test_corpus)

    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, TFIDF_PATH)
    log.info("TF-IDF vocab=%d, saved -> %s", len(vec.vocabulary_), TFIDF_PATH)
    return X_tr, X_va, X_te, vec


# ══════════════════════════════════════════════════════════════════════════════
#  Option-level expansion: 4 rows per question (1 correct + 3 distractors)
# ══════════════════════════════════════════════════════════════════════════════

def _expand_to_options(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        article = _clean_text(r["article"])
        question = _clean_text(r["question"])
        correct = str(r["answer"]).strip().upper()
        for i, opt in enumerate(OPTIONS):
            opt_text = _clean_text(r.get(opt, ""))
            rows.append({
                "article": article,
                "question": question,
                "option_text": opt_text,
                "option_idx": i,
                "label": 1 if opt == correct else 0,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  One-Hot Encoding (option-level, primary feature representation)
# ══════════════════════════════════════════════════════════════════════════════

def build_ohe_features(train_df, val_df, test_df):
    """
    Fit a binary CountVectorizer on training option-level rows, transform all splits.
    Returns (X_train, X_val, X_test, expanded_dfs_dict, fitted_vectorizer).
    """
    log.info("Building One-Hot Encoding features (option-level) ...")
    expanded = {
        "train": _expand_to_options(train_df),
        "val": _expand_to_options(val_df),
        "test": _expand_to_options(test_df),
    }

    def _combined(df):
        return (df["article"] + " " + df["question"] + " " + df["option_text"]).tolist()

    vec = CountVectorizer(**OHE_PARAMS)
    X_tr = vec.fit_transform(_combined(expanded["train"]))
    X_va = vec.transform(_combined(expanded["val"]))
    X_te = vec.transform(_combined(expanded["test"]))

    MODEL_A_TRAD_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, OHE_PATH)
    log.info("OHE vocab=%d, saved -> %s", len(vec.vocabulary_), OHE_PATH)
    log.info("OHE shapes train=%s val=%s test=%s", X_tr.shape, X_va.shape, X_te.shape)
    return X_tr, X_va, X_te, expanded, vec


# ══════════════════════════════════════════════════════════════════════════════
#  Handcrafted lexical features (option-level)
# ══════════════════════════════════════════════════════════════════════════════

def build_lexical_features(expanded_df: pd.DataFrame) -> np.ndarray:
    """
    Returns a (n_rows, 6) dense float array:
       article_len, question_len, option_len,
       keyword_overlap, answer_in_article, option_position
    """
    feats = np.zeros((len(expanded_df), 6), dtype=np.float32)
    for i, r in enumerate(expanded_df.itertuples(index=False)):
        a_words = str(r.article).split()
        q_words = str(r.question).split()
        o_words = str(r.option_text).split()
        a_set = set(a_words)
        feats[i, 0] = len(a_words)
        feats[i, 1] = len(q_words)
        feats[i, 2] = len(o_words)
        feats[i, 3] = sum(1 for w in o_words if w in a_set)
        feats[i, 4] = 1.0 if str(r.option_text) and str(r.option_text) in str(r.article) else 0.0
        feats[i, 5] = float(r.option_idx)
    return feats


def combine_features(ohe_matrix: sp.csr_matrix, lexical_matrix: np.ndarray) -> sp.csr_matrix:
    """Horizontally stack sparse OHE matrix with sparse lexical matrix."""
    lex_sparse = sp.csr_matrix(lexical_matrix.astype(np.float32))
    return sp.hstack([ohe_matrix, lex_sparse], format="csr")


# ══════════════════════════════════════════════════════════════════════════════
#  Persistence helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_split_artefacts(split: str, X_tfidf, X_ohe, X_combined, y) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(PROCESSED_DIR / f"X_tfidf_{split}.npz", X_tfidf)
    sp.save_npz(PROCESSED_DIR / f"X_ohe_{split}.npz", X_ohe)
    sp.save_npz(PROCESSED_DIR / f"X_combined_{split}.npz", X_combined)
    np.save(PROCESSED_DIR / f"y_{split}.npy", y)
    log.info("Saved %s artefacts (tfidf=%s ohe=%s combined=%s y=%d)",
             split, X_tfidf.shape, X_ohe.shape, X_combined.shape, len(y))


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing() -> None:
    log.info("=" * 65)
    log.info("RACE Preprocessing Pipeline")
    log.info("=" * 65)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_A_TRAD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw CSVs
    train_df = load_raw_csv("train")
    val_df = load_raw_csv("val")
    test_df = load_raw_csv("test")

    # 2. Persist cleaned doc-level CSVs (for downstream model_b/template_generator)
    for split, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        save_processed_csv(df, split)

    # 3. Doc-level TF-IDF (for distractor / hint cosine similarity)
    X_tfidf_tr, X_tfidf_va, X_tfidf_te, _ = build_tfidf(train_df, val_df, test_df)

    # 4. Option-level OHE (for verifier / classifiers)
    X_ohe_tr, X_ohe_va, X_ohe_te, expanded, _ = build_ohe_features(train_df, val_df, test_df)

    # 5. Handcrafted lexical features (option-level) + combined matrix
    log.info("Building lexical features ...")
    lex_tr = build_lexical_features(expanded["train"])
    lex_va = build_lexical_features(expanded["val"])
    lex_te = build_lexical_features(expanded["test"])

    X_comb_tr = combine_features(X_ohe_tr, lex_tr)
    X_comb_va = combine_features(X_ohe_va, lex_va)
    X_comb_te = combine_features(X_ohe_te, lex_te)

    y_tr = expanded["train"]["label"].to_numpy(dtype=np.int8)
    y_va = expanded["val"]["label"].to_numpy(dtype=np.int8)
    y_te = expanded["test"]["label"].to_numpy(dtype=np.int8)

    # 6. Per-split TF-IDF aligned at option-level (each doc-row repeated 4 times)
    X_tfidf_tr_opt = _row_repeat(X_tfidf_tr, 4)
    X_tfidf_va_opt = _row_repeat(X_tfidf_va, 4)
    X_tfidf_te_opt = _row_repeat(X_tfidf_te, 4)

    save_split_artefacts("train", X_tfidf_tr_opt, X_ohe_tr, X_comb_tr, y_tr)
    save_split_artefacts("val",   X_tfidf_va_opt, X_ohe_va, X_comb_va, y_va)
    save_split_artefacts("test",  X_tfidf_te_opt, X_ohe_te, X_comb_te, y_te)

    log.info("=" * 65)
    log.info("Preprocessing complete. Artefacts under data/processed/ and models/model_a/")
    log.info("=" * 65)


def _row_repeat(X: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Repeat each row of sparse matrix k times (alignment with option-level rows)."""
    if X.shape[0] == 0:
        return sp.csr_matrix((0, X.shape[1]))
    idx = np.repeat(np.arange(X.shape[0]), k)
    return X[idx]


if __name__ == "__main__":
    run_preprocessing()
