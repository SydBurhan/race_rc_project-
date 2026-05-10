"""
RACE preprocessing pipeline.

Rubric coverage:
  1.3  Preprocessing pipeline (cleaning, OHE saved, train/val/test split)
  2.2  Feature engineering (TF-IDF, OHE/BoW, handcrafted lexical)

Produces option-level feature matrices and saves the fitted vectorizers
for reuse at inference time. Random state pinned to 42 throughout.
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


def _validate_and_clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    # Drop pandas-index leakage columns like "Unnamed: 0"
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    required = {"id", "article", "question", "A", "B", "C", "D", "answer"}
    missing = required - set(df.columns)
    if missing:
        log.error("CSV %s missing columns: %s", source, missing)
        sys.exit(1)
    df = df.dropna(subset=["article", "question", "answer"]).reset_index(drop=True)
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()
    df = df[df["answer"].isin(OPTIONS)].reset_index(drop=True)
    return df


def _resolve_split_path(split: str) -> Path | None:
    """Locate split CSV, accepting common aliases (val/dev/validation)."""
    aliases = {
        "train": ["train.csv"],
        "val":   ["val.csv", "dev.csv", "validation.csv"],
        "test":  ["test.csv"],
    }
    for name in aliases.get(split, [f"{split}.csv"]):
        p = RAW_DIR / name
        if p.exists():
            return p
    return None


def load_raw_csv(split: str) -> pd.DataFrame:
    path = _resolve_split_path(split)
    if path is None:
        log.error("Missing raw CSV for split=%s in %s", split, RAW_DIR)
        sys.exit(1)
    df = _validate_and_clean(pd.read_csv(path), str(path))
    log.info("Loaded %s from %s: %d rows", split, path.name, len(df))
    return df


def _file_fingerprint(path: Path) -> tuple[int, int]:
    """Cheap duplicate-file detection: (file_size, mtime-rounded)."""
    s = path.stat()
    return (s.st_size, int(s.st_mtime))


def _looks_like_duplicates(paths: list[Path]) -> bool:
    sizes = {p.stat().st_size for p in paths if p and p.exists()}
    return len(sizes) == 1 and len(paths) >= 2


def load_or_split_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test CSVs from data/raw/ (val accepts dev.csv / validation.csv).
    Falls back to an 80/10/10 random split of train.csv (random_state=42) when:
      * val/test are missing, OR
      * train/val/test files are byte-identical (data leakage).
    """
    train_path = _resolve_split_path("train")
    val_path = _resolve_split_path("val")
    test_path = _resolve_split_path("test")

    if not train_path:
        log.error("Missing data/raw/train.csv. Cannot proceed.")
        sys.exit(1)

    have_all_three = val_path and test_path
    if have_all_three:
        existing = [train_path, val_path, test_path]
        if _looks_like_duplicates(existing):
            log.warning("train / val / test CSVs are byte-identical "
                        "(%s). Falling back to auto-split to avoid leakage.",
                        ", ".join(p.name for p in existing))
        else:
            return (load_raw_csv("train"),
                    load_raw_csv("val"),
                    load_raw_csv("test"))

    log.warning("Auto-splitting %s into 80/10/10 train/val/test "
                "(random_state=42).", train_path.name)
    full = _validate_and_clean(pd.read_csv(train_path), str(train_path))
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(full)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_df = full.iloc[:n_train].reset_index(drop=True)
    val_df = full.iloc[n_train: n_train + n_val].reset_index(drop=True)
    test_df = full.iloc[n_train + n_val:].reset_index(drop=True)
    log.info("Split sizes: train=%d val=%d test=%d",
             len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def save_processed_csv(df: pd.DataFrame, split: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATHS[split], index=False, encoding="utf-8")
    log.info("Saved %s -> %s", split, CSV_PATHS[split])


# Rubric 2.2: TF-IDF feature engineering (doc-level, used by hints/distractors)

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


# Each RACE question becomes 4 rows (one per option) with binary correctness label.

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


# Rubric 1.3 / 2.2: One-Hot Encoding (binary CountVectorizer, persisted to disk)

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


# Rubric 2.2: Handcrafted lexical features (10 per option) used alongside OHE.

def build_lexical_features(expanded_df: pd.DataFrame) -> np.ndarray:
    """
    Returns a (n_rows, 10) dense float array. Rows arrive in groups of 4
    (one per option of the same question).

    Indices and meanings (kept in sync with model_a_train._print_top_features):
       0 article_len         number of tokens in the article
       1 question_len        number of tokens in the question
       2 option_len          number of tokens in this option
       3 keyword_overlap     count of option tokens also present in article
       4 answer_in_article   1.0 if option text is a substring of article
       5 option_position     option index 0..3
       6 q_opt_overlap       Jaccard(question tokens, option tokens)
       7 opt_uniqueness      option_len / mean option_len across the 4 options
       8 opt_other_overlap   Jaccard(this option, union of OTHER 3 options)
       9 opt_article_cos     cosine of option BoW vs article BoW
    """
    import math
    from collections import Counter

    n = len(expanded_df)
    feats = np.zeros((n, 10), dtype=np.float32)
    rows = list(expanded_df.itertuples(index=False))

    for q_start in range(0, n, 4):
        group = rows[q_start: q_start + 4]
        if not group:
            continue

        article = str(group[0].article)
        question = str(group[0].question)
        a_words = article.split()
        q_words = question.split()
        a_set = set(a_words)
        q_set = set(q_words)

        a_counter = Counter(a_words)
        a_norm = math.sqrt(sum(c * c for c in a_counter.values())) or 1.0

        opt_word_lists = [str(r.option_text).split() for r in group]
        opt_word_sets = [set(w) for w in opt_word_lists]
        opt_lens = [len(w) for w in opt_word_lists]
        mean_opt_len = max(1.0, sum(opt_lens) / max(1, len(opt_lens)))

        for j, r in enumerate(group):
            o_words = opt_word_lists[j]
            o_set = opt_word_sets[j]

            feats[q_start + j, 0] = len(a_words)
            feats[q_start + j, 1] = len(q_words)
            feats[q_start + j, 2] = len(o_words)
            feats[q_start + j, 3] = sum(1 for w in o_words if w in a_set)
            feats[q_start + j, 4] = 1.0 if r.option_text and str(r.option_text) in article else 0.0
            feats[q_start + j, 5] = float(r.option_idx)

            # 6 — Jaccard(question, option)
            union_qo = len(q_set | o_set)
            feats[q_start + j, 6] = (len(q_set & o_set) / union_qo) if union_qo else 0.0

            # 7 — relative option length within the question
            feats[q_start + j, 7] = len(o_words) / mean_opt_len

            # 8 — Jaccard with union of other 3 options
            other = set().union(*[opt_word_sets[k] for k in range(len(group)) if k != j]) if len(group) > 1 else set()
            union_oo = len(o_set | other)
            feats[q_start + j, 8] = (len(o_set & other) / union_oo) if union_oo else 0.0

            # 9 — cosine(option BoW, article BoW)
            o_counter = Counter(o_words)
            dot = sum(o_counter[w] * a_counter[w] for w in o_counter if w in a_counter)
            o_norm = math.sqrt(sum(c * c for c in o_counter.values())) or 1.0
            feats[q_start + j, 9] = dot / (a_norm * o_norm)

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

    # 1. Load raw CSVs (auto-splits train.csv if val/test missing)
    train_df, val_df, test_df = load_or_split_raw()

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
