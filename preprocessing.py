"""
preprocessing.py  —  RACE Reading Comprehension System  (AL2002 Lab Project)
=============================================================================

Pipeline
--------
Stage 1 — Parse raw RACE directories
    Walks  data/raw/train/, data/raw/dev/, data/raw/test/
    Each file is a JSON object with the schema used by the original RACE
    dataset release:

        {
            "article":   "...",
            "questions": ["...", "..."],
            "options":   [["A_text","B_text","C_text","D_text"], ...],
            "answers":   ["A", "C", ...]      # one answer per question
        }

    One question  →  one row in the DataFrame.

Stage 2 — Persist CSVs
    Saves three files to data/processed/ so the slow directory-walk
    only happens once:
        train.csv  |  val.csv  |  test.csv

    Schema (matches project specification exactly):
        id | article | question | A | B | C | D | answer

Stage 3 — TF-IDF vectorisation
    Corpus per row = article + article + question + A + B + C + D
    (article repeated to up-weight passage content — TF-IDF Manual §6)

    Vectorizer constraints (fixed by AL2002 project spec):
        stop_words='english'   — removes noise dimensions
        sublinear_tf=True      — TF = log(1 + count)
        max_features=15000     — safe ceiling for RACE's 100K+ vocab

    DATA-LEAKAGE PREVENTION
    -------------------------
    fit_transform()  ← training corpus ONLY
    transform()      ← val and test (no refitting, ever)

Stage 4 — Persist artefacts
    models/model_a/tfidf_vectorizer.joblib   ← fitted vectorizer
    data/processed/X_train.npz              ← sparse TF-IDF matrices
    data/processed/X_val.npz
    data/processed/X_test.npz
    data/processed/y_train.npy              ← integer labels (0-3)
    data/processed/y_val.npy
    data/processed/y_test.npy   (if answers are present in test split)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

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
# Directory / file paths
# ---------------------------------------------------------------------------
RAW_DIR       = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
MODEL_DIR     = Path("models") / "model_a"

# Raw RACE subdirectory names.
# The original RACE release uses "dev" for the validation split.
RAW_SUBDIRS: dict[str, str] = {
    "train": "train",
    "val":   "dev",    # RACE names the val split "dev" on disk
    "test":  "test",
}

CSV_PATHS: dict[str, Path] = {
    split: PROCESSED_DIR / f"{split}.csv"
    for split in ("train", "val", "test")
}

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

# ---------------------------------------------------------------------------
# Project-mandated TF-IDF hyper-parameters (do not change)
# ---------------------------------------------------------------------------
TFIDF_PARAMS: dict = dict(
    stop_words="english",
    sublinear_tf=True,
    max_features=15_000,
)

# Answer letter -> integer label
ANSWER_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}


# ===========================================================================
# Stage 1 — Parse raw RACE directory
# ===========================================================================

def _parse_race_file(filepath: Path, split_name: str) -> list[dict]:
    """
    Parse a single RACE JSON file and return a list of row dicts, one per
    question in that article.

    RACE JSON schema
    ----------------
    {
        "article":   str,
        "questions": [str, ...],
        "options":   [[A, B, C, D], ...],   # parallel to questions
        "answers":   [str, ...]             # "A"|"B"|"C"|"D", may be absent
    }
    """
    rows: list[dict] = []

    try:
        with filepath.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Skipping %s — could not parse: %s", filepath, exc)
        return rows

    article   = str(data.get("article",   "")).strip()
    questions = data.get("questions", [])
    options   = data.get("options",   [])
    answers   = data.get("answers",   [])   # may be empty list for test split

    if not questions:
        log.warning("No questions in %s — skipping file.", filepath)
        return rows

    for q_idx, question in enumerate(questions):
        opts = options[q_idx] if q_idx < len(options) else ["", "", "", ""]

        # Pad / trim to exactly 4 options
        while len(opts) < 4:
            opts.append("")
        opts = opts[:4]

        answer = answers[q_idx] if q_idx < len(answers) else None

        row = {
            # Unique identifier: filename stem + question index
            "id":       f"{split_name}_{filepath.stem}_{q_idx}",
            "article":  article,
            "question": str(question).strip(),
            "A":        str(opts[0]).strip(),
            "B":        str(opts[1]).strip(),
            "C":        str(opts[2]).strip(),
            "D":        str(opts[3]).strip(),
            "answer":   str(answer).strip() if answer is not None else None,
        }
        rows.append(row)

    return rows


def parse_race_directory(split_name: str) -> pd.DataFrame:
    """
    Recursively walk the raw RACE subdirectory for *split_name* and
    assemble all JSON files into a single DataFrame.

    Handles both flat layouts  (data/raw/train/*.txt)  and the nested layout
    the original RACE release ships as  (data/raw/train/high/*.txt, .../middle/*.txt).
    Any file that json.load can parse is accepted regardless of extension.
    """
    subdir_name = RAW_SUBDIRS[split_name]
    root_dir    = RAW_DIR / subdir_name

    if not root_dir.exists():
        log.error(
            "Raw directory not found: %s\n"
            "  Expected layout: data/raw/%s/**/*.txt  (or *.json)\n"
            "  Please download the RACE dataset and place it there.",
            root_dir, subdir_name,
        )
        sys.exit(1)

    log.info("Scanning %s ...", root_dir)
    all_rows: list[dict] = []

    # rglob picks up both flat and nested directory structures
    json_files = sorted(f for f in root_dir.rglob("*") if f.is_file())

    if not json_files:
        log.error("No files found under %s. Aborting.", root_dir)
        sys.exit(1)

    for filepath in json_files:
        rows = _parse_race_file(filepath, split_name)
        all_rows.extend(rows)

    if not all_rows:
        log.error("Parsed 0 rows from %s. Check file format.", root_dir)
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=["id", "article", "question",
                                          "A", "B", "C", "D", "answer"])
    log.info(
        "  %s -> %d files, %d question-rows  (answer coverage: %d / %d)",
        split_name,
        len(json_files),
        len(df),
        df["answer"].notna().sum(),
        len(df),
    )
    return df


# ===========================================================================
# Stage 2 — Persist CSVs
# ===========================================================================

def save_dataframes(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Write three DataFrames to data/processed/ as UTF-8 CSVs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_path = CSV_PATHS[split_name]
        df.to_csv(out_path, index=False, encoding="utf-8")
        log.info("Saved %s -> %s  (%d rows)", split_name, out_path, len(df))


# ===========================================================================
# Stage 3 — Feature engineering
# ===========================================================================

def build_corpus(df: pd.DataFrame) -> pd.Series:
    """
    Combine article (repeated), question, and all four options into a
    single string per row.

    Formula (TF-IDF Manual section 6 — common-mistake table):
        combined = article + article + question + A + B + C + D

    Repeating the article gives passage content more influence over IDF
    weights than the shorter option strings.
    """
    article  = df["article"].fillna("").astype(str)
    question = df["question"].fillna("").astype(str)
    a = df["A"].fillna("").astype(str)
    b = df["B"].fillna("").astype(str)
    c = df["C"].fillna("").astype(str)
    d = df["D"].fillna("").astype(str)

    return (
        article + " " + article + " "
        + question + " "
        + a + " " + b + " " + c + " " + d
    )


def extract_labels(df: pd.DataFrame, split_name: str) -> Optional[np.ndarray]:
    """Convert A/B/C/D answer strings to integer labels 0-3."""
    if "answer" not in df.columns or df["answer"].isna().all():
        log.warning(
            "Split '%s' has no answer labels — skipping label extraction.",
            split_name,
        )
        return None

    labels = df["answer"].map(ANSWER_MAP)

    unmapped = df.loc[labels.isna(), "answer"].dropna().unique()
    if len(unmapped):
        log.warning("Unmapped answer values in '%s': %s", split_name, unmapped)

    return labels.to_numpy()


def vectorise(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """
    Fit a TfidfVectorizer on the training corpus only, then transform val
    and test corpora with the already-fitted object.

    Returns (X_train, X_val, X_test, fitted_vectorizer).

    DATA-LEAKAGE RULE
    -----------------
    fit_transform()  is called EXACTLY ONCE, on train_corpus.
    transform()      is called on val and test — it uses the vocabulary and
                     IDF weights computed from training data only, so no
                     information from unseen splits contaminates the features.
    """
    log.info("Building text corpora ...")
    train_corpus = build_corpus(train_df)
    val_corpus   = build_corpus(val_df)
    test_corpus  = build_corpus(test_df)

    log.info("  Train: %d documents", len(train_corpus))
    log.info("  Val  : %d documents", len(val_corpus))
    log.info("  Test : %d documents", len(test_corpus))

    log.info("Initialising TfidfVectorizer  params=%s", TFIDF_PARAMS)
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

    # ── FIT on training data only ──────────────────────────────────────────
    log.info("fit_transform() on training corpus ...")
    X_train: sp.csr_matrix = vectorizer.fit_transform(train_corpus)

    # ── TRANSFORM val & test — NO refitting ───────────────────────────────
    log.info("transform() on validation corpus (no refitting) ...")
    X_val:  sp.csr_matrix = vectorizer.transform(val_corpus)

    log.info("transform() on test corpus (no refitting) ...")
    X_test: sp.csr_matrix = vectorizer.transform(test_corpus)

    log.info(
        "Vocabulary size : %d  (max_features=%d)",
        len(vectorizer.vocabulary_),
        TFIDF_PARAMS["max_features"],
    )
    log.info(
        "Matrix shapes   : train=%s  val=%s  test=%s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    return X_train, X_val, X_test, vectorizer


# ===========================================================================
# Stage 4 — Persist artefacts
# ===========================================================================

def save_artefacts(
    vectorizer: TfidfVectorizer,
    X_train: sp.csr_matrix,
    X_val:   sp.csr_matrix,
    X_test:  sp.csr_matrix,
    y_train: Optional[np.ndarray],
    y_val:   Optional[np.ndarray],
    y_test:  Optional[np.ndarray],
) -> None:
    """
    Persist the vectorizer and all feature matrices / label arrays.

    Matrices are kept as scipy sparse objects.  Converting to dense with
    .toarray() on the full RACE dataset requires ~10 GB RAM — a common
    mistake flagged in the TF-IDF Manual section 6.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Fitted vectorizer — mandatory for inference
    joblib.dump(vectorizer, VECTORIZER_PATH)
    log.info("Vectorizer saved -> %s", VECTORIZER_PATH)

    # Sparse feature matrices
    for name, mat in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
        out = PROCESSED_DIR / f"{name}.npz"
        sp.save_npz(str(out), mat)
        log.info("Saved %s -> %s", name, out)

    # Integer labels
    for name, arr in (("y_train", y_train), ("y_val", y_val), ("y_test", y_test)):
        if arr is not None:
            out = PROCESSED_DIR / f"{name}.npy"
            np.save(str(out), arr)
            log.info("Saved %s -> %s  (shape %s)", name, out, arr.shape)


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_preprocessing() -> None:
    log.info("=" * 65)
    log.info("RACE Preprocessing Pipeline  (AL2002 Lab Project)")
    log.info("=" * 65)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 : Parse raw directories ──────────────────────────────────
    log.info("[Stage 1] Parsing raw RACE directories ...")
    train_df = parse_race_directory("train")
    val_df   = parse_race_directory("val")
    test_df  = parse_race_directory("test")

    # ── Stage 2 : Save CSVs ───────────────────────────────────────────────
    log.info("[Stage 2] Saving DataFrames as CSVs ...")
    save_dataframes(train_df, val_df, test_df)

    # ── Stage 3 : TF-IDF vectorisation ────────────────────────────────────
    log.info("[Stage 3] TF-IDF vectorisation ...")
    X_train, X_val, X_test, vectorizer = vectorise(train_df, val_df, test_df)

    y_train = extract_labels(train_df, "train")
    y_val   = extract_labels(val_df,   "val")
    y_test  = extract_labels(test_df,  "test")

    # ── Stage 4 : Persist artefacts ───────────────────────────────────────
    log.info("[Stage 4] Saving artefacts ...")
    save_artefacts(vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)

    log.info("=" * 65)
    log.info("Preprocessing complete.  All artefacts written to:")
    log.info("  CSVs       -> %s/", PROCESSED_DIR)
    log.info("  Vectorizer -> %s",  VECTORIZER_PATH)
    log.info("=" * 65)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()