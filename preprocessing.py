#23i0757 23i0541 Syed Burhan Ahmad + Mushahid Hussain Project AI, Machine Learning

# This will be the main preproccesor for data, as files are large and may be redundant, what to do?
# 1) Parse RACE dir, one quesrtiom, one row 
# 2) Save files to test train val, what aboyt tts? id->art->Q->ABDC-> answer
# 3) Vectoirzation (corpus), 
# 4) Persist artefacts, all data that has been processed ok
# Once done, model A can be trained :)

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
#Logging details
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
#File dir struct
RAW_DIR       = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
MODEL_DIR     = Path("models") / "model_a"

RAW_SUBDIRS: dict[str, str] = {
    "train": "train",
    "val":   "dev",    # is this split?
    "test":  "test",
}

CSV_PATHS: dict[str, Path] = {
    split: PROCESSED_DIR / f"{split}.csv"
    for split in ("train", "val", "test")
}

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
#Project specific
TFIDF_PARAMS: dict = dict(
    stop_words="english",
    sublinear_tf=True,
    max_features=15_000,
)

#Map answers sequentially
ANSWER_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}

#1)
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

        # exactly 4 options ABCD
        while len(opts) < 4:
            opts.append("")
        opts = opts[:4]

        answer = answers[q_idx] if q_idx < len(answers) else None

        row = {
            # primary key-ish, uniqueness 
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
    # Add all files into a single DataFrame, be it high medium
    
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


 #2) 
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

#3)
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
   
    log.info("Building text corpora ...")
    train_corpus = build_corpus(train_df)
    val_corpus   = build_corpus(val_df)
    test_corpus  = build_corpus(test_df)

    log.info("  Train: %d documents", len(train_corpus))
    log.info("  Val  : %d documents", len(val_corpus))
    log.info("  Test : %d documents", len(test_corpus))

    log.info("Initialising TfidfVectorizer  params=%s", TFIDF_PARAMS)
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

    # FIT on training data only 
    log.info("fit_transform() on training corpus ...")
    X_train: sp.csr_matrix = vectorizer.fit_transform(train_corpus)

    # TRANSFORM val & test  NO refitting 
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


#4)
def save_artefacts(
    vectorizer: TfidfVectorizer,
    X_train: sp.csr_matrix,
    X_val:   sp.csr_matrix,
    X_test:  sp.csr_matrix,
    y_train: Optional[np.ndarray],
    y_val:   Optional[np.ndarray],
    y_test:  Optional[np.ndarray],
) -> None:
   
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    
    joblib.dump(vectorizer, VECTORIZER_PATH)
    log.info("Vectorizer saved -> %s", VECTORIZER_PATH)

    for name, mat in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
        out = PROCESSED_DIR / f"{name}.npz"
        sp.save_npz(str(out), mat)
        log.info("Saved %s -> %s", name, out)

    for name, arr in (("y_train", y_train), ("y_val", y_val), ("y_test", y_test)):
        if arr is not None:
            out = PROCESSED_DIR / f"{name}.npy"
            np.save(str(out), arr)
            log.info("Saved %s -> %s  (shape %s)", name, out, arr.shape)



#Running pipeline
def run_preprocessing() -> None:
    log.info("=" * 65)
    log.info("RACE Preprocessing Pipeline  (AL2002 Lab Project)")
    log.info("=" * 65)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    
    log.info("[Stage 1] Parsing raw RACE directories ...")
    train_df = parse_race_directory("train")
    val_df   = parse_race_directory("val")
    test_df  = parse_race_directory("test")

    
    log.info("[Stage 2] Saving DataFrames as CSVs ...")
    save_dataframes(train_df, val_df, test_df)

    
    log.info("[Stage 3] TF-IDF vectorisation ...")
    X_train, X_val, X_test, vectorizer = vectorise(train_df, val_df, test_df)

    y_train = extract_labels(train_df, "train")
    y_val   = extract_labels(val_df,   "val")
    y_test  = extract_labels(test_df,  "test")

   
    log.info("[Stage 4] Saving artefacts ...")
    save_artefacts(vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)

    log.info("=" * 65)
    log.info("Preprocessing complete.  All artefacts written to:")
    log.info("  CSVs       -> %s/", PROCESSED_DIR)
    log.info("  Vectorizer -> %s",  VECTORIZER_PATH)
    log.info("=" * 65)



if __name__ == "__main__":
    run_preprocessing()