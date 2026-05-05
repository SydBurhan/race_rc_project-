"""
model_a_unsupervised.py  —  RACE Reading Comprehension System  (AL2002 Lab Project)
====================================================================================
Model A: Unsupervised Component — K-Means Clustering on TF-IDF Features

Why unsupervised clustering on a reading-comprehension dataset?
---------------------------------------------------------------
The RACE dataset has 4 options per question.  Without access to labels, we ask:
"Do the TF-IDF representations of (article + question + option) naturally cluster
into groups that correlate with correctness?"

A perfect clustering would recover the binary correct / wrong distinction.
In practice, K-Means discovers latent topical or syntactic patterns that we
evaluate with:
    1. Silhouette Score   — geometric quality of clusters (range -1 → +1)
    2. Purity Score       — alignment of clusters with ground-truth labels
    3. Comparison table   — unsupervised results vs. Logistic Regression baseline

Pipeline
--------
1.  Load data/processed/train.csv  +  data/processed/val.csv
2.  Load models/model_a/tfidf_vectorizer.joblib  (fitted on train, never refit)
3.  Expand each question row into 4 option rows (same as Model A supervised)
4.  Build TF-IDF feature matrices via .transform() only
5.  Run K-Means with four different n_clusters  (2, 4, 8, 16)
    — n_clusters=4 is the focal experiment (4 MC options)
    — the sweep gives the report a richer picture
6.  Compute Silhouette Score and Purity Score per configuration
7.  Print the final comparison table (unsupervised vs. LR baseline)
8.  Save cluster assignments for downstream analysis
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=UserWarning)

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
OUTPUT_DIR      = Path("models") / "model_a" / "unsupervised"

TRAIN_CSV       = PROCESSED_DIR / "train.csv"
VAL_CSV         = PROCESSED_DIR / "val.csv"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

OPTION_COLS: list[str] = ["A", "B", "C", "D"]
ANSWER_MAP:  dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}

# ---------------------------------------------------------------------------
# Supervised baseline (from model_a_train.py run — fixed constants for report)
# ---------------------------------------------------------------------------
LR_BASELINE = {
    "model":     "Logistic Regression (Model A)",
    "accuracy":  0.5203,
    "macro_f1":  0.4779,
    "note":      "Option-level binary classification on val set",
}

# ---------------------------------------------------------------------------
# K-Means sweep configuration
# ---------------------------------------------------------------------------
KMEANS_CONFIGS: list[dict] = [
    {"n_clusters": 2,  "label": "K=2  (binary: correct / wrong)"},
    {"n_clusters": 4,  "label": "K=4  (one cluster per MC option) ← focal"},
    {"n_clusters": 8,  "label": "K=8  (mid-range)"},
    {"n_clusters": 16, "label": "K=16 (fine-grained)"},
]

RANDOM_STATE = 42

# Silhouette is expensive on 15 000-dim sparse data.
# We use TruncatedSVD to project to 100 dims for scoring only.
SVD_COMPONENTS = 100

# MiniBatchKMeans is used when the dataset exceeds this threshold (rows)
MINIBATCH_THRESHOLD = 20_000


# ===========================================================================
# 1. Data loading
# ===========================================================================

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error("File not found: %s", path)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"article", "question", "answer"} | set(OPTION_COLS)
    missing  = required - set(df.columns)
    if missing:
        log.error("Missing columns in %s: %s", path, missing)
        sys.exit(1)
    df = df.dropna(subset=["article", "question", "answer"]).reset_index(drop=True)
    log.info("Loaded %s  (%d rows)", path, len(df))
    return df


def load_vectorizer(path: Path):
    if not path.exists():
        log.error(
            "Vectorizer not found: %s\n  Run preprocessing.py first.", path
        )
        sys.exit(1)
    vec = joblib.load(path)
    log.info("Loaded vectorizer  (vocab: %d)", len(vec.vocabulary_))
    return vec


# ===========================================================================
# 2. Feature engineering — identical to model_a_train.py
# ===========================================================================

def expand_to_option_rows(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Expand each question row into 4 option rows with binary is_correct label."""
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
                "is_correct":    1 if letter == answer_letter else 0,
            })
    expanded = pd.DataFrame(records)
    log.info(
        "  %s: %d question-rows → %d option-rows  (positive rate: %.1f %%)",
        split_name, len(df), len(expanded),
        100.0 * expanded["is_correct"].mean(),
    )
    return expanded


def build_tfidf_matrix(
    df_expanded: pd.DataFrame,
    vectorizer,
    split_name: str,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build TF-IDF feature matrix and extract binary labels.

    Corpus formula (TF-IDF Manual §6):
        combined = article + article + question + option

    CRITICAL: .transform() only — never fit_transform().
    """
    corpus = (
        df_expanded["article"] + " "
        + df_expanded["article"] + " "
        + df_expanded["question"] + " "
        + df_expanded["option"]
    )
    X: sp.csr_matrix = vectorizer.transform(corpus)
    y: np.ndarray    = df_expanded["is_correct"].to_numpy(dtype=np.int32)

    log.info("  %s feature matrix: %s  positives: %d/%d",
             split_name, X.shape, y.sum(), len(y))
    return X, y


# ===========================================================================
# 3. Dimensionality reduction for Silhouette scoring
# ===========================================================================

def reduce_for_scoring(
    X_train: sp.csr_matrix,
    X_val:   sp.csr_matrix,
    n_components: int = SVD_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project sparse TF-IDF matrices to a dense low-dimensional space using
    Truncated SVD (LSA — Latent Semantic Analysis).

    Why SVD for Silhouette?
    -----------------------
    sklearn's silhouette_score computes pairwise Euclidean distances.  On a
    15 000-dim sparse matrix this is both slow and misleading (curse of
    dimensionality).  Projecting to 100 SVD components preserves the dominant
    variance while making distance metrics meaningful — standard practice in
    information retrieval (Manning et al., 2008).

    The SVD is fit on TRAINING data only, then applied to val — same
    no-leakage discipline as the TF-IDF vectorizer.
    """
    log.info("Fitting TruncatedSVD  (n_components=%d)  on training data …",
             n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd   = svd.transform(X_val)

    # L2-normalise rows so K-Means uses cosine-like geometry
    X_train_svd = normalize(X_train_svd, norm="l2")
    X_val_svd   = normalize(X_val_svd,   norm="l2")

    explained = svd.explained_variance_ratio_.sum()
    log.info("  SVD explains %.1f %% of variance in %d components",
             100 * explained, n_components)

    return X_train_svd, X_val_svd


# ===========================================================================
# 4. Purity score helper
# ===========================================================================

def purity_score(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Compute clustering purity.

    For each cluster we find the majority true class; purity is the fraction
    of samples that belong to that majority class across all clusters.

        Purity = (1/N) Σ_k  max_j |C_k ∩ L_j|

    where C_k is the set of samples in cluster k and L_j is the set of
    samples with true label j.

    Range: [0, 1].  Higher is better.  A trivial upper bound of 1.0 is
    achieved by assigning each sample its own cluster, so this metric must
    be read alongside Silhouette and cluster count.
    """
    n_samples  = len(y_true)
    clusters   = np.unique(cluster_labels)
    total_hits = 0

    for cluster_id in clusters:
        mask          = cluster_labels == cluster_id
        cluster_true  = y_true[mask]
        if len(cluster_true) == 0:
            continue
        # Majority vote: most frequent true label in this cluster
        majority_count = np.bincount(cluster_true).max()
        total_hits    += majority_count

    return total_hits / n_samples


# ===========================================================================
# 5. K-Means fitting and evaluation
# ===========================================================================

def fit_kmeans(
    X_fit: np.ndarray,          # dense SVD-reduced matrix to fit on
    n_clusters: int,
    n_rows: int,
) -> KMeans | MiniBatchKMeans:
    """
    Fit K-Means.  Uses MiniBatchKMeans for large datasets (>MINIBATCH_THRESHOLD
    rows) to keep runtime manageable without sacrificing cluster quality.
    """
    if n_rows > MINIBATCH_THRESHOLD:
        log.info("  Using MiniBatchKMeans (n_rows=%d > threshold)", n_rows)
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            batch_size=4096,
            max_iter=300,
            n_init=10,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            max_iter=300,
            n_init=10,
        )

    model.fit(X_fit)
    return model


def evaluate_kmeans(
    model,
    X_val_svd:   np.ndarray,
    y_val:       np.ndarray,
    config:      dict,
) -> dict:
    """
    Predict cluster labels on val set, compute Silhouette and Purity.

    Returns a result dict ready for the comparison table.
    """
    n_clusters  = config["n_clusters"]
    label_str   = config["label"]

    log.info("Evaluating  %s …", label_str)
    cluster_labels = model.predict(X_val_svd)

    # ── Silhouette Score ────────────────────────────────────────────────
    # Sample 5 000 rows max for speed (silhouette is O(n^2) in memory)
    n_sil    = min(5_000, len(y_val))
    sil_idx  = np.random.default_rng(RANDOM_STATE).choice(
        len(y_val), size=n_sil, replace=False
    )
    sil_score = silhouette_score(
        X_val_svd[sil_idx],
        cluster_labels[sil_idx],
        metric="euclidean",
        sample_size=None,
    )

    # ── Purity Score ────────────────────────────────────────────────────
    pur_score = purity_score(y_val, cluster_labels)

    log.info(
        "  n_clusters=%-3d  Silhouette=%.4f  Purity=%.4f",
        n_clusters, sil_score, pur_score,
    )

    return {
        "model":       f"K-Means (K={n_clusters})",
        "n_clusters":  n_clusters,
        "silhouette":  round(sil_score, 4),
        "purity":      round(pur_score, 4),
        # Accuracy and Macro F1 are not applicable for unsupervised models
        "accuracy":    "N/A",
        "macro_f1":    "N/A",
        "note":        label_str,
        "cluster_labels": cluster_labels,
    }


# ===========================================================================
# 6. Cluster analysis — what does each cluster contain?
# ===========================================================================

def analyse_clusters(
    df_val_exp:    pd.DataFrame,
    cluster_labels: np.ndarray,
    y_val:          np.ndarray,
    n_clusters:     int,
) -> None:
    """
    For the focal K=4 model, print a per-cluster breakdown showing:
        - cluster size
        - fraction of correct answers (is_correct=1)
        - most common option letters (A/B/C/D)

    This tells us whether the clusters correspond to semantic groups or
    merely to option-position artefacts.
    """
    log.info("")
    log.info("── Cluster Analysis  (K=%d, val set) ──────────────────────────", n_clusters)

    df_analysis = df_val_exp.copy()
    df_analysis["cluster"] = cluster_labels

    for cid in range(n_clusters):
        mask    = df_analysis["cluster"] == cid
        subset  = df_analysis[mask]
        n_total = len(subset)
        if n_total == 0:
            log.info("  Cluster %d: empty", cid)
            continue

        n_correct    = subset["is_correct"].sum()
        pct_correct  = 100.0 * n_correct / n_total
        letter_dist  = subset["option_letter"].value_counts().to_dict()

        log.info(
            "  Cluster %d  │ size=%-5d │ correct=%d (%.1f %%)  │ "
            "letter dist: A=%d B=%d C=%d D=%d",
            cid, n_total, n_correct, pct_correct,
            letter_dist.get("A", 0), letter_dist.get("B", 0),
            letter_dist.get("C", 0), letter_dist.get("D", 0),
        )
    log.info("")


# ===========================================================================
# 7. Comparison table printer
# ===========================================================================

def print_comparison_table(results: list[dict]) -> None:
    """
    Print a clean terminal table comparing all unsupervised configurations
    against the Logistic Regression baseline.  Ready to copy into the report.
    """
    W = 90
    SEP = "─" * W

    print()
    print("=" * W)
    print("  MODEL A — UNSUPERVISED vs. SUPERVISED COMPARISON  (Validation Set)")
    print("=" * W)
    print(
        f"  {'Model':<35}  {'Silhouette':>10}  {'Purity':>8}  "
        f"{'Accuracy':>10}  {'Macro F1':>10}"
    )
    print(SEP)

    # Supervised baseline row
    lr = LR_BASELINE
    print(
        f"  {lr['model']:<35}  {'—':>10}  {'—':>8}  "
        f"{lr['accuracy']:>10.4f}  {lr['macro_f1']:>10.4f}   ← supervised baseline"
    )
    print(SEP)

    # Unsupervised rows
    for r in results:
        focal = "  ← FOCAL" if r["n_clusters"] == 4 else ""
        print(
            f"  {r['model']:<35}  {r['silhouette']:>10.4f}  {r['purity']:>8.4f}  "
            f"{'N/A':>10}  {'N/A':>10}{focal}"
        )

    print(SEP)
    print()
    print("  Metric notes:")
    print("    Silhouette  : [-1, +1].  >0 means samples are closer to their own")
    print("                  cluster than to neighbours.  0 = overlapping clusters.")
    print("    Purity      : [0, 1].  Fraction of samples matching the majority")
    print("                  true label in their cluster.  Trivially 1.0 at K=N.")
    print("    Accuracy /  : Supervised option-level binary classification metrics.")
    print("    Macro F1      Not applicable to unsupervised models.")
    print()
    print("  Interpretation for final report:")
    print("    • K-Means purity > 0.75 means clusters align well with correctness.")
    print("    • A negative or near-zero silhouette indicates the TF-IDF space does")
    print("      not form naturally separated clusters at this granularity.")
    print("    • The gap between purity and LR accuracy quantifies the value of")
    print("      supervision for this task.")
    print("=" * W)
    print()


# ===========================================================================
# 8. Save artefacts
# ===========================================================================

def save_results(results: list[dict], df_val_exp: pd.DataFrame) -> None:
    """
    Save cluster assignment CSVs and a summary TSV to models/model_a/unsupervised/.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary_rows = []
    for r in results:
        summary_rows.append({
            "model":       r["model"],
            "n_clusters":  r["n_clusters"],
            "silhouette":  r["silhouette"],
            "purity":      r["purity"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "kmeans_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Saved summary → %s", summary_path)

    # Cluster assignments for each K configuration
    for r in results:
        k = r["n_clusters"]
        assign_df = df_val_exp[["id", "option_letter", "is_correct"]].copy()
        assign_df["cluster"] = r["cluster_labels"]
        out_path = OUTPUT_DIR / f"val_clusters_k{k}.csv"
        assign_df.to_csv(out_path, index=False)
        log.info("  Cluster assignments (K=%d) → %s", k, out_path)


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_unsupervised() -> None:
    log.info("=" * 65)
    log.info("Model A — Unsupervised Component (K-Means Clustering)")
    log.info("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────
    log.info("[Step 1] Loading data and vectorizer …")
    train_df   = load_csv(TRAIN_CSV)
    val_df     = load_csv(VAL_CSV)
    vectorizer = load_vectorizer(VECTORIZER_PATH)

    # ── 2. Feature engineering ────────────────────────────────────────────
    log.info("[Step 2] Expanding rows and building TF-IDF matrices …")
    log.info("  train:")
    train_exp       = expand_to_option_rows(train_df, "train")
    log.info("  val:")
    val_exp         = expand_to_option_rows(val_df, "val")

    X_train, y_train = build_tfidf_matrix(train_exp, vectorizer, "train")
    X_val,   y_val   = build_tfidf_matrix(val_exp,   vectorizer, "val")

    # ── 3. Dimensionality reduction for evaluation ────────────────────────
    log.info("[Step 3] Reducing to %d SVD components for scoring …",
             SVD_COMPONENTS)
    X_train_svd, X_val_svd = reduce_for_scoring(X_train, X_val)

    # ── 4. K-Means sweep ──────────────────────────────────────────────────
    log.info("[Step 4] Fitting K-Means models …")
    log.info("  (K-Means is fit on TRAINING features — val is unseen during fit)")

    all_results: list[dict] = []

    for config in KMEANS_CONFIGS:
        k = config["n_clusters"]
        log.info("  Fitting  %s …", config["label"])

        model  = fit_kmeans(X_train_svd, n_clusters=k, n_rows=len(train_exp))
        result = evaluate_kmeans(model, X_val_svd, y_val, config)
        all_results.append(result)

        # Full per-cluster analysis only for the focal K=4 model
        if k == 4:
            analyse_clusters(val_exp, result["cluster_labels"], y_val, k)

        # Save model
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / f"kmeans_k{k}.joblib"
        joblib.dump(model, model_path)
        log.info("  Saved K-Means (K=%d) → %s", k, model_path)

    # ── 5. Comparison table ───────────────────────────────────────────────
    log.info("[Step 5] Printing comparison table …")
    print_comparison_table(all_results)

    # ── 6. Save artefacts ─────────────────────────────────────────────────
    log.info("[Step 6] Saving artefacts …")
    save_results(all_results, val_exp)

    log.info("=" * 65)
    log.info("model_a_unsupervised.py complete.")
    log.info("  Results     → %s/", OUTPUT_DIR)
    log.info("  Summary CSV → %s", OUTPUT_DIR / "kmeans_summary.csv")
    log.info("=" * 65)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_unsupervised()