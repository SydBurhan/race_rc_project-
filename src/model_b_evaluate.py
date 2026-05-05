"""
Model B — Distractor Generation: Evaluation Pipeline
FAST-NUCES AI Lab Project — RACE Dataset

Evaluates generated distractors against ground-truth RACE incorrect options
using a Unigram Token Overlap approach to compute Precision, Recall, and F1.

Usage:
    python model_b_evaluate.py
    python model_b_evaluate.py --rows 500    # evaluate on a subset
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/val.csv")
OVERLAP_THRESHOLD = 0.3   # IoU-style: fraction of generated tokens that must
                           # overlap with a ground-truth option to count as TP

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","have","has","had","that","this",
    "it","he","she","they","we","by","from","not","no","as","which","who",
}

# ══════════════════════════════════════════════════════════════════════════
# 1.  Tokenisation helper
# ══════════════════════════════════════════════════════════════════════════
def tokenise(text: str) -> set:
    """Lowercase, strip punctuation, remove stopwords, return token set."""
    tokens = re.findall(r"\b[a-z]{2,}\b", str(text).lower())
    return {t for t in tokens if t not in STOPWORDS}


# ══════════════════════════════════════════════════════════════════════════
# 2.  Distractor generation  (plug your real logic here)
# ══════════════════════════════════════════════════════════════════════════
def generate_distractors(article: str, question: str, answer: str,
                          n: int = 3) -> list[str]:
    """
    Placeholder generator — extracts the top-n content keywords from the
    article that do NOT appear in the correct answer.

    Replace the body of this function with your actual Model B logic.
    The return type must stay: list[str]  (n distractor strings).
    """
    article_tokens = re.findall(r"\b[a-z]{4,}\b", article.lower())
    answer_tokens  = tokenise(answer)

    freq = Counter(t for t in article_tokens
                   if t not in STOPWORDS and t not in answer_tokens)

    candidates = [word for word, _ in freq.most_common(20)]
    return candidates[:n]


# ══════════════════════════════════════════════════════════════════════════
# 3.  Single-row evaluation
# ══════════════════════════════════════════════════════════════════════════
def evaluate_row(generated: list[str], ground_truth: list[str]) -> dict:
    """
    For each generated distractor find the best-matching ground-truth option
    (highest unigram overlap).  A match is a True Positive if overlap >= threshold.

    Returns per-row TP, FP, FN counts.
    """
    gt_token_sets = [tokenise(g) for g in ground_truth]
    matched_gt    = set()          # indices of GT options already claimed
    tp = fp = 0

    for gen in generated:
        gen_tokens = tokenise(gen)
        if not gen_tokens:
            fp += 1
            continue

        best_overlap = 0.0
        best_idx     = -1
        for i, gt_tokens in enumerate(gt_token_sets):
            if i in matched_gt:
                continue
            union = gen_tokens | gt_tokens
            if not union:
                continue
            overlap = len(gen_tokens & gt_tokens) / len(union)  # Jaccard
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx     = i

        if best_overlap >= OVERLAP_THRESHOLD and best_idx != -1:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)          # GT options not matched
    return {"tp": tp, "fp": fp, "fn": fn}


# ══════════════════════════════════════════════════════════════════════════
# 4.  Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════
def main(max_rows: int | None = None):
    print("=" * 62)
    print("  Model B — Distractor Generation: Evaluation Report")
    print("=" * 62)

    # Load data
    df = pd.read_csv(DATA_PATH)
    if max_rows:
        df = df.head(max_rows)
    print(f"\n  Evaluating on {len(df):,} questions from {DATA_PATH}\n")

    OPTIONS = ["A", "B", "C", "D"]
    total_tp = total_fp = total_fn = 0
    row_f1s  = []

    for _, row in df.iterrows():
        correct  = str(row["answer"]).strip().upper()
        gt_distractors = [
            str(row[opt]) for opt in OPTIONS if opt != correct
        ]

        generated = generate_distractors(
            article=row["article"],
            question=row["question"],
            answer=row[correct] if correct in row else "",
        )

        counts = evaluate_row(generated, gt_distractors)
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        row_f1s.append(f1)

    # ── Aggregate metrics ──────────────────────────────────────────────────
    macro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    macro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0
    avg_f1   = float(np.mean(row_f1s))

    # ══════════════════════════════════════════════════════════════════════
    # 5.  Print Results
    # ══════════════════════════════════════════════════════════════════════

    # --- Confusion-Matrix style counts ---
    print("  AGGREGATE TOKEN-OVERLAP COUNTS")
    print("  " + "-" * 42)
    print(f"  {'True  Positives (TP)':<32} {total_tp:>7,}")
    print(f"  {'False Positives (FP)':<32} {total_fp:>7,}")
    print(f"  {'False Negatives (FN)':<32} {total_fn:>7,}")
    print()

    # --- Precision / Recall / F1 table ---
    print("  EVALUATION METRICS  (Unigram Overlap, threshold={:.0%})".format(
        OVERLAP_THRESHOLD))
    print("  " + "=" * 42)
    print(f"  {'Metric':<28} {'Score':>8}  {'Percentage':>10}")
    print("  " + "-" * 42)
    print(f"  {'Precision':<28} {macro_p:>8.4f}  {macro_p*100:>9.2f}%")
    print(f"  {'Recall':<28} {macro_r:>8.4f}  {macro_r*100:>9.2f}%")
    print(f"  {'F1-Score (micro)':<28} {macro_f1:>8.4f}  {macro_f1*100:>9.2f}%")
    print(f"  {'F1-Score (avg over rows)':<28} {avg_f1:>8.4f}  {avg_f1*100:>9.2f}%")
    print("  " + "=" * 42)
    print()

    # --- Pseudo Confusion Matrix (2×2 binary view) ---
    tn = 0   # TN is ill-defined for open-ended generation; shown as N/A
    print("  PSEUDO-CONFUSION MATRIX  (Generated vs Ground-Truth Distractors)")
    print("  " + "-" * 46)
    print(f"  {'':25} {'Pred: Match':>12}  {'Pred: No Match':>14}")
    print(f"  {'GT: Has matching option':25} {total_tp:>12,}  {'N/A (open-set)':>14}")
    print(f"  {'GT: No matching option':25} {total_fp:>12,}  {'N/A (open-set)':>14}")
    print("  " + "-" * 46)
    print()
    print("  NOTE: TN is undefined for open-ended generation tasks.")
    print("        FP = generated distractors with no good GT match.")
    print("        FN = GT distractors the model failed to cover.")
    print()
    print("  Screenshot the block above for your lab report.")
    print("=" * 62)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None,
                        help="Limit evaluation to first N rows (default: all)")
    args = parser.parse_args()
    main(max_rows=args.rows)
