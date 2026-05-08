"""
src/evaluate_nlp.py
===================
NLP Evaluation Module — BLEU · ROUGE-L · METEOR
================================================
Per professor's override:
  "Since your model is performing a text generation task, the appropriate
   evaluation metrics are BLEU, ROUGE, and METEOR.
   Do NOT use Accuracy or Precision."

Libraries used:
  • nltk        — BLEU (corpus_bleu) and METEOR (meteor_score)
  • rouge-score — ROUGE-L (RougeScorer)

Evaluation target:
  • Model A outputs (generated questions) are scored against the
    reference questions from the RACE dataset test split.

Usage (CLI):
    python src/evaluate_nlp.py --predictions preds.txt --references refs.txt

Usage (Python import):
    from src.evaluate_nlp import evaluate_generation, score_single
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# ── NLTK data bootstrap ────────────────────────────────────────────────────────
_NLTK_PACKAGES = ["wordnet", "punkt", "punkt_tab", "omw-1.4"]

def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                logger.info(f"[evaluate_nlp] Downloading NLTK package: {pkg}")
                nltk.download(pkg, quiet=True)

_ensure_nltk_data()

# ── ROUGE scorer (reusable instance) ──────────────────────────────────────────
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# ── Smoothing function for BLEU (avoids zero scores on short sequences) ────────
_smoother = SmoothingFunction().method1


# ══════════════════════════════════════════════════════════════════════════════
#  Core scoring functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_bleu(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4.

    Args:
        hypotheses: List of generated strings (one per sample).
        references: List of reference strings (one per sample, same order).

    Returns:
        {"bleu_1": float, "bleu_2": float, "bleu_3": float, "bleu_4": float}
    """
    tokenised_hyps = [nltk.word_tokenize(h.lower()) for h in hypotheses]
    tokenised_refs = [[nltk.word_tokenize(r.lower())] for r in references]

    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        bleu_scores[f"bleu_{n}"] = round(
            corpus_bleu(tokenised_refs, tokenised_hyps,
                        weights=weights,
                        smoothing_function=_smoother),
            4,
        )
    return bleu_scores


def compute_rouge_l(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute macro-averaged ROUGE-L precision, recall, and F1.

    Args:
        hypotheses: Generated strings.
        references: Reference strings.

    Returns:
        {"rouge_l_precision": float, "rouge_l_recall": float, "rouge_l_f1": float}
    """
    p_scores, r_scores, f_scores = [], [], []

    for hyp, ref in zip(hypotheses, references):
        scores = _rouge.score(ref, hyp)
        p_scores.append(scores["rougeL"].precision)
        r_scores.append(scores["rougeL"].recall)
        f_scores.append(scores["rougeL"].fmeasure)

    n = max(len(hypotheses), 1)
    return {
        "rouge_l_precision": round(sum(p_scores) / n, 4),
        "rouge_l_recall":    round(sum(r_scores) / n, 4),
        "rouge_l_f1":        round(sum(f_scores) / n, 4),
    }


def compute_meteor(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute macro-averaged METEOR score.

    NLTK's meteor_score takes tokenised lists.

    Args:
        hypotheses: Generated strings.
        references: Reference strings.

    Returns:
        {"meteor": float}
    """
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tok = nltk.word_tokenize(hyp.lower())
        ref_tok = nltk.word_tokenize(ref.lower())
        # meteor_score expects (list_of_references, hypothesis) — both tokenised
        scores.append(meteor_score([ref_tok], hyp_tok))

    avg = round(sum(scores) / max(len(scores), 1), 4)
    return {"meteor": avg}


# ══════════════════════════════════════════════════════════════════════════════
#  High-level evaluation entry points
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_generation(
    hypotheses: list[str],
    references: list[str],
    verbose: bool = True,
) -> dict[str, float]:
    """
    Run all three metrics (BLEU, ROUGE-L, METEOR) on a batch of predictions.

    Args:
        hypotheses: Model A generated questions.
        references: Corresponding RACE reference questions.
        verbose   : If True, pretty-print results to stdout.

    Returns:
        Flat dict with all metric values, e.g.:
        {
          "bleu_1": 0.42, "bleu_2": 0.31, "bleu_3": 0.22, "bleu_4": 0.16,
          "rouge_l_precision": 0.38, "rouge_l_recall": 0.41, "rouge_l_f1": 0.39,
          "meteor": 0.34,
          "num_samples": 100
        }
    """
    assert len(hypotheses) == len(references), (
        f"Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references."
    )

    results: dict[str, float] = {}
    results.update(compute_bleu(hypotheses, references))
    results.update(compute_rouge_l(hypotheses, references))
    results.update(compute_meteor(hypotheses, references))
    results["num_samples"] = len(hypotheses)

    if verbose:
        _pretty_print(results)

    return results


def score_single(hypothesis: str, reference: str) -> dict[str, float]:
    """
    Score a *single* hypothesis / reference pair.
    Useful for real-time per-inference scoring in the Streamlit UI.

    Returns:
        {"bleu_1", "bleu_2", "bleu_3", "bleu_4",
         "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
         "meteor"}
    """
    return evaluate_generation([hypothesis], [reference], verbose=False)


# ══════════════════════════════════════════════════════════════════════════════
#  RACE dataset evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_race(
    generator,                   # FlanT5Generator instance
    race_samples: list[dict],    # List of {"article", "question"} dicts
    max_samples: Optional[int] = 100,
    output_json: Optional[str]  = None,
) -> dict[str, float]:
    """
    Run Model A over `max_samples` RACE examples and compute all NLP metrics.

    Args:
        generator   : Instantiated FlanT5Generator.
        race_samples: List of dicts with at least {"article", "question"} keys.
        max_samples : Cap on number of samples to evaluate (None = all).
        output_json : If provided, save per-sample results to this path.

    Returns:
        Aggregate metrics dict from evaluate_generation().
    """
    samples = race_samples[:max_samples] if max_samples else race_samples
    hypotheses, references, per_sample = [], [], []

    logger.info(f"[evaluate_nlp] Evaluating on {len(samples)} RACE samples …")

    for i, sample in enumerate(samples):
        passage   = sample["article"]
        reference = sample["question"]

        result    = generator.run_pipeline(passage)
        hypothesis = result["question"]

        hypotheses.append(hypothesis)
        references.append(reference)

        per_sample.append({
            "index":      i,
            "passage":    passage[:200] + "…",
            "reference":  reference,
            "hypothesis": hypothesis,
            "latency":    result["latency_sec"],
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  … {i + 1}/{len(samples)} done")

    metrics = evaluate_generation(hypotheses, references, verbose=True)

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "samples": per_sample}, f, indent=2)
        logger.info(f"[evaluate_nlp] Results saved → {output_json}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty printer
# ══════════════════════════════════════════════════════════════════════════════

def _pretty_print(metrics: dict) -> None:
    divider = "─" * 42
    print(f"\n{divider}")
    print("  Model A — NLP Evaluation Results")
    print(divider)
    print(f"  Samples evaluated : {metrics.get('num_samples', '—')}")
    print(divider)
    print(f"  BLEU-1  : {metrics.get('bleu_1',  0):.4f}")
    print(f"  BLEU-2  : {metrics.get('bleu_2',  0):.4f}")
    print(f"  BLEU-3  : {metrics.get('bleu_3',  0):.4f}")
    print(f"  BLEU-4  : {metrics.get('bleu_4',  0):.4f}")
    print(divider)
    print(f"  ROUGE-L P : {metrics.get('rouge_l_precision', 0):.4f}")
    print(f"  ROUGE-L R : {metrics.get('rouge_l_recall',    0):.4f}")
    print(f"  ROUGE-L F1: {metrics.get('rouge_l_f1',        0):.4f}")
    print(divider)
    print(f"  METEOR  : {metrics.get('meteor', 0):.4f}")
    print(divider + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate generated questions with BLEU, ROUGE-L, METEOR."
    )
    p.add_argument(
        "--predictions", required=True,
        help="Path to a text file with one generated question per line."
    )
    p.add_argument(
        "--references", required=True,
        help="Path to a text file with one reference question per line."
    )
    p.add_argument(
        "--output-json", default=None,
        help="Optional path to save results as JSON."
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        hyps = [line.strip() for line in f if line.strip()]
    with open(args.references, encoding="utf-8") as f:
        refs = [line.strip() for line in f if line.strip()]

    metrics = evaluate_generation(hyps, refs, verbose=True)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved → {args.output_json}")
