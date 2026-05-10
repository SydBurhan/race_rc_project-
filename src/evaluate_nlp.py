"""
src/evaluate_nlp.py
===================
NLP evaluation — BLEU / ROUGE-L / METEOR for Model A (questions) and
Model B (distractors + hints).

CLI:
    python src/evaluate_nlp.py                  # full corpus run on test split
    python src/evaluate_nlp.py --max 500        # limit to first 500 rows
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics_test.json"

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
for _pkg in ("wordnet", "punkt", "punkt_tab", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        try:
            nltk.data.find(f"tokenizers/{_pkg}")
        except LookupError:
            try:
                nltk.download(_pkg, quiet=True)
            except Exception:
                pass

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_smoother = SmoothingFunction().method1


# ══════════════════════════════════════════════════════════════════════════════
#  Core metric functions
# ══════════════════════════════════════════════════════════════════════════════

def _tok(s: str) -> list[str]:
    try:
        return nltk.word_tokenize(str(s).lower())
    except LookupError:
        return str(s).lower().split()


def compute_bleu(hyps: list[str], refs: list[str]) -> dict:
    th = [_tok(h) for h in hyps]
    tr = [[_tok(r)] for r in refs]
    out = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        out[f"bleu_{n}"] = round(
            corpus_bleu(tr, th, weights=weights, smoothing_function=_smoother), 4
        )
    return out


def compute_rouge_l(hyps: list[str], refs: list[str]) -> dict:
    p, r, f = [], [], []
    for h, ref in zip(hyps, refs):
        s = _rouge.score(str(ref), str(h))["rougeL"]
        p.append(s.precision); r.append(s.recall); f.append(s.fmeasure)
    n = max(len(hyps), 1)
    return {
        "rouge_l_precision": round(sum(p) / n, 4),
        "rouge_l_recall":    round(sum(r) / n, 4),
        "rouge_l_f1":        round(sum(f) / n, 4),
    }


def compute_meteor(hyps: list[str], refs: list[str]) -> dict:
    scores = []
    for h, ref in zip(hyps, refs):
        scores.append(meteor_score([_tok(ref)], _tok(h)))
    return {"meteor": round(sum(scores) / max(len(scores), 1), 4)}


def evaluate_generation(hyps: list[str], refs: list[str], verbose=True) -> dict:
    assert len(hyps) == len(refs), f"size mismatch: {len(hyps)} vs {len(refs)}"
    out: dict = {}
    out.update(compute_bleu(hyps, refs))
    out.update(compute_rouge_l(hyps, refs))
    out.update(compute_meteor(hyps, refs))
    out["num_samples"] = len(hyps)
    if verbose:
        _print_table("Generation Metrics", out)
    return out


def score_single(hyp: str, ref: str) -> dict:
    return evaluate_generation([hyp], [ref], verbose=False)


def _print_table(title: str, d: dict) -> None:
    print(f"\n{'─' * 50}\n  {title}\n{'─' * 50}")
    for k in ("bleu_1", "bleu_2", "bleu_3", "bleu_4",
              "rouge_l_precision", "rouge_l_recall", "rouge_l_f1", "meteor",
              "num_samples"):
        if k in d:
            v = d[k]
            print(f"  {k:<22} : {v}")
    print("─" * 50)


def save_metrics_json(metrics: dict, path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics saved -> %s", path)


# ══════════════════════════════════════════════════════════════════════════════
#  Corpus-level evaluation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    model_a_predictions: list[str],
    model_b_distractor_predictions: list[list[str]],
    model_b_hint_predictions: list[str],
    references_a: list[str],
    references_b_dist: list[list[str]],
    references_b_hint: list[str],
) -> dict:
    log.info("=" * 60)
    log.info("Running full NLP evaluation")
    log.info("=" * 60)

    out: dict = {}

    # Model A — questions
    out["model_a"] = evaluate_generation(model_a_predictions, references_a)

    # Model B — distractors: flatten each row's 3 generated distractors against
    # the 3 ground-truth wrong options (positional pairing).
    flat_h, flat_r = [], []
    for hyps3, refs3 in zip(model_b_distractor_predictions, references_b_dist):
        for i in range(min(3, len(hyps3), len(refs3))):
            flat_h.append(str(hyps3[i]))
            flat_r.append(str(refs3[i]))
    if flat_h:
        out["model_b_distractors"] = evaluate_generation(flat_h, flat_r)
    else:
        out["model_b_distractors"] = {"num_samples": 0}

    # Model B — hints (top-1 hint per sample vs gold answer sentence)
    if model_b_hint_predictions:
        out["model_b_hints"] = evaluate_generation(
            model_b_hint_predictions, references_b_hint
        )
    else:
        out["model_b_hints"] = {"num_samples": 0}

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main: run inference on test split and save metrics
# ══════════════════════════════════════════════════════════════════════════════

def _gold_answer_sentence(article: str, answer_text: str) -> str:
    """Pick the article sentence that best contains the answer text."""
    import re as _re
    sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", str(article)) if s.strip()]
    ans_lc = str(answer_text).lower()
    for s in sents:
        if ans_lc and ans_lc in s.lower():
            return s
    return sents[0] if sents else ""


def main(max_samples: int = 500) -> None:
    test_csv = PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        log.error("Test CSV not found: %s. Run preprocessing.py first.", test_csv)
        sys.exit(1)
    df = pd.read_csv(test_csv).head(max_samples).reset_index(drop=True)
    log.info("Evaluating on %d test samples", len(df))

    # Local import to avoid circular dependency at module load
    from src.inference import (
        generate_question, generate_distractors, generate_hints,
    )

    OPTIONS = ["A", "B", "C", "D"]
    model_a_preds, refs_a = [], []
    model_b_dist_preds, refs_b_dist = [], []
    model_b_hint_preds, refs_b_hint = [], []

    for i, row in df.iterrows():
        article = str(row.get("article", "")).strip()
        question_real = str(row.get("question", "")).strip()
        correct_label = str(row.get("answer", "A")).strip().upper()
        answer_text = str(row.get(correct_label, "")).strip()
        gt_distractors = [str(row.get(o, "")) for o in OPTIONS if o != correct_label]
        if not article or not answer_text:
            continue

        try:
            gen_q = generate_question(article, answer_text)
            gen_d = generate_distractors(article, gen_q, answer_text)
            gen_h = generate_hints(article, gen_q, answer_text)
        except Exception as e:
            log.warning("row %d failed: %s", i, e)
            continue

        model_a_preds.append(gen_q)
        refs_a.append(question_real)

        model_b_dist_preds.append(gen_d)
        refs_b_dist.append(gt_distractors)

        model_b_hint_preds.append(gen_h[0] if gen_h else "")
        refs_b_hint.append(_gold_answer_sentence(article, answer_text))

        if (i + 1) % 50 == 0:
            log.info("  processed %d / %d", i + 1, len(df))

    metrics = run_full_evaluation(
        model_a_predictions=model_a_preds,
        model_b_distractor_predictions=model_b_dist_preds,
        model_b_hint_predictions=model_b_hint_preds,
        references_a=refs_a,
        references_b_dist=refs_b_dist,
        references_b_hint=refs_b_hint,
    )
    save_metrics_json(metrics, METRICS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=500,
                        help="Max test rows to evaluate.")
    args = parser.parse_args()
    main(max_samples=args.max)
