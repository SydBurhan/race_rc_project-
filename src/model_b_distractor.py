"""
src/model_b_distractor.py
=========================
Wrapper for Model B — Distractor & Hint Generation
Connects the classical ML logic to the Streamlit UI.

Fix applied (Action 2)
----------------------
b_metrics previously returned hardcoded constants (0.85 / 0.78 / 0.81).
Now calls evaluate_row() from model_b_evaluate.py at inference time
so Screen 4 displays real per-request Precision / Recall / F1.

The pipeline accepts an optional `ground_truth_options` dict so that
when a RACE sample is loaded the true incorrect options are available
for scoring. When the user pastes custom text (no ground truth), the
metrics fall back to graceful N/A zeros rather than crashing.
"""

from src.model_b_train import generate_distractors, generate_hints, load_vectorizer
from src.model_b_evaluate import evaluate_row


class ModelBPipeline:
    def __init__(self):
        """Initialise the pipeline by loading the pre-fitted TF-IDF vectorizer."""
        self.vectorizer = load_vectorizer()

    def run_pipeline(
        self,
        passage: str,
        question: str,
        answer: str,
        ground_truth_options: dict | None = None,
    ) -> dict:
        """
        Runs the full Model B pipeline:
        1. Generates 3 distractors using MMR and TF-IDF similarity.
        2. Generates 3 extractive hints based on question relevance.
        3. Evaluates generated distractors against ground-truth RACE options
           (when available) to produce live Precision / Recall / F1.

        Parameters
        ----------
        passage              : reading passage text
        question             : generated or RACE question string
        answer               : correct answer text (not label — the actual string)
        ground_truth_options : optional dict {"A": "...", "B": "...", "C": "...", "D": "..."}
                               from the RACE sample. Used to extract the three
                               ground-truth incorrect options for scoring.
                               Pass None when no reference is available.

        Returns
        -------
        dict with keys:
            distractors : list[str]        — 3 generated distractor strings
            hints       : list[str]        — 3 extractive hint strings
            b_metrics   : dict             — precision, recall, f1, has_reference
        """

        # ── 1. Generate Distractors ──────────────────────────────────────────
        dist_results = generate_distractors(
            article=passage,
            question=question,
            correct_answer=answer,
            vectorizer=self.vectorizer,
        )
        distractors = [d["distractor"] for d in dist_results]

        # ── 2. Generate Hints ────────────────────────────────────────────────
        hint_results = generate_hints(
            article=passage,
            question=question,
            vectorizer=self.vectorizer,
        )
        hints = [h["sentence"] for h in hint_results]

        # ── 3. Live evaluation against ground-truth options ──────────────────
        b_metrics = _compute_live_metrics(distractors, answer, ground_truth_options)

        return {
            "distractors": distractors,
            "hints":       hints,
            "b_metrics":   b_metrics,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_live_metrics(
    distractors: list[str],
    correct_answer: str,
    ground_truth_options: dict | None,
) -> dict:
    """
    Compute Precision / Recall / F1 for the generated distractors.

    Ground-truth distractor set = all options whose text != correct_answer.
    Returns has_reference=False and zero scores when no ground truth is given.
    """
    if not ground_truth_options:
        # No RACE reference available (user pasted custom text)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "has_reference": False}

    # Extract the three incorrect options as ground-truth distractors
    gt_distractors = [
        text for text in ground_truth_options.values()
        if text.strip().lower() != correct_answer.strip().lower()
    ]

    if not gt_distractors:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "has_reference": False}

    counts = evaluate_row(distractors, gt_distractors)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "f1":            round(f1,        4),
        "tp":            tp,
        "fp":            fp,
        "fn":            fn,
        "has_reference": True,
    }