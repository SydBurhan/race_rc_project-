"""Smoke tests for src.inference. Requires trained artefacts under models/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

ARTICLE = (
    "The Amazon rainforest, often referred to as the lungs of the Earth, "
    "produces about 20 percent of the world's oxygen. It spans more than "
    "five million square kilometres across nine countries in South America. "
    "Deforestation has threatened this ecosystem for decades."
)


def _models_present() -> bool:
    return (ROOT / "models" / "model_a" / "traditional" / "ohe_vectorizer.pkl").exists()


@pytest.mark.skipif(not _models_present(), reason="trained models not present")
def test_signature_and_io():
    from src.inference import (
        predict_answer, generate_question,
        generate_distractors, generate_hints,
    )

    q = generate_question(ARTICLE, "Amazon rainforest")
    assert isinstance(q, str) and len(q) > 0

    d = generate_distractors(ARTICLE, q, "Amazon rainforest")
    assert isinstance(d, list) and len(d) == 3

    h = generate_hints(ARTICLE, q, "Amazon rainforest")
    assert isinstance(h, list) and len(h) == 3

    options = {"A": "Amazon rainforest", "B": "Pacific Ocean",
               "C": "Sahara Desert",   "D": "Mount Everest"}
    label, conf = predict_answer(ARTICLE, q, options)
    assert label in options
    assert 0.0 <= conf <= 1.0


def test_empty_input_raises():
    from src.inference import generate_question
    with pytest.raises(ValueError):
        generate_question("", "anything")
