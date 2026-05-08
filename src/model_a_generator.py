"""
src/model_a_generator.py
========================
Model A — Generative Question & Answer Generator
Tech Stack : HuggingFace Transformers (google/flan-t5-base)
Task       : Given a raw reading passage →
               1. Generate a reading-comprehension question.
               2. Extract the correct answer span from the passage.

Professor's constraint: "text generation task → evaluate with BLEU, ROUGE, METEOR".
This module therefore returns raw text outputs that evaluate.py will score.

NO scikit-learn, NO classical-ML code in this file.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-base"
MAX_PASSAGE_TOKENS = 512        # Flan-T5 context limit (safe ceiling)
MAX_QUESTION_TOKENS = 64        # Generated question length ceiling
MAX_ANSWER_TOKENS = 64          # Extracted answer length ceiling
GENERATION_BEAM_SIZE = 4        # Beam search width
NO_REPEAT_NGRAM = 3             # Avoid repetitive n-grams


# ── Model Loader (singleton pattern for Streamlit caching) ─────────────────────
class FlanT5Generator:
    """
    Wraps google/flan-t5-base for:
      • Question generation  (passage  → question)
      • Answer extraction    (passage + question → answer span)

    Designed to be loaded once and cached by Streamlit's @st.cache_resource.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"[Model A] Loading {model_name} on {device} …")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("[Model A] Model ready.")

    # ── Internal helper ────────────────────────────────────────────────────────
    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Tokenise prompt, run beam-search, decode output."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_PASSAGE_TOKENS,
            truncation=True,
            padding=False,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=GENERATION_BEAM_SIZE,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                early_stopping=True,
            )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded.strip()

    # ── Public API ─────────────────────────────────────────────────────────────
    def generate_question(self, passage: str) -> str:
        """
        Prompt Flan-T5 to write a reading-comprehension question for the passage.

        Prompt pattern (instruction-tuned style):
            "Generate a reading comprehension question for the following passage:
             <passage>"

        Returns:
            A question string (may or may not end with '?').
        """
        # Truncate passage to avoid overflowing context
        passage_snippet = _truncate_text(passage, max_words=300)

        prompt = (
            "Generate a reading comprehension question for the following passage:\n\n"
            f"{passage_snippet}"
        )
        question = self._generate(prompt, max_new_tokens=MAX_QUESTION_TOKENS)

        # Ensure it ends with a question mark
        if question and not question.endswith("?"):
            question = question + "?"

        return question

    def extract_answer(self, passage: str, question: str) -> str:
        """
        Prompt Flan-T5 to extract the correct answer span from the passage
        given the generated question.

        Prompt pattern:
            "Answer the following question based only on the passage below.
             Question: <question>
             Passage:  <passage>"

        Returns:
            The extracted answer string.
        """
        passage_snippet = _truncate_text(passage, max_words=300)

        prompt = (
            "Answer the following question based only on the passage below.\n\n"
            f"Question: {question}\n\n"
            f"Passage: {passage_snippet}"
        )
        answer = self._generate(prompt, max_new_tokens=MAX_ANSWER_TOKENS)
        return answer

    def run_pipeline(self, passage: str) -> dict:
        """
        Full Model-A inference pipeline.

        Args:
            passage: Raw reading passage text.

        Returns:
            dict with keys:
                question      : str   — generated question
                answer        : str   — extracted correct answer
                latency_sec   : float — wall-clock inference time
        """
        t0 = time.time()

        question = self.generate_question(passage)
        answer   = self.extract_answer(passage, question)

        latency = round(time.time() - t0, 3)

        return {
            "question":    question,
            "answer":      answer,
            "latency_sec": latency,
        }


# ── Utility ────────────────────────────────────────────────────────────────────
def _truncate_text(text: str, max_words: int) -> str:
    """Return at most `max_words` words from text (whole-word boundary)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def clean_passage(raw: str) -> str:
    """
    Light preprocessing before feeding into Model A.
    Keeps punctuation (needed for coherent generation).
    """
    # Collapse excess whitespace / newlines
    text = re.sub(r"\s+", " ", raw).strip()
    return text


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    SAMPLE_PASSAGE = (
        "The Amazon rainforest, often referred to as the 'lungs of the Earth', "
        "produces about 20% of the world's oxygen. It spans over 5.5 million "
        "square kilometres across nine countries in South America. Deforestation "
        "has threatened this ecosystem; scientists warn that losing more than 20–25% "
        "of the forest could trigger a tipping point, converting large areas into "
        "savannah and releasing vast amounts of stored carbon dioxide."
    )

    generator = FlanT5Generator()
    result = generator.run_pipeline(SAMPLE_PASSAGE)

    print("\n=== Model A — Flan-T5 Output ===")
    print(f"Question : {result['question']}")
    print(f"Answer   : {result['answer']}")
    print(f"Latency  : {result['latency_sec']} s")