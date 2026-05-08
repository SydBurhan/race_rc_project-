"""
src/template_generator.py
=========================
Classical ML Template-Based Question Generation (§4.2.3)
Uses One-Hot Encoding overlap for sentence extraction, rule-based 
templates for question formation, and an SVM for ranking.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Simple rule-based Wh-templates
TEMPLATES = [
    ("What is the significance of {answer}?", "what"),
    ("Who is associated with {answer}?", "who"),
    ("Where did {answer} take place?", "where"),
    ("When did {answer} occur?", "when"),
    ("Why is {answer} mentioned in the passage?", "why")
]

def _split_sentences(text: str) -> list[str]:
    """Simple regex-based sentence splitter."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def extract_candidate_sentences(article: str, answer: str, top_k: int = 3) -> list[str]:
    """
    Step 1: Extract candidate sentences using One-Hot Encoding (CountVectorizer)
    overlap with the correct answer.
    """
    sentences = _split_sentences(article)
    if not sentences:
        return []

    # One-Hot Encoding via CountVectorizer (binary=True)
    vectorizer = CountVectorizer(binary=True, stop_words='english')
    
    try:
        # Fit on sentences + answer to ensure shared vocabulary
        vocab_corpus = sentences + [answer]
        vectorizer.fit(vocab_corpus)
        
        sent_vectors = vectorizer.transform(sentences).toarray()
        answer_vector = vectorizer.transform([answer]).toarray()[0]
        
        # Calculate overlap (dot product of one-hot vectors)
        overlaps = np.dot(sent_vectors, answer_vector)
        
        # Get top-k indices
        top_indices = overlaps.argsort()[-top_k:][::-1]
        return [sentences[i] for i in top_indices if overlaps[i] > 0]
        
    except ValueError:
        return sentences[:top_k] # Fallback if vocabulary is empty

def generate_template_questions(article: str, answer: str) -> list[dict]:
    """
    Step 2: Apply Wh-word templates to transform candidate sentences.
    """
    candidates = extract_candidate_sentences(article, answer)
    generated = []
    
    for sent in candidates:
        for template, wh_type in TEMPLATES:
            # Basic heuristic: only use 'Who' if answer is capitalized (name proxy)
            if wh_type == "who" and not answer[0].isupper():
                continue
            
            question_text = template.format(answer=answer)
            generated.append({
                "question": question_text,
                "source_sentence": sent
            })
            
    return generated

def rank_questions(generated_questions: list[dict], ranker_model=None) -> list[dict]:
    """
    Step 3: Rank generated questions. 
    (Uses a placeholder heuristic if the trained SVM isn't passed).
    """
    if not generated_questions:
        return []
        
    if ranker_model is not None:
        # If you pass the SVM from model_a_train, it scores them here
        pass 
    else:
        # Fallback scoring: penalize overly long or short questions
        for q in generated_questions:
            length = len(q["question"].split())
            score = 1.0 - abs(10 - length) * 0.05 
            q["score"] = max(0.1, score)
            
    return sorted(generated_questions, key=lambda x: x.get("score", 0), reverse=True)

# --- Quick Test ---
if __name__ == "__main__":
    passage = "Albert Einstein developed the theory of relativity. He was a theoretical physicist born in Germany."
    ans = "Albert Einstein"
    
    print(f"Passage: {passage}\nAnswer: {ans}\n")
    questions = generate_template_questions(passage, ans)
    ranked = rank_questions(questions)
    
    for i, q in enumerate(ranked[:3], 1):
        print(f"Rank {i}: {q['question']}")