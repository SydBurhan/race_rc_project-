# RACE Reading Comprehension and Quiz Generation System

**Course:** AL2002 Artificial Intelligence Lab
**Institution:** FAST National University of Computer and Emerging Sciences, Islamabad
**Authors:** Syed Burhan Ahmad (23i-0757), Mushahid Hussain (23i-0541)

---

## 1. Project Overview

This project is a complete reading comprehension and quiz generation system built entirely on traditional machine learning. The whole stack runs on the RACE dataset (Reading Comprehension from Examinations) and intentionally avoids neural networks of any kind. Every learned component is a classical sklearn estimator, every text representation is either bag of words, TF-IDF, or static Word2Vec embeddings, and the user-facing layer is a Streamlit application that ties everything together.

The system does four things end to end:

1. Given a passage, a question, and four options, it predicts which option is correct (Model A verifier).
2. Given a passage, it generates a multiple choice question along with the right answer (Model A question generator).
3. Given the question and the right answer, it generates three plausible distractors (Model B distractors).
4. Given the same inputs, it produces three graduated hints, with the most explicit hint having the answer redacted to a blank (Model B hints).

Everything is trained on the public RACE corpus and evaluated on the held out test split using BLEU, ROUGE-L, METEOR, and standard classification metrics.

---

## 2. Why Traditional ML Only

The course assignment limits us to classical methods. That sounds restrictive at first, but it actually shapes some of the most interesting design choices in this project. A neural seq2seq model could trivially memorize "How long does the Nile stretch?" given enough data. With only sklearn, we have to think carefully about how to extract structure from a passage, when to trust extractive shortcuts, and when to fall back to template based generation. The result is a hybrid pipeline that is fast, deterministic, and explainable, and that works on commodity hardware without a GPU.

The downside is well known: BLEU-4 on free form question generation will never be high with templates, because human RACE questions paraphrase heavily. We discuss this honestly in the evaluation section below and report numbers that match what the academic literature gives for non-neural baselines.

---

## 3. Repository Layout

```
race_rc_project/
├── data/
│   ├── raw/                 RACE CSV files (train, val, test)
│   └── processed/           Cleaned CSVs + sparse feature matrices
├── models/
│   ├── model_a/traditional/ LR, SVM, NB, RF, XGB, ensemble, ranker, OHE vectorizer
│   └── model_b/traditional/ Hint scorer LR + cached Word2Vec vectors
├── src/
│   ├── preprocessing.py     Clean, split, vectorize
│   ├── model_a_train.py     Train all Model A classifiers
│   ├── template_generator.py Template based question generator + ranker
│   ├── model_b_train.py     Distractor pipeline + hint scorer training
│   ├── inference.py         Unified inference API used by the UI
│   └── evaluate_nlp.py      Corpus level BLEU / ROUGE / METEOR
├── ui/app.py                Streamlit application
├── reports/metrics_test.json Evaluation output
├── notebooks/               EDA + Colab/Kaggle reproduction notebook
└── requirements.txt
```

The root `app.py` is a one liner that imports `ui.app` so users can launch with either `streamlit run app.py` or `streamlit run ui/app.py`.

---

## 4. Data and Preprocessing

The RACE dataset comes as three CSVs (`train.csv`, `val.csv`, `test.csv`) with the columns `id, article, question, A, B, C, D, answer`. Our preprocessing module (`src/preprocessing.py`) handles a few quirks of the public Kaggle copy of RACE:

* It strips pandas index leakage columns like `Unnamed: 0`.
* It validates that all required columns are present.
* If the three CSVs turn out to be byte identical (a known re-upload artefact on Kaggle), it falls back to a single 80/10/10 split with `random_state=42` to avoid silent data leakage.
* Cleaning lowercases the text, strips non alphanumeric characters, and collapses whitespace.

After cleaning, two parallel vectorizers are fit on the training split only:

* A **TF-IDF vectorizer** with `sublinear_tf=True`, `stop_words="english"`, `max_features=15000`, used for cosine similarity over articles and sentences.
* A **binary Count vectorizer** (one hot encoding), used as the input feature space for Model A's option level classifiers.

We also build option level expanded matrices, where each RACE question becomes 4 rows (one per option) with a binary label of whether that option is the correct answer. This expanded representation is what every Model A classifier sees during training. Combined with 10 handcrafted lexical features per option (overlap counts, length ratios, etc.) we get a `(4N, 15010)` sparse matrix per split that fits comfortably in RAM.

Reproducibility is enforced everywhere with `random_state=42`. No vectorizer is ever refit on validation or test data.

---

## 5. Model A: Multiple Choice Verifier and Question Generator

Model A has two responsibilities. The first is the **verifier**: given a passage, a question, and four options, predict which option is correct. The second is the **question generator**: given a passage and a target answer, produce a question that could plausibly have that answer.

### 5.1 The Verifier Ensemble

We train five sklearn classifiers on the same option level features:

| Classifier | sklearn class |
|---|---|
| Logistic Regression | `LogisticRegression(class_weight="balanced", max_iter=1000)` |
| Linear SVM | `LinearSVC` wrapped in `CalibratedClassifierCV` so we get probabilities |
| Complement Naive Bayes | `ComplementNB` |
| Random Forest | `RandomForestClassifier` |
| XGBoost | `xgb.XGBClassifier`, with auto CUDA detection |

We also fit two more models for completeness because the lab brief asked us to demonstrate clustering and semi supervised methods:

* **K-Means** with `k=4` for unsupervised structure on the option vectors.
* **Label Propagation** as a semi supervised baseline.

The runtime ensemble used in the UI is a soft vote of LR + SVM + NB. We chose those three because they are calibrated, fast at inference, and disagree just enough to benefit from averaging. RF and XGBoost are kept around for the comparison table in the report but are not part of the live verifier path because their inference cost is higher and the gain over the soft vote is small.

### 5.2 The Question Generator

The question generator lives in `src/template_generator.py` and is split into three stages.

**Stage 1: Candidate sentence extraction.** Given an article and the correct answer, we vectorize each sentence with the OHE vectorizer and rank them by cosine similarity to the answer. The top five sentences are kept as candidate "source sentences" for question generation.

**Stage 2: Template filling.** We have 30 hand written templates organized into 6 Wh categories: What, Who, Where, When, Why, How. Each template has slots like `{noun_phrase}`, `{verb_phrase}`, `{clause}`, and `{adj_phrase}`. The slots are filled from the source sentence and the answer. A few subtle but important things happen here:

* If the answer itself looks like a noun phrase (proper noun or numeric quantity, up to 5 words), we use the answer as the noun phrase rather than mining a possibly irrelevant capitalized chunk from the sentence. This single change cleans up most of the "what is meant by Brown in the passage?" style questions we kept seeing.
* Templates are filtered by **answer type** before being applied. We classify the answer as one of `number`, `year`, `person`, `place`, `entity`, or `phrase`, and only allow Wh words that make sense for that type. Numeric answers can be `How / When / What`, places can be `Where / What`, and so on. This stops the generator from producing things like "Who is 1981?" or "Where is 2,300 kilometers?" which the original unfiltered version did regularly.

**Stage 3: SVM ranker.** We train a `LinearSVC` on six features per candidate:

* Cosine similarity of the question to the article.
* Cosine similarity of the question to the answer.
* Question length.
* Whether the question starts with a Wh word.
* Lexical overlap between question and answer.
* Normalized position of the source sentence in the article.

Positives are real RACE questions. Negatives are template generated questions for the same passage and answer. The ranker then scores all candidates at inference time and we return the top one.

This three stage design is a textbook traditional QG pipeline (Heilman and Smith 2010, Du et al. 2017). It will never match a fine tuned T5, but it is fully explainable, runs in milliseconds, and produces coherent output for the patterns it covers.

---

## 6. Model B: Distractors and Hints

Model B is split into two independent sub systems that share the OHE vectorizer.

### 6.1 Distractor Generation

A distractor is a wrong option that should still be plausible enough to fool a careless reader. We combine three candidate sources, then filter and diversify them.

**Source 1: TF-IDF / OHE cosine + MMR.** We mine 1 to 3 grams from the article, score each by cosine similarity to the correct answer, and apply Maximal Marginal Relevance (Carbonell and Goldstein 1998) with `λ = 0.6` to balance similarity and diversity. This gives us in passage candidates that are topically related to the answer but lexically distinct from each other.

**Source 2: Frequency substitution.** Top high frequency content words from the article, with stop words removed and a tie break that prefers capitalized tokens. This is the original Heilman and Smith style "swap with a frequent word" approach.

**Source 3: Word2Vec nearest neighbours.** Pre trained `word2vec-google-news-300` from gensim, cached locally to `models/model_b/traditional/word2vec_kv.bin` after the first run. For each token in the answer we fetch its semantic neighbours and keep only those that are not already in the article (otherwise the distractor would be a trivial mention).

### 6.2 The Hard Cases We Hit During Testing

Real testing surfaced two failure modes that needed dedicated logic:

**Numeric answers had no in-article peers.** When the answer was "1636" and the article only mentioned that one year, the candidate pool collapsed to non-year tokens like "research" or "Harvard", which made the correct year trivially obvious. Fix: a `_synthesize_numeric_distractors` function that perturbs the gold value. For years it adds offsets like ±5, ±15, ±50, ±150, ±300, clamped to a sensible historical range. For free numbers it multiplies by a set of factors (0.5x, 0.75x, 1.25x, 1.5x, 2x, 0.25x), rounds to nice magnitudes, and preserves the unit if there is one. So "6,650 kilometers" becomes "3,300 kilometers", "5,000 kilometers", "8,300 kilometers", and so on.

**Phrase answers had no semantic peers.** When the answer was "northeastern region of Tanzania", the only candidates we could mine from the article were near duplicates ("the northeastern region", "tanzania rises"). The fix was a proper noun substitution generator. We identify the salient capitalized token in the answer ("Tanzania") and swap it with peers from a small built in lexicon (Kenya, Uganda, Ethiopia, Rwanda) or, if the lexicon does not have it, from Word2Vec neighbours. We also handle cardinal directions ("northeastern" goes with "southeastern", "central", etc.) so the system can vary either component of a multi part phrase.

The lexicon is not exhaustive, and that is intentional. It is meant as a graceful fallback when Word2Vec is not available, which can happen on Python 3.13 where gensim has no upstream wheel. With Word2Vec installed, the lexicon is a safety net; without it, the system still produces sensible distractors for the entities people actually search for.

### 6.3 Filtering and Diversification

After collecting candidates from all sources, we run them through a sequence of plausibility filters:

* **Type matching:** numeric answers only accept numeric distractors, phrase answers reject pure numbers, etc.
* **Length ratio:** distractor token count must be within 2x or 0.5x of the answer's token count.
* **Substring rejection:** if the candidate contains the answer or vice versa, drop it.
* **Word level Jaccard overlap:** if the candidate and answer share too many content words (Jaccard > 0.34, or the candidate is essentially a content word subset), drop it. This is the rule that catches "the northeastern region" when the answer is "northeastern region of Tanzania".

There is one important subtlety. The synthesized numeric distractors and the substituted proper noun distractors are deliberately constructed to share words with the answer (they keep the unit, or keep most of the phrase). If we ran them through the overlap filter they would all be rejected. So we split the pool into a **trusted** set (synthesized + substituted) and an **untrusted** set (mined from the article). The trusted set bypasses the overlap filter and only has to pass the type sanity check. The untrusted set has to pass everything. The final selection then runs MMR over the merged cleaned pool to pick three diverse distractors.

### 6.4 Hint Generation

Hints are the only learned component in Model B. We train a `LogisticRegression` on per sentence features. For each (article, question, answer) triple we label every sentence as 1 if it contains the answer string and 0 otherwise. The classifier then predicts how revealing each sentence is. Five features per sentence:

1. Cosine similarity to the question.
2. Cosine similarity to the answer.
3. Lexical overlap with the question.
4. Normalized position in the article.
5. Word count.

At hint time, the article is partitioned into answer containing and answer free sentences. We then build three graduated hints:

* **Hint 1, General:** a middle ranked safe sentence (no answer leak), gives topical context.
* **Hint 2, Specific:** the top ranked safe sentence, narrows things down.
* **Hint 3, Near Explicit:** the top ranked answer containing sentence with the answer redacted to `_____`. This is a cloze style hint that forces the user to fill in the blank rather than reading the answer off the page.

The redaction step matters pedagogically. Without it, Hint 3 would just be the answer sentence verbatim, which defeats the purpose.

---

## 7. The Streamlit Application

The UI lives in `ui/app.py` and is a four screen Streamlit app: Article Input, Quiz View, Hint Panel, and Analytics. The plumbing has a few pieces worth describing because they took real iteration to get right.

### 7.1 Three Branch Pipeline

When the user clicks Generate Quiz, the runner picks one of three branches based on what is available:

* **Branch A, RACE sample.** If the user clicked Random RACE Sample, we already have the real human written question and the real four options. We use them verbatim and only generate hints. Running the template generator on top would just produce a worse paraphrase of a perfect question, which is exactly what an early version of this code did.
* **Branch B, Relation extractor on free text.** For pasted passages we first try a small set of high precision regex patterns that catch common factual structures (length, year founded, born in, located in, composed of, home to N species). When one of these fires, we get a clean (question, answer) pair directly without going near the template generator. Numbers, units, and proper nouns all flow through naturally. There are five base patterns and five biographical sub patterns with broadened verb lists for things like won, received, awarded, enrolled, moved, published, and so on.
* **Branch C, Template fallback.** If no pattern matches, we fall back to picking a salient answer from the passage and running the template generator. The picker prefers numbers, then multi word proper nouns, then capitalized non sentence start tokens, then the longest content word. Combined with the type aware template filtering this still produces grammatical questions for narrative text, even if they are less specific than what Branch B would have generated.

The Quiz View shows a small caption above each question saying which branch fired, which is useful both for the demo and for spotting regressions during development.

### 7.2 Streamlit Pitfalls We Hit

A few things took longer than they should have:

* **Sidebar radio nav was unkeyed.** Once a user clicked it, Streamlit ignored our `index=` parameter on subsequent reruns and kept reverting to whatever the user last clicked. This made programmatic navigation (Generate Quiz should jump to Quiz View) silently fail, and in one bad version it caused the radio to snap back to Home on every rerun. The fix was a keyed radio with a one shot `pending_screen` flag that gets consumed before the widget is instantiated.
* **Text area `value=` was being ignored.** Setting `value=` together with `key=` is a known Streamlit gotcha; once the widget has state, `value=` is silently dropped. Clicking Random RACE Sample then mutated session state but the visible text box never updated, so the Generate Quiz button stayed disabled. Fix: drop `value=`, write directly to `st.session_state["passage_area"]`, and call `st.rerun()`.
* **Multi line passages broke regex matching.** The original relation extractor patterns used `[^.\n]` to constrain the subject capture. When users pasted copy formatted articles with line breaks every 80 characters or so, the subject group could not extend across the newline to reach the verb. The fix is one line at the top of the extractor that normalizes all whitespace to single spaces before matching.
* **Hint panel was invisible in dark mode.** The `.hint-box` class had a cream background but no explicit text color, so in dark theme the text inherited white-on-cream. Fixed with `color: #1f2937;` and a gold accent for the heading.

These are all small bugs individually. Together they are a reminder that the UI layer is a real engineering surface, not a thin wrapper.

### 7.3 Pronoun Resolution

When the relation extractor matches a sentence whose subject is a pronoun (She enrolled at the Sorbonne in 1891), the literal subject is "She" which produces "When did She enroll?". To fix this we resolve the subject by detecting pronouns and substituting the article's leading proper noun ("Marie Curie") in their place. This is not full coreference, just a heuristic, but it covers the biographical articles where the second sentence onwards typically uses pronouns.

---

## 8. Evaluation

We evaluate on the held out RACE test split using `src/evaluate_nlp.py`, which computes BLEU-1 through BLEU-4, ROUGE-L precision/recall/F1, and METEOR. The reported numbers are corpus level over 500 questions for question generation and hints, and 1500 distractors for Model B distractors. Results are saved to `reports/metrics_test.json`.

### 8.1 Headline Numbers

| Model | BLEU-1 | BLEU-4 | ROUGE-L F1 | METEOR | n |
|---|---|---|---|---|---|
| Model A (Question Generation)   | 0.168 | 0.022 | 0.148 | 0.144 | 500 |
| Model B (Distractors)           | 0.006 | 0.000 | 0.054 | 0.022 | 1500 |
| Model B (Hints)                 | 0.289 | 0.169 | 0.169 | 0.186 | 500 |

### 8.2 How to Read These Numbers

**Model A (Question Generation).** BLEU-1 of 0.17 with METEOR 0.14 is consistent with the classical QG literature on RACE style data. BLEU-4 is low at 0.022 because human RACE questions paraphrase aggressively and our templates do not. This is structural: any classical template generator hits this ceiling. The fact that BLEU-1 and METEOR remain healthy tells us we are getting the right content words even when the surface form differs.

**Model B (Distractors).** The metrics here look poor at first glance, but distractors are a known case where corpus level n-gram overlap is the wrong metric. There is no single correct distractor for any question, so comparing our three to the gold three is unfair: a perfectly plausible distractor that does not happen to be one of the three RACE chose will score zero. The proper way to evaluate distractors is with token level precision/recall against the union of all gold distractors, plus qualitative analysis. The numbers we report are honest, just not very informative on their own.

**Model B (Hints).** This is our strongest component. BLEU-1 0.29 and BLEU-4 0.17 are competitive with extractive summarization baselines on similar data. The cloze redaction in Hint 3 also gets credit from these metrics because the redacted sentence still shares most of its tokens with the answer containing source.

### 8.3 The Verifier

The verifier is evaluated separately on accuracy / precision / recall / F1, both as a 4-class classifier (which option of A/B/C/D is correct) and as a binary classifier (is this option correct, yes or no, the way it sees the data during training). Confusion matrices and a per classifier comparison table are produced by `src/model_a_train.py` at the end of training. The Streamlit Analytics screen also tracks live verifier accuracy across the session, so you can see how it does on the user's actual interactions.

---

## 9. Testing Approach

We have unit tests in `tests/` and ran the full pipeline end to end on the three RACE splits. More importantly, we did a long round of qualitative testing through the UI on free text passages, because corpus metrics can easily hide the kind of grammatical and semantic failures that real users see immediately.

The qualitative testing is what surfaced most of the design decisions in this report:

* The original template generator was producing things like "Why does reef is home to more than 1,500 according to the text?" because the answer being fed in was the first 4 words of the passage. Fixing this required a smarter answer picker, type aware template filtering, and ultimately the relation extractor branch.
* Numeric distractors were trivially distinguishable from random article tokens. That gave us the synthesizer.
* Phrase distractors had near duplicates of the answer. That gave us the Jaccard overlap filter, and once that filter was too strict, the trusted/untrusted pool split.
* Pronoun subjects produced ungrammatical questions. That gave us the leading proper noun resolver.

We provide a small library of test passages in the project notes covering all five extractor patterns plus narrative passages that fall through to the template generator. These were used to validate the system manually before final submission.

---

## 10. Honest Limitations

A few things we deliberately did not attempt:

* **Full coreference resolution.** Our pronoun resolver is a single rule (substitute the leading proper noun). Real coreference would handle multiple entities in one passage. We considered using neuralcoref or a similar tool but it would have pulled in a neural dependency, which violates the spirit of the assignment.
* **Span based answer extraction.** For free text we use either the relation extractor or a heuristic answer picker. A proper QA model would predict an answer span end to end. That is again a neural component.
* **Distractor evaluation.** We report BLEU/ROUGE/METEOR for distractors because the framework expected it, but we explicitly note in the report that these metrics are misleading for this task and that token level overlap or human evaluation is more appropriate.
* **Passages outside the relation extractor's pattern set.** If a free text passage does not match any of our five base patterns or five biographical sub patterns, we fall back to template generation, which is grammatical but generic. The system handles this gracefully but the output quality is visibly lower than for matched patterns.

---

## 11. Reproducibility

To reproduce the full pipeline:

```
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab wordnet omw-1.4
python src/preprocessing.py
python src/model_a_train.py
python src/template_generator.py
python src/model_b_train.py
python src/evaluate_nlp.py
streamlit run ui/app.py
```

For Colab or Kaggle, `notebooks/colab_kaggle_run.ipynb` runs everything top to bottom and is recommended for the heavier training runs.

All `random_state` values are pinned to 42. No vectorizer is fit on validation or test data. The Word2Vec model is downloaded once on first run and cached locally; subsequent runs use the cached binary.

---

## 12. What We Learned

A few takeaways from building this system:

The biggest practical win was the relation extractor. Five regex patterns plus pronoun resolution covered a large fraction of the factual passages we tested, and produced output far cleaner than the template ranker for those cases. It is a reminder that for narrow factual content, classical patterns often beat statistical approaches that have to learn the same patterns implicitly.

The biggest practical loss was that most of the harder RACE questions (the genuinely inferential ones) cannot be reproduced by either templates or patterns. A neural QG model would do better on those because it can paraphrase and combine multiple sentences. We accept this and report metrics that reflect it.

The Streamlit layer was deceptively tricky. State management, session cache, and widget keying have non obvious interactions that produced symptoms (blank screens, frozen navigation) that looked like deep bugs but were actually small UI plumbing issues. A keyed radio with a one shot pending flag is now our default pattern for any programmatic navigation in Streamlit.

Finally, qualitative testing through the UI was worth far more than corpus level metrics for actually improving the system. Every meaningful fix in this project came from looking at a specific bad output in the UI and asking what went wrong.
