# Classical NLP: Sentiment Classification and Vector-Space Similarity

Two NLP projects built from first principles with NLTK and scikit-learn: no pre-trained embeddings,
every feature engineered and evaluated by hand.

## 1. Tweet sentiment classification

**Notebook:** [`01_tweet_sentiment.ipynb`](01_tweet_sentiment.ipynb)

Binary sentiment classification of ~27,000 labelled tweets with a linear SVM.

- Pre-processing: lower-casing, removal of non-alphabetic characters, whitespace tokenisation,
  lemmatisation, and experiments with stop-word removal and trigrams.
- Features: unigram + bigram bag-of-words, fed to a scikit-learn `LinearSVC` pipeline.
- Evaluation: 10-fold cross-validation with precision, recall, F1 and accuracy per fold.
- Error analysis: false positives and false negatives are exported and inspected to drive the next
  round of pre-processing and feature changes.

| Setup | Result |
|---|---|
| 10-fold CV, best pipeline | **F1 0.898**, accuracy 0.866 |

## 2. Character similarity in TV scripts

**Notebook:** [`02_character_vector_similarity.ipynb`](02_character_vector_similarity.ipynb)

Each character in a soap-opera script becomes a document of their lines. The task is to build a
vector space where a character's held-out lines land closest to their own training vector, measured
by retrieval rank among 16 characters.

- Better pre-processing and linguistic features (POS tags, n-grams), TF-IDF weighting.
- Dialogue context and scene features: the surrounding lines in a scene are added to each line.
- Systematic parameter search over the feature and weighting choices.
- Analysis of which characters are confused with which, and why.

| Setup | Result |
|---|---|
| Baseline representation | 14 / 16 correct |
| Final system on the test set | **15 / 16 correct (93.75%)** |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook
```

The tweet and script datasets were provided for teaching and are **not included**. Place a
tab-separated sentiment file (label, text) and the script `training.csv`, `val.csv` and `test.csv`
next to the notebooks to re-run them.
