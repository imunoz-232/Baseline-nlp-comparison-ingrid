# NLP Route Architecture Guide

## Why this guide exists
This project includes four practical NLP routes that use vectorized text features.
The routes solve different business problems and use different learning assumptions.

## Shared foundation: vectorization layer
Before any route, text is converted into numeric vectors.

### Bag of Words (CountVectorizer)
- Build a vocabulary of unique tokens.
- Represent each document as a sparse count vector.
- If vocabulary size is V, each document is a vector in R^V.

### TF-IDF (TfidfVectorizer)
- Starts from token counts.
- Downweights very frequent terms and upweights distinctive terms.
- Common formula intuition:
  - tf(term, doc): term frequency in document
  - idf(term): larger for rarer terms across corpus
  - tfidf = tf * idf

## Route 1: Classification (supervised)
Goal: Predict known labels for new documents.

### When to use
- You have labeled examples.
- You need explicit predictions, such as topic, spam, or sentiment.

### Mathematical architecture
For Logistic Regression with multiclass output:
- score per class c: z_c = w_c^T x + b_c
- probability: p(y=c|x) = exp(z_c) / sum_j exp(z_j)
- train by minimizing cross-entropy + regularization.

### Main tuning controls
- ngram_range: unigram vs bigram richness
- min_df / max_features: vocabulary size and noise
- C: regularization strength in Logistic Regression

### Common failure modes
- Overfitting with too many sparse features
- Label leakage from metadata
- Underfitting if vocabulary is too constrained

## Route 2: Clustering (unsupervised)
Goal: Discover structure without labels.

### When to use
- No labels exist yet.
- You need exploratory grouping of documents.

### Mathematical architecture (KMeans)
- Assign each document x_i to nearest centroid mu_k.
- Update each centroid to mean of assigned points.
- Minimize: sum_i ||x_i - mu_{z_i}||^2
- Iterate until assignments stabilize or max iterations reached.

### Main tuning controls
- k (number of clusters)
- vectorizer features and min_df
- initialization and n_init

### Common failure modes
- Wrong k causes topic mixing or fragmentation
- Sparse high-dimensional space can create weak separation

## Route 3: Retrieval/Search (ranking)
Goal: Return top-k similar documents for a query.

### When to use
- Need search results, not class labels.
- Want related-document recommendations.

### Mathematical architecture
- Build vector for query q and document d.
- Compute cosine similarity:
  - cosine(q,d) = (q dot d) / (||q|| ||d||)
- Rank documents by similarity score.

### Main tuning controls
- Count vs TF-IDF vectors
- preprocessing consistency for query/doc
- k (top-k depth)

### Common failure modes
- Token mismatch between query and corpus
- Overly aggressive cleaning removes useful terms

## Route 4: Baseline comparison
Goal: Quickly identify strongest classical approach before deep learning.

### When to use
- Early project stage
- Need benchmark for future model upgrades

### Mathematical architecture
- Train multiple pipelines (vectorizer + model family).
- Evaluate all on same test split.
- Compare metrics (accuracy, macro F1, speed, simplicity).

### Main tuning controls
- model family (NB, Logistic Regression, Linear SVM)
- vectorizer settings
- regularization and smoothing parameters

## How to decide route in practice
- Need prediction and labels available: start with classification.
- Need structure but no labels: start with clustering.
- Need top related docs: start with retrieval.
- Need robust first benchmark: run baseline comparison.

## Hard-coding from scratch: what you would implement
1. Text cleaning and tokenization.
2. Vocabulary builder (token to index).
3. Sparse matrix creation for counts or tf-idf.
4. Model core math:
   - Logistic Regression gradient updates, or
   - KMeans assignment/update loop, or
   - Cosine similarity ranking.
5. Evaluation loops and hyperparameter sweeps.

## Practical tuning workflow for students
1. Lock one train/test split.
2. Tune preprocessing and vectorizer first.
3. Tune model hyperparameters second.
4. Keep an experiment log with settings and results.
5. Validate improvements with at least one secondary metric (for example macro F1).
