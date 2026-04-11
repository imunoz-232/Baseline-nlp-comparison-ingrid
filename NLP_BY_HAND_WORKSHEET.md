# NLP By-Hand Worksheet (Printable)

Name: ____________________    Date: ____________________

Use this worksheet with WORKED_NLP_MATH_EXAMPLE.md.

## Tiny corpus

D1: I love NLP, NLP is fun.
D2: I love pizza and coding.
D3: Coding in Python is fun.

Cleaning rules:
- lowercase
- remove punctuation
- remove stopwords: i, is, and, in

## Part A) Cleaning

1. Clean D1:

______________________________________________________________________________

2. Clean D2:

______________________________________________________________________________

3. Clean D3:

______________________________________________________________________________

## Part B) Vocabulary and index map

Write sorted vocabulary:

[__________________, __________________, __________________, __________________,
 __________________, __________________]

Index mapping:

- __________________ -> 0
- __________________ -> 1
- __________________ -> 2
- __________________ -> 3
- __________________ -> 4
- __________________ -> 5

## Part C) Bag-of-Words matrix

Fill counts in vocabulary order.

- D1: [____, ____, ____, ____, ____, ____]
- D2: [____, ____, ____, ____, ____, ____]
- D3: [____, ____, ____, ____, ____, ____]

## Part D) TF-IDF setup

Use:
- N = 3
- idf(t) = ln((N + 1) / (df(t) + 1)) + 1

1. Fill df values:

- df(coding) = ____
- df(fun) = ____
- df(love) = ____
- df(nlp) = ____
- df(pizza) = ____
- df(python) = ____

2. Compute idf for:

- terms with df=2: __________________
- terms with df=1: __________________

3. Compute two tf-idf values for D1:

- tfidf(nlp, D1) = __________________
- tfidf(fun, D1) = __________________

## Part E) Retrieval by cosine similarity

Query: love fun

Use BoW vectors for this part.

1. Query vector q in vocabulary order:

q = [____, ____, ____, ____, ____, ____]

2. Compute cosine(q, D1), cosine(q, D2), cosine(q, D3):

- cosine(q, D1) = __________________
- cosine(q, D2) = __________________
- cosine(q, D3) = __________________

3. Rank documents from most similar to least similar:

1) __________________
2) __________________
3) __________________

## Part F) One classifier score

Given toy linear model for class "tech":

- z = w^T x + b
- w = [0.2, 1.0, 0.1, 0.8, -0.9, 0.7]
- b = -0.2
- Use D3 vector for x

1. Compute z:

z = __________________

2. Optional probability (sigmoid):

p = 1 / (1 + exp(-z)) = __________________

## Part G) One KMeans cycle

Given k = 2 with initial centroids at D1 and D2:

1. Compute distances from D3:

- ||D3 - D1||^2 = __________________
- ||D3 - D2||^2 = __________________

2. Which cluster does D3 join?

____________________

3. Updated centroid for cluster containing D2 and D3:

mu2 = __________________

## Reflection (2-3 sentences each)

1. Why can cleaning rules change model performance?

______________________________________________________________________________

2. Why is retrieval a ranking task instead of a classification task?

______________________________________________________________________________

3. Why are clustering metrics not directly comparable to classification accuracy?

______________________________________________________________________________
