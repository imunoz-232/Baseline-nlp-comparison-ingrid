# NLP By-Hand Worksheet Answer Key

This key corresponds to NLP_BY_HAND_WORKSHEET.md.

## Part A) Cleaning

- D1 -> love nlp nlp fun
- D2 -> love pizza coding
- D3 -> coding python fun

## Part B) Vocabulary and index map

Sorted vocabulary:
[coding, fun, love, nlp, pizza, python]

- coding -> 0
- fun -> 1
- love -> 2
- nlp -> 3
- pizza -> 4
- python -> 5

## Part C) Bag-of-Words matrix

- D1: [0, 1, 1, 2, 0, 0]
- D2: [1, 0, 1, 0, 1, 0]
- D3: [1, 1, 0, 0, 0, 1]

## Part D) TF-IDF setup

N = 3, idf(t) = ln((N + 1) / (df(t) + 1)) + 1

df values:
- df(coding) = 2
- df(fun) = 2
- df(love) = 2
- df(nlp) = 1
- df(pizza) = 1
- df(python) = 1

idf values:
- df=2 -> ln(4/3)+1 = 1.2877
- df=1 -> ln(4/2)+1 = 1.6931

For D1 (length 4):
- tf(nlp, D1)=2/4=0.5 -> tfidf=0.5*1.6931=0.8466
- tf(fun, D1)=1/4=0.25 -> tfidf=0.25*1.2877=0.3219

## Part E) Retrieval by cosine similarity

Query q = [0, 1, 1, 0, 0, 0]

- cosine(q, D1)=0.577
- cosine(q, D2)=0.408
- cosine(q, D3)=0.408

Ranking: D1 first, then D2 and D3 (tie).

## Part F) One classifier score

w=[0.2,1.0,0.1,0.8,-0.9,0.7], b=-0.2, x=D3=[1,1,0,0,0,1]

z = 0.2(1)+1.0(1)+0.1(0)+0.8(0)-0.9(0)+0.7(1)-0.2 = 1.7

sigmoid probability:
p = 1/(1+exp(-1.7)) = 0.8455

## Part G) One KMeans cycle

- ||D3-D1||^2 = 7
- ||D3-D2||^2 = 4

D3 joins D2 cluster.

Updated centroid:
mu2 = (D2 + D3)/2 = [1, 0.5, 0.5, 0, 0.5, 0.5]

## Reflection guidance

1. Cleaning changes token inventory and feature sparsity, which changes model input geometry.
2. Retrieval optimizes rank order of similar items, not a discrete class label.
3. Clustering metrics measure structure quality without labels or with optional external label alignment, unlike supervised accuracy.
