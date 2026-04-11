"""Search and retrieval route for 20 Newsgroups.

This module treats each document as searchable content and evaluates retrieval quality
with Precision@k based on topic-label matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class RetrievalResult:
    precision_at_k: float
    k: int


def run_retrieval_route(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    y_train: Sequence[int],
    y_test: Sequence[int],
    target_names: List[str],
    k: int = 5,
    sample_query_count: int = 5,
) -> RetrievalResult:
    """Run lexical retrieval and report Precision@k."""
    print("\n" + "=" * 80)
    print("ROUTE 3: SEARCH AND RETRIEVAL (TF-IDF + COSINE SIMILARITY)")
    print("=" * 80)
    print("Goal:")
    print("  Use a document as a query and retrieve top-k similar documents.")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=45000, min_df=2)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    sim_matrix = linear_kernel(x_test, x_train)

    per_query_precisions = []
    for q_idx in range(sim_matrix.shape[0]):
        ranked = np.argsort(-sim_matrix[q_idx])[:k]
        correct = sum(1 for idx in ranked if y_train[idx] == y_test[q_idx])
        per_query_precisions.append(correct / k)

    p_at_k = float(np.mean(per_query_precisions))

    print(f"\nEvaluation metric: Precision@{k} = {p_at_k:.4f}")
    print("How to interpret:")
    print(f"  - For each query, we inspect top {k} results.")
    print("  - Precision@k is the fraction of retrieved docs with matching topic labels.")

    print("\nSample query walkthrough:")
    limit = min(sample_query_count, len(test_texts))
    for q_idx in range(limit):
        ranked = np.argsort(-sim_matrix[q_idx])[:k]
        top_labels = [target_names[y_train[idx]] for idx in ranked]
        print(f"\n  Query {q_idx + 1}: true topic='{target_names[y_test[q_idx]]}'")
        print(f"  Top-{k} retrieved topics: {top_labels}")
        print(
            f"  Query Precision@{k}: "
            f"{sum(1 for idx in ranked if y_train[idx] == y_test[q_idx]) / k:.2f}"
        )

    return RetrievalResult(precision_at_k=p_at_k, k=k)
