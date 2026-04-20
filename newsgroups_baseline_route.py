"""Baseline modeling route for 20 Newsgroups.

This module compares several classic NLP baselines quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class BaselineResult:
    scores: Dict[str, float]
    best_model_name: str
    best_model_accuracy: float
    best_model_macro_f1: float


def run_baseline_route(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    y_train: Sequence[int],
    y_test: Sequence[int],
) -> BaselineResult:
    """Compare baseline pipelines and report best model."""
    print("\n" + "=" * 80)
    print("ROUTE 4: BASELINE MODEL COMPARISON")
    print("=" * 80)
    print("Goal:")
    print("  Compare common classical NLP baselines before moving to deep learning.")

    pipelines = {
        "Count + MultinomialNB": Pipeline(
            [
                ("vec", CountVectorizer(stop_words="english", min_df=2)),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
        "Count + LogisticRegression": Pipeline(
            [
                ("vec", CountVectorizer(stop_words="english", min_df=2, max_features=30000)),
                ("clf", LogisticRegression(max_iter=1200, C=1.0, random_state=42)),
            ]
        ),
        "TFIDF + MultinomialNB": Pipeline(
            [
                ("vec", TfidfVectorizer(stop_words="english", min_df=2)),
                ("clf", MultinomialNB(alpha=0.3)),
            ]
        ),
        "TFIDF + LinearSVC": Pipeline(
            [
                ("vec", TfidfVectorizer(stop_words="english", min_df=2, max_features=40000)),
                ("clf", LinearSVC(C=1.0, random_state=42)),
            ]
        ),
    }

    scores: Dict[str, float] = {}
    best_model_name = ""
    best_acc = -1.0
    best_macro_f1 = -1.0

    print("\nTuning spot:")
    print("  Baseline comparison itself is a tuning strategy.")
    print("  Start broad across model families before deep hyperparameter search.")

    for name, pipe in pipelines.items():
        pipe.fit(train_texts, y_train)
        preds = pipe.predict(test_texts)
        acc = accuracy_score(y_test, preds)
        m_f1 = f1_score(y_test, preds, average="macro")
        scores[name] = acc

        print(f"  - {name}: accuracy={acc:.4f}, macro_f1={m_f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_macro_f1 = m_f1
            best_model_name = name

    print("\nHow to interpret:")
    print("  - If one model consistently leads, use it as your deployment baseline.")
    print("  - If scores are close, prioritize simpler and faster models first.")

    return BaselineResult(
        scores=scores,
        best_model_name=best_model_name,
        best_model_accuracy=best_acc,
        best_model_macro_f1=best_macro_f1,
    )
if __name__ == "__main__":
    print("Running baseline models...")

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = fetch_20newsgroups(subset="all")
    texts = data.data
    labels = data.target

    # Split
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Run pipeline
    result = run_baseline_route(train_texts, test_texts, y_train, y_test)

    print("\nFINAL RESULTS")
    print("Best Model:", result.best_model_name)
    print("Best Accuracy:", result.best_model_accuracy)
    print("Best Macro F1:", result.best_model_macro_f1)
    
    