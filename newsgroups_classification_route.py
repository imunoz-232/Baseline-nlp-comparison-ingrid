"""Text classification route for 20 Newsgroups.

This module demonstrates a supervised route using BoW + Logistic Regression,
including a simple, student-friendly tuning loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class ClassificationResult:
    best_validation_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    best_config: Dict[str, object]


def _train_eval_once(
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    y_train: Sequence[int],
    y_val: Sequence[int],
    config: Dict[str, object],
) -> Tuple[float, CountVectorizer, LogisticRegression]:
    vectorizer = CountVectorizer(
        ngram_range=config["ngram_range"],
        min_df=config["min_df"],
        max_features=config["max_features"],
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    clf = LogisticRegression(
        C=config["C"],
        max_iter=1200,
        random_state=42,
    )
    clf.fit(x_train, y_train)

    val_preds = clf.predict(x_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    return val_accuracy, vectorizer, clf


def run_classification_route(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    y_train: Sequence[int],
    y_test: Sequence[int],
    target_names: List[str],
) -> ClassificationResult:
    """Run supervised classification with guided tuning and interpretation."""
    print("\n" + "=" * 80)
    print("ROUTE 1: TEXT CLASSIFICATION (BoW + Logistic Regression)")
    print("=" * 80)
    print("Goal:")
    print("  Predict which of the 20 newsgroup topics each document belongs to.")

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    candidate_configs = [
        {"name": "cfg_1", "ngram_range": (1, 1), "min_df": 2, "max_features": 25000, "C": 1.0},
        {"name": "cfg_2", "ngram_range": (1, 2), "min_df": 2, "max_features": 35000, "C": 1.0},
        {"name": "cfg_3", "ngram_range": (1, 2), "min_df": 3, "max_features": 30000, "C": 2.0},
    ]

    print("\nTuning spot:")
    print("  We test a few settings for n-grams, vocabulary size, min_df, and regularization C.")
    print("  Why: these control feature richness vs. overfitting risk.")

    best_acc = -1.0
    best_config: Dict[str, object] = {}

    for cfg in candidate_configs:
        acc, _, _ = _train_eval_once(tr_texts, val_texts, tr_labels, val_labels, cfg)
        print(
            f"  - {cfg['name']}: ngram_range={cfg['ngram_range']}, min_df={cfg['min_df']}, "
            f"max_features={cfg['max_features']}, C={cfg['C']} -> val_accuracy={acc:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            best_config = cfg

    print("\nBest config selected from validation:")
    print(f"  {best_config}")

    final_vectorizer = CountVectorizer(
        ngram_range=best_config["ngram_range"],
        min_df=best_config["min_df"],
        max_features=best_config["max_features"],
        stop_words="english",
    )
    x_train_full = final_vectorizer.fit_transform(train_texts)
    x_test = final_vectorizer.transform(test_texts)

    final_clf = LogisticRegression(
        C=best_config["C"],
        max_iter=1200,
        random_state=42,
    )
    final_clf.fit(x_train_full, y_train)

    test_preds = final_clf.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")

    print("\nTest set results:")
    print(f"  - Accuracy: {test_accuracy:.4f}")
    print(f"  - Macro F1: {test_macro_f1:.4f}")
    print("\nHow to interpret:")
    print("  - Accuracy = fraction of all documents predicted correctly.")
    print("  - Macro F1 = average per-class quality, useful when classes are imbalanced.")

    print("\nShort classification report sample:")
    report = classification_report(y_test, test_preds, target_names=target_names, digits=3)
    report_lines = report.splitlines()
    for line in report_lines[:16]:
        print(line)
    print("... (truncated for readability)")

    return ClassificationResult(
        best_validation_accuracy=best_acc,
        test_accuracy=test_accuracy,
        test_macro_f1=test_macro_f1,
        best_config=best_config,
    )
if __name__ == "__main__":
    print("Running classification route...")

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split

    data = fetch_20newsgroups(subset="all")
    texts = data.data
    labels = data.target
    target_names = data.target_names  # 🔥 necesario

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    result = run_classification_route(
        train_texts,
        test_texts,
        y_train,
        y_test,
        target_names
    )

    print("\nFINAL RESULTS")
    print("Best Validation Accuracy:", result.best_validation_accuracy)
    print("Test Accuracy:", result.test_accuracy)
    print("Test Macro F1:", result.test_macro_f1)
    print("Best Config:", result.best_config)