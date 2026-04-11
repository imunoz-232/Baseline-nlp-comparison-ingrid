"""End-to-end 20 Newsgroups practice script.

Dataset origin and collection context:
- Source: 20 Newsgroups corpus distributed in scikit-learn.
- Original material: Usenet newsgroup posts from around the 1990s.
- Collection: Curated by Ken Lang for text learning research; later reused widely
  for education and benchmarking.

Important caveat for students:
- This is a historical internet dataset and may contain outdated language,
  bias, or offensive content.
- Real-world pipelines should include data governance and content safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.datasets import fetch_20newsgroups

from newsgroups_baseline_route import BaselineResult, run_baseline_route
from newsgroups_classification_route import ClassificationResult, run_classification_route
from newsgroups_cleaning import CleaningConfig, clean_documents, print_cleaning_guide
from newsgroups_clustering_route import ClusteringResult, run_clustering_route
from newsgroups_fallback_data import load_fallback_train_test
from newsgroups_retrieval_route import RetrievalResult, run_retrieval_route


@dataclass
class PracticeSummary:
    classification: ClassificationResult
    clustering: ClusteringResult
    retrieval: RetrievalResult
    baseline: BaselineResult


def print_dataset_context() -> None:
    """Explain where data comes from and why that matters."""
    print("=" * 80)
    print("20 NEWSGROUPS: DATA ORIGIN AND COLLECTION CONTEXT")
    print("=" * 80)
    print("Where data comes from:")
    print("  - Public Usenet newsgroup postings, grouped into 20 topic categories.")
    print("  - Popularized for ML by Ken Lang's curated split and mirrored in scikit-learn.")
    print("\nHow it was collected:")
    print("  - Posts were gathered from online discussion groups.")
    print("  - Documents were labeled by their newsgroup topic.")
    print("\nWhy students should care:")
    print("  - It is realistic enough for NLP workflow practice.")
    print("  - It can include noise (quotes, signatures, headers) and historical bias.")


def load_data(remove_metadata_noise: bool = True):
    """Load train/test splits with optional metadata stripping.

    Falls back to a bundled local dataset when internet download is unavailable.
    """
    remove_parts = ("headers", "footers", "quotes") if remove_metadata_noise else ()

    try:
        train = fetch_20newsgroups(subset="train", remove=remove_parts, shuffle=True, random_state=42)
        test = fetch_20newsgroups(subset="test", remove=remove_parts, shuffle=True, random_state=42)
        print("Loaded official 20 Newsgroups dataset from scikit-learn cache/source.")
        return train, test
    except Exception as exc:
        print("Could not download/load 20 Newsgroups. Switching to offline fallback corpus.")
        print(f"Reason: {exc}")
        train, test = load_fallback_train_test()
        return train, test


def print_sample_before_after(original_docs: List[str], cleaned_docs: List[str], n: int = 2) -> None:
    """Show cleaning effects for student interpretation."""
    print("\n" + "=" * 80)
    print("CLEANING EXAMPLES (BEFORE VS AFTER)")
    print("=" * 80)
    limit = min(n, len(original_docs))
    for i in range(limit):
        print(f"\nDocument {i + 1} BEFORE (first 250 chars):")
        print(original_docs[i][:250].replace("\n", " "))
        print("Document AFTER (first 250 chars):")
        print(cleaned_docs[i][:250])


def print_final_comparison(summary: PracticeSummary) -> None:
    """Print a cross-route summary table with interpretation."""
    print("\n" + "=" * 80)
    print("FINAL CROSS-ROUTE SUMMARY")
    print("=" * 80)
    print("Route                               Main metric                     Value")
    print("-" * 80)
    print(
        "Classification                      Test accuracy                  "
        f"{summary.classification.test_accuracy:.4f}"
    )
    print(
        "Clustering                          Best silhouette                "
        f"{summary.clustering.best_silhouette:.4f}"
    )
    print(
        f"Retrieval                           Precision@{summary.retrieval.k}                  "
        f"{summary.retrieval.precision_at_k:.4f}"
    )
    print(
        "Baseline comparison                 Best baseline accuracy         "
        f"{summary.baseline.best_model_accuracy:.4f}"
    )

    print("\nHow to compare these fairly:")
    print("  - These metrics are not directly interchangeable.")
    print("  - Use each metric inside its task context.")
    print("  - In many projects, classification and baseline routes are the first production path.")

    print("\nRecommended student exercises:")
    print("  1. Toggle remove_metadata_noise to see how leaked artifacts affect scores.")
    print("  2. Change min_df and ngram_range to observe vocabulary and performance tradeoffs.")
    print("  3. Compare CountVectorizer vs TfidfVectorizer choices in each route.")
    print("  4. Increase retrieval k and observe precision@k behavior.")


def main() -> None:
    print_dataset_context()

    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)
    train_ds, test_ds = load_data(remove_metadata_noise=True)
    print(f"Train documents: {len(train_ds.data)}")
    print(f"Test documents : {len(test_ds.data)}")
    print(f"Number of classes: {len(train_ds.target_names)}")

    cleaning_config = CleaningConfig(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_numbers=False,
        keep_only_letters=True,
        min_token_length=2,
    )
    print_cleaning_guide(cleaning_config)

    print("\n" + "=" * 80)
    print("STEP 2: CLEAN TEXT")
    print("=" * 80)
    x_train_clean = clean_documents(train_ds.data, cleaning_config)
    x_test_clean = clean_documents(test_ds.data, cleaning_config)
    print_sample_before_after(train_ds.data, x_train_clean)

    classification = run_classification_route(
        train_texts=x_train_clean,
        test_texts=x_test_clean,
        y_train=train_ds.target,
        y_test=test_ds.target,
        target_names=train_ds.target_names,
    )

    clustering = run_clustering_route(
        all_texts=x_train_clean + x_test_clean,
        all_labels=list(train_ds.target) + list(test_ds.target),
        expected_k=len(train_ds.target_names),
    )

    retrieval = run_retrieval_route(
        train_texts=x_train_clean,
        test_texts=x_test_clean,
        y_train=train_ds.target,
        y_test=test_ds.target,
        target_names=train_ds.target_names,
        k=5,
        sample_query_count=5,
    )

    baseline = run_baseline_route(
        train_texts=x_train_clean,
        test_texts=x_test_clean,
        y_train=train_ds.target,
        y_test=test_ds.target,
    )

    summary = PracticeSummary(
        classification=classification,
        clustering=clustering,
        retrieval=retrieval,
        baseline=baseline,
    )
    print_final_comparison(summary)


if __name__ == "__main__":
    main()
