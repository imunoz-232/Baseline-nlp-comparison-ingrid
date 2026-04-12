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

import hashlib
import os
from dataclasses import dataclass
from dataclasses import asdict
from typing import List

import joblib
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


@dataclass
class PreparedData:
    train_texts: List[str]
    test_texts: List[str]
    y_train: List[int]
    y_test: List[int]
    target_names: List[str]
    sample_before: List[str]
    sample_after: List[str]
    data_source: str


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


def _cache_file_path(cleaning_config: CleaningConfig, remove_metadata_noise: bool) -> str:
    """Build a deterministic cache file path for this preprocessing configuration."""
    cfg_str = repr(sorted(asdict(cleaning_config).items())) + f"|remove_meta={remove_metadata_noise}"
    cfg_hash = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()[:12]
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"prepared_data_{cfg_hash}.joblib")


def prepare_data_for_routes(
    cleaning_config: CleaningConfig,
    remove_metadata_noise: bool = True,
    use_cache: bool = True,
    force_rebuild_cache: bool = False,
) -> PreparedData:
    """Load and clean data once, then reuse from local cache on later runs."""
    cache_path = _cache_file_path(cleaning_config, remove_metadata_noise)

    if use_cache and not force_rebuild_cache and os.path.exists(cache_path):
        cached = joblib.load(cache_path)
        print("\n" + "=" * 80)
        print("STEP 1: LOAD DATA")
        print("=" * 80)
        print("Loaded prepared dataset from local cache.")
        print(f"Cache file: {cache_path}")
        print(f"Data source: {cached['data_source']}")
        print(f"Train documents: {len(cached['train_texts'])}")
        print(f"Test documents : {len(cached['test_texts'])}")
        print(f"Number of classes: {len(cached['target_names'])}")

        return PreparedData(
            train_texts=cached["train_texts"],
            test_texts=cached["test_texts"],
            y_train=cached["y_train"],
            y_test=cached["y_test"],
            target_names=cached["target_names"],
            sample_before=cached["sample_before"],
            sample_after=cached["sample_after"],
            data_source=cached["data_source"],
        )

    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)
    train_ds, test_ds = load_data(remove_metadata_noise=remove_metadata_noise)
    print(f"Train documents: {len(train_ds.data)}")
    print(f"Test documents : {len(test_ds.data)}")
    print(f"Number of classes: {len(train_ds.target_names)}")

    print("\n" + "=" * 80)
    print("STEP 2: CLEAN TEXT")
    print("=" * 80)
    x_train_clean = clean_documents(train_ds.data, cleaning_config)
    x_test_clean = clean_documents(test_ds.data, cleaning_config)

    sample_before = [doc[:250].replace("\n", " ") for doc in train_ds.data[:2]]
    sample_after = [doc[:250] for doc in x_train_clean[:2]]

    data_source = "20_newsgroups"
    if len(train_ds.target_names) == 4 and train_ds.target_names == ["tech", "sports", "health", "finance"]:
        data_source = "offline_fallback"

    prepared = PreparedData(
        train_texts=list(x_train_clean),
        test_texts=list(x_test_clean),
        y_train=list(train_ds.target),
        y_test=list(test_ds.target),
        target_names=list(train_ds.target_names),
        sample_before=sample_before,
        sample_after=sample_after,
        data_source=data_source,
    )

    if use_cache:
        joblib.dump(
            {
                "train_texts": prepared.train_texts,
                "test_texts": prepared.test_texts,
                "y_train": prepared.y_train,
                "y_test": prepared.y_test,
                "target_names": prepared.target_names,
                "sample_before": prepared.sample_before,
                "sample_after": prepared.sample_after,
                "data_source": prepared.data_source,
            },
            cache_path,
        )
        print(f"Saved prepared dataset cache: {cache_path}")

    return prepared


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

    # Caching saves time on repeated runs while debugging/tuning code.
    use_cache = True
    force_rebuild_cache = False
    remove_metadata_noise = True

    cleaning_config = CleaningConfig(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_numbers=False,
        keep_only_letters=True,
        min_token_length=2,
    )
    print_cleaning_guide(cleaning_config)

    prepared = prepare_data_for_routes(
        cleaning_config=cleaning_config,
        remove_metadata_noise=remove_metadata_noise,
        use_cache=use_cache,
        force_rebuild_cache=force_rebuild_cache,
    )

    print("\n" + "=" * 80)
    print("CLEANING EXAMPLES (BEFORE VS AFTER)")
    print("=" * 80)
    for idx, (before, after) in enumerate(zip(prepared.sample_before, prepared.sample_after), 1):
        print(f"\nDocument {idx} BEFORE (first 250 chars):")
        print(before)
        print("Document AFTER (first 250 chars):")
        print(after)

    classification = run_classification_route(
        train_texts=prepared.train_texts,
        test_texts=prepared.test_texts,
        y_train=prepared.y_train,
        y_test=prepared.y_test,
        target_names=prepared.target_names,
    )

    clustering = run_clustering_route(
        all_texts=prepared.train_texts + prepared.test_texts,
        all_labels=prepared.y_train + prepared.y_test,
        expected_k=len(prepared.target_names),
    )

    retrieval = run_retrieval_route(
        train_texts=prepared.train_texts,
        test_texts=prepared.test_texts,
        y_train=prepared.y_train,
        y_test=prepared.y_test,
        target_names=prepared.target_names,
        k=5,
        sample_query_count=5,
    )

    baseline = run_baseline_route(
        train_texts=prepared.train_texts,
        test_texts=prepared.test_texts,
        y_train=prepared.y_train,
        y_test=prepared.y_test,
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
