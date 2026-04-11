"""Document clustering route for 20 Newsgroups.

This module demonstrates unsupervised learning using TF-IDF + KMeans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import homogeneity_score, silhouette_score


@dataclass
class ClusteringResult:
    best_k: int
    best_silhouette: float
    homogeneity_at_best_k: float


def run_clustering_route(
    all_texts: Sequence[str],
    all_labels: Sequence[int],
    expected_k: int,
) -> ClusteringResult:
    """Run clustering with a tuning sweep over number of clusters."""
    print("\n" + "=" * 80)
    print("ROUTE 2: DOCUMENT CLUSTERING (TF-IDF + KMeans)")
    print("=" * 80)
    print("Goal:")
    print("  Group similar documents without using labels during training.")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=40000, min_df=2)
    x_all = vectorizer.fit_transform(all_texts)

    candidate_k = sorted({max(5, expected_k // 2), expected_k, expected_k + 10})

    print("\nTuning spot:")
    print("  We vary number of clusters (k).")
    print("  Why: too small mixes topics, too large fragments topics.")

    best_k = candidate_k[0]
    best_silhouette = -1.0
    best_homogeneity = -1.0

    for k in candidate_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(x_all)

        # Silhouette can be expensive on large sparse matrices, so sample for speed.
        sil = silhouette_score(x_all, cluster_ids, sample_size=min(3000, x_all.shape[0]), random_state=42)
        homo = homogeneity_score(all_labels, cluster_ids)
        print(f"  - k={k}: silhouette={sil:.4f}, homogeneity={homo:.4f}")

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k
            best_homogeneity = homo

    print("\nHow to interpret:")
    print("  - Silhouette closer to 1 means tighter, better-separated clusters.")
    print("  - Homogeneity closer to 1 means each cluster tends to contain one true topic.")
    print("  - In real projects, clustering scores are often lower than classification scores.")

    return ClusteringResult(
        best_k=best_k,
        best_silhouette=best_silhouette,
        homogeneity_at_best_k=best_homogeneity,
    )
