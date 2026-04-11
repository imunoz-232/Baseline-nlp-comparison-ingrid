"""Offline fallback dataset for NLP route practice.

This lightweight corpus is used when 20 Newsgroups cannot be downloaded.
It preserves the same training flow so students can still practice all routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SimpleDataset:
    data: List[str]
    target: List[int]
    target_names: List[str]


def load_fallback_train_test() -> tuple[SimpleDataset, SimpleDataset]:
    """Return small train/test splits across several topics."""
    target_names = ["tech", "sports", "health", "finance"]

    train_data = [
        "Python package management and virtual environments for reproducible machine learning projects.",
        "Neural networks use gradient descent and backpropagation for training.",
        "Basketball teams use spacing and fast breaks to create open shots.",
        "The soccer coach changed formation to improve midfield pressing.",
        "Balanced nutrition supports long term heart health and energy levels.",
        "Regular sleep and hydration can improve recovery and immune response.",
        "Interest rates influence borrowing costs and market valuations.",
        "Diversification can reduce portfolio risk during market volatility.",
        "Feature engineering with tokenization and tf idf improves text baselines.",
        "A baseball pitcher improved control by adjusting release point.",
        "Strength training plus mobility work reduces injury risk.",
        "Quarterly earnings and cash flow shape investor expectations.",
    ]
    train_target = [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3]

    test_data = [
        "Model evaluation uses precision recall and cross validation metrics.",
        "The team won after improving defensive rebounding and transition play.",
        "Meditation and sleep quality can support stress management.",
        "Inflation data affected bond yields and equity prices this week.",
        "Tokenization and vocabulary size strongly influence bag of words models.",
        "The striker scored twice after tactical substitutions in the second half.",
        "Healthy diet patterns can improve blood pressure outcomes.",
        "Central bank policy statements moved currency markets.",
    ]
    test_target = [0, 1, 2, 3, 0, 1, 2, 3]

    train_ds = SimpleDataset(data=train_data, target=train_target, target_names=target_names)
    test_ds = SimpleDataset(data=test_data, target=test_target, target_names=target_names)
    return train_ds, test_ds
