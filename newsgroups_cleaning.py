"""Text cleaning utilities for the 20 Newsgroups practice pipeline.

This module keeps cleaning logic separate so students can inspect and tune preprocessing
without touching model code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class CleaningConfig:
    """Configuration for text cleaning.

    Attributes:
        lowercase: Convert text to lowercase.
        remove_urls: Remove URL patterns.
        remove_emails: Remove email-like patterns.
        remove_numbers: Remove standalone digits.
        keep_only_letters: Keep letters and spaces only.
        min_token_length: Drop very short tokens after cleaning.
    """

    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = False
    keep_only_letters: bool = True
    min_token_length: int = 2


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+\b")
_NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, config: CleaningConfig) -> str:
    """Clean one text sample according to configuration."""
    cleaned = text

    if config.lowercase:
        cleaned = cleaned.lower()
    if config.remove_urls:
        cleaned = _URL_RE.sub(" ", cleaned)
    if config.remove_emails:
        cleaned = _EMAIL_RE.sub(" ", cleaned)
    if config.remove_numbers:
        cleaned = _NUM_RE.sub(" ", cleaned)
    if config.keep_only_letters:
        cleaned = _NON_LETTER_RE.sub(" ", cleaned)

    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    if config.min_token_length > 1 and cleaned:
        tokens = [tok for tok in cleaned.split() if len(tok) >= config.min_token_length]
        cleaned = " ".join(tokens)

    return cleaned


def clean_documents(documents: Iterable[str], config: CleaningConfig) -> List[str]:
    """Clean a sequence of documents."""
    return [clean_text(doc, config) for doc in documents]


def print_cleaning_guide(config: CleaningConfig) -> None:
    """Print student-focused guidance on current cleaning configuration."""
    print("\n" + "=" * 80)
    print("PREPROCESSING GUIDE")
    print("=" * 80)
    print("Why cleaning matters:")
    print("  - BoW treats each unique token as a separate feature column.")
    print("  - Noisy tokens inflate vocabulary and often hurt generalization.")
    print("\nCurrent cleaning settings:")
    print(f"  - lowercase={config.lowercase}")
    print(f"  - remove_urls={config.remove_urls}")
    print(f"  - remove_emails={config.remove_emails}")
    print(f"  - remove_numbers={config.remove_numbers}")
    print(f"  - keep_only_letters={config.keep_only_letters}")
    print(f"  - min_token_length={config.min_token_length}")
    print("\nTuning note:")
    print("  - If your task needs numbers (for example model numbers), set remove_numbers=False.")
    print("  - If short words matter (for example AI, ML), lower min_token_length.")
