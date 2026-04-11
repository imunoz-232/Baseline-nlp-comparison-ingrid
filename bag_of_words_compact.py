"""
Compact Bag of Words Feature Extractor

Efficient implementation for advanced NLP workflows.
Supports multiple vectorization methods, optional TF-IDF weighting, and flexible output.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from typing import List, Tuple, Union


class BagOfWordsProcessor:
    """Compact vectorizer for text-to-numerical conversion."""
    
    def __init__(self, method: str = "count", max_features: int = None, 
                 ngram_range: Tuple[int, int] = (1, 1)):
        """
        Initialize processor.
        
        Args:
            method: "count" for word counts or "tfidf" for TF-IDF weights
            max_features: Limit vocabulary size (None = no limit)
            ngram_range: Unigram (1,1), Bigram (1,2), etc.
        """
        VectorClass = TfidfVectorizer if method == "tfidf" else CountVectorizer
        self.vectorizer = VectorClass(max_features=max_features, ngram_range=ngram_range)
        self.matrix = None
        self.features = None
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform documents to feature matrix.
        
        Note: "fit" here means "learn vocabulary", NOT "fit a model".
        It scans documents to find unique words and builds the vocabulary dictionary.
        """
        self.matrix = self.vectorizer.fit_transform(documents).toarray()
        self.features = self.vectorizer.get_feature_names_out()
        return self.matrix
    
    def get_feature_dict(self, doc_idx: int) -> dict:
        """Return document features as key-value pairs."""
        return {word: count for word, count in 
                zip(self.features, self.matrix[doc_idx]) if count > 0}
    
    def get_top_features(self, doc_idx: int, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top N features for a document."""
        features_dict = self.get_feature_dict(doc_idx)
        return sorted(features_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]


def main():
    """Execute compact BoW processing."""
    
    documents = [
        "I love programming in Python",
        "Python is a great language",
        "I enjoy machine learning"
    ]
    
    # Initialize and process
    processor = BagOfWordsProcessor(method="count")
    matrix = processor.fit_transform(documents)
    
    # Output
    print("Features:", processor.features)
    print("\nMatrix:\n", matrix)
    print("\nTop features by document:")
    for i, doc in enumerate(documents):
        print(f"  Doc {i}: {processor.get_top_features(i, top_n=3)}")


if __name__ == "__main__":
    main()
