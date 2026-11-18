"""Exact cosine vector index; TF-IDF baseline or Ollama dense embeddings."""
from collections import Counter
import math
import re

import numpy as np

from .corpus import Passage
from .llm import ModelError


def tokens(text: str) -> list[str]:
    return re.findall(r'[^\W_]{2,}', text.casefold(), re.UNICODE)


def lexical_features(text):
    stop = set('the and of to in is are with for from that this these what how why '
               'le la les des du un une et dans est sont pour avec sur aux ces '
               'comment quels quelles quel quelle pourquoi'.split())
    words = [word for word in tokens(text) if word not in stop]
    return words + ['#' + word[:5] for word in words if len(word) >= 7]


class VectorIndex:
    def __init__(self, passages: list[Passage], embed=None):
        if not passages:
            raise ValueError('Upload at least one readable document.')
        self.passages = passages
        self.embed = embed
        counts = [Counter(lexical_features(p.text)) for p in passages]
        vocab = sorted(set().union(*(set(c) for c in counts)))
        self.vocab = {word: i for i, word in enumerate(vocab)}
        self.idf = np.array([math.log((1 + len(counts)) /
                            (1 + sum(word in c for c in counts))) + 1 for word in vocab])
        self.lexical_matrix = self._normalize(np.asarray([self._tfidf(c) for c in counts]))
        if embed:
            matrix = np.asarray(embed([p.text for p in passages]), dtype=float)
        else:
            matrix = self.lexical_matrix
        if matrix.ndim != 2 or matrix.shape[0] != len(passages) or not np.isfinite(matrix).all():
            raise ValueError('Invalid embedding response.')
        self.matrix = self._normalize(matrix)

    @staticmethod
    def _normalize(matrix):
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    def _tfidf(self, count):
        row = np.zeros(len(self.vocab))
        for word, n in count.items():
            if word in self.vocab:
                row[self.vocab[word]] = (1 + math.log(n)) * (0.8 if word.startswith('#') else 1)
        return row * self.idf

    def search(self, query: str, k: int = 4) -> list[tuple[Passage, float]]:
        if not tokens(query):
            return []
        lexical = self._normalize(np.asarray([self._tfidf(Counter(lexical_features(query)))]))[0]
        lexical_scores = self.lexical_matrix @ lexical
        if not self.embed:
            scores = lexical_scores
            return [(self.passages[i], float(scores[i]))
                    for i in np.argsort(-scores, kind='stable')[:k] if scores[i] > 0]
        try:
            raw = self.embed([query])
        except ModelError:
            return [(self.passages[i], float(lexical_scores[i]))
                    for i in np.argsort(-lexical_scores, kind='stable')[:k] if lexical_scores[i] > 0]
        vector = np.asarray(raw, dtype=float)
        if vector.shape != (1, self.matrix.shape[1]) or not np.isfinite(vector).all():
            raise ValueError('Invalid query embedding response.')
        scores = 0.45 * (self.matrix @ self._normalize(vector)[0]) + 0.55 * lexical_scores
        return [(self.passages[i], float(scores[i]))
                for i in np.argsort(-scores, kind='stable')[:k] if scores[i] > 0]
