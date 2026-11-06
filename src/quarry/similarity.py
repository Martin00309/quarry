"""Relevance scoring models.

A similarity turns the raw signals of a match -- term frequency, document
frequency, field length -- into a score. Two models are provided:

``BM25Similarity`` (default)
    The Okapi BM25 ranking function, the modern default for lexical search. It
    saturates term frequency (the tenth occurrence of a word adds far less than
    the second) and normalizes by field length.
``TFIDFSimilarity``
    A classic ``tf-idf`` with a square-root term-frequency damp and inverse
    square-root length norm, useful as a simpler baseline.

Scoring is split into a per-term :class:`TermWeight`, built once from collection
statistics, and a per-document ``score`` call. This mirrors how a real engine
hoists the idf computation out of the inner loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Explanation


@dataclass
class CollectionStatistics:
    """Collection-wide statistics a similarity needs for normalization."""

    num_docs: int
    #: Average indexed length of the field being scored, across the collection.
    avg_field_length: float


class TermWeight:
    """Per-term scorer with its idf (and any constants) precomputed."""

    def __init__(self, idf: float) -> None:
        self.idf = idf

    def score(self, tf: int, field_length: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def explain(self, tf: int, field_length: int) -> Explanation:  # pragma: no cover - interface
        raise NotImplementedError


class Similarity:
    """Base class for scoring models."""

    name = "similarity"

    def term_weight(
        self, doc_freq: int, stats: CollectionStatistics, boost: float = 1.0
    ) -> TermWeight:  # pragma: no cover - interface
        raise NotImplementedError


# -- BM25 ----------------------------------------------------------------
class _BM25Weight(TermWeight):
    def __init__(self, idf: float, k1: float, b: float, avg_len: float, boost: float) -> None:
        super().__init__(idf)
        self.k1 = k1
        self.b = b
        self.avg_len = avg_len if avg_len > 0 else 1.0
        self.boost = boost

    def score(self, tf: int, field_length: int) -> float:
        if tf <= 0:
            return 0.0
        norm = 1.0 - self.b + self.b * (field_length / self.avg_len)
        saturated = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return self.boost * self.idf * saturated

    def explain(self, tf: int, field_length: int) -> Explanation:
        norm = 1.0 - self.b + self.b * (field_length / self.avg_len)
        saturated = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return Explanation(
            self.score(tf, field_length),
            "BM25 score = boost * idf * tf_norm",
            [
                Explanation(self.boost, "boost"),
                Explanation(self.idf, "idf"),
                Explanation(
                    saturated,
                    f"tf_norm: tf={tf} k1={self.k1} b={self.b} dl={field_length} "
                    f"avgdl={self.avg_len:.3f}",
                ),
            ],
        )


class BM25Similarity(Similarity):
    """Okapi BM25 with the usual ``k1`` and ``b`` tuning parameters."""

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b

    def idf(self, doc_freq: int, num_docs: int) -> float:
        # Probabilistic idf with the +0.5 smoothing that keeps it non-negative.
        return math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def term_weight(
        self, doc_freq: int, stats: CollectionStatistics, boost: float = 1.0
    ) -> TermWeight:
        idf = self.idf(doc_freq, stats.num_docs)
        return _BM25Weight(idf, self.k1, self.b, stats.avg_field_length, boost)


# -- TF-IDF --------------------------------------------------------------
class _TFIDFWeight(TermWeight):
    def __init__(self, idf: float, boost: float) -> None:
        super().__init__(idf)
        self.boost = boost

    def score(self, tf: int, field_length: int) -> float:
        if tf <= 0:
            return 0.0
        tf_weight = math.sqrt(tf)
        length_norm = 1.0 / math.sqrt(field_length) if field_length > 0 else 1.0
        return self.boost * tf_weight * (self.idf * self.idf) * length_norm

    def explain(self, tf: int, field_length: int) -> Explanation:
        return Explanation(
            self.score(tf, field_length),
            "tf-idf score = boost * sqrt(tf) * idf^2 * lengthNorm",
            [
                Explanation(self.boost, "boost"),
                Explanation(math.sqrt(tf), f"sqrt(tf), tf={tf}"),
                Explanation(self.idf, "idf"),
                Explanation(
                    1.0 / math.sqrt(field_length) if field_length > 0 else 1.0,
                    f"lengthNorm, dl={field_length}",
                ),
            ],
        )


class TFIDFSimilarity(Similarity):
    """A classic tf-idf with sqrt tf damping and inverse sqrt length norm."""

    name = "tfidf"

    def idf(self, doc_freq: int, num_docs: int) -> float:
        return 1.0 + math.log((num_docs + 1.0) / (doc_freq + 1.0))

    def term_weight(
        self, doc_freq: int, stats: CollectionStatistics, boost: float = 1.0
    ) -> TermWeight:
        return _TFIDFWeight(self.idf(doc_freq, stats.num_docs), boost)


_SIMILARITIES = {
    "bm25": BM25Similarity,
    "tfidf": TFIDFSimilarity,
}


def get_similarity(name: str) -> Similarity:
    try:
        return _SIMILARITIES[name]()
    except KeyError:
        raise KeyError(f"unknown similarity: {name!r}") from None
