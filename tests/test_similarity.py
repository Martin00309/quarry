"""Tests for the BM25 and TF-IDF scoring models."""

from __future__ import annotations

import pytest

from quarry.similarity import (
    BM25Similarity,
    CollectionStatistics,
    TFIDFSimilarity,
    get_similarity,
)


def _stats(num_docs=1000, avg_len=10.0):
    return CollectionStatistics(num_docs=num_docs, avg_field_length=avg_len)


def test_bm25_idf_non_negative_and_monotonic():
    sim = BM25Similarity()
    rare = sim.idf(doc_freq=1, num_docs=1000)
    common = sim.idf(doc_freq=500, num_docs=1000)
    assert rare > common > 0


def test_bm25_tf_saturation():
    sim = BM25Similarity()
    weight = sim.term_weight(doc_freq=10, stats=_stats())
    s1 = weight.score(tf=1, field_length=10)
    s2 = weight.score(tf=2, field_length=10)
    s10 = weight.score(tf=10, field_length=10)
    assert s2 > s1
    # diminishing returns: going 1->2 adds more than 9->10
    assert (s2 - s1) > (s10 - weight.score(tf=9, field_length=10))


def test_bm25_length_normalization():
    sim = BM25Similarity()
    weight = sim.term_weight(doc_freq=10, stats=_stats(avg_len=10.0))
    short = weight.score(tf=2, field_length=5)
    long = weight.score(tf=2, field_length=50)
    assert short > long


def test_bm25_zero_tf_is_zero():
    weight = BM25Similarity().term_weight(doc_freq=5, stats=_stats())
    assert weight.score(tf=0, field_length=10) == 0.0


def test_bm25_parameter_validation():
    with pytest.raises(ValueError):
        BM25Similarity(k1=-1)
    with pytest.raises(ValueError):
        BM25Similarity(b=2.0)


def test_bm25_boost_scales_score():
    sim = BM25Similarity()
    base = sim.term_weight(doc_freq=10, stats=_stats(), boost=1.0)
    boosted = sim.term_weight(doc_freq=10, stats=_stats(), boost=3.0)
    assert boosted.score(2, 10) == pytest.approx(3.0 * base.score(2, 10))


def test_tfidf_basic_behaviour():
    sim = TFIDFSimilarity()
    weight = sim.term_weight(doc_freq=10, stats=_stats())
    assert weight.score(tf=4, field_length=10) > weight.score(tf=1, field_length=10)
    assert weight.score(tf=2, field_length=4) > weight.score(tf=2, field_length=40)


def test_explanation_matches_score():
    weight = BM25Similarity().term_weight(doc_freq=10, stats=_stats())
    exp = weight.explain(tf=3, field_length=12)
    assert exp.value == pytest.approx(weight.score(3, 12))
    assert "BM25" in exp.description


def test_get_similarity():
    assert isinstance(get_similarity("bm25"), BM25Similarity)
    assert isinstance(get_similarity("tfidf"), TFIDFSimilarity)
    with pytest.raises(KeyError):
        get_similarity("nonsense")
