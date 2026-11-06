"""Tests for query execution: term, boolean, phrase, prefix, wildcard, range."""

from __future__ import annotations

import pytest

from quarry.query import (
    BooleanQuery,
    Clause,
    MatchAllQuery,
    NumericRangeQuery,
    Occur,
    PhraseQuery,
    PrefixQuery,
    TermQuery,
    WildcardQuery,
    phrase_freq,
)
from quarry.searcher import Searcher


@pytest.fixture
def searcher(sample_index):
    return Searcher.open(sample_index)


def _ids(result):
    return {hit.doc_id for hit in result.hits}


def test_match_all(searcher):
    result = searcher.search(MatchAllQuery())
    assert result.total_hits == 5


def test_term_query(searcher):
    assert _ids(searcher.search(TermQuery("body", "fox"))) == {"1", "5"}
    assert _ids(searcher.search(TermQuery("title", "brown"))) == {"1", "4"}


def test_term_query_no_match(searcher):
    result = searcher.search(TermQuery("body", "nonexistent"))
    assert result.total_hits == 0
    assert result.hits == []


def test_boolean_must(searcher):
    query = BooleanQuery(
        [
            Clause(TermQuery("body", "fox"), Occur.MUST),
            Clause(TermQuery("body", "brown"), Occur.MUST),
        ]
    )
    assert _ids(searcher.search(query)) == {"1"}


def test_boolean_should(searcher):
    query = BooleanQuery(
        [
            Clause(TermQuery("body", "fox"), Occur.SHOULD),
            Clause(TermQuery("body", "bear"), Occur.SHOULD),
        ]
    )
    assert _ids(searcher.search(query)) == {"1", "4", "5"}


def test_boolean_must_not(searcher):
    query = BooleanQuery(
        [
            Clause(TermQuery("body", "brown"), Occur.SHOULD),
            Clause(TermQuery("body", "fox"), Occur.MUST_NOT),
        ]
    )
    assert _ids(searcher.search(query)) == {"4"}


def test_boolean_minimum_should_match(searcher):
    query = BooleanQuery(
        [
            Clause(TermQuery("body", "fox"), Occur.SHOULD),
            Clause(TermQuery("body", "brown"), Occur.SHOULD),
            Clause(TermQuery("body", "dog"), Occur.SHOULD),
        ],
        minimum_should_match=2,
    )
    # only doc 1 has at least two of fox/brown/dog in its body
    assert _ids(searcher.search(query)) == {"1"}


def test_phrase_query_exact(searcher):
    assert _ids(searcher.search(PhraseQuery("body", ["quick", "brown"]))) == {"1"}
    assert searcher.search(PhraseQuery("body", ["brown", "quick"])).total_hits == 0


def test_phrase_query_slop(searcher):
    assert searcher.search(PhraseQuery("body", ["quick", "fox"], slop=0)).total_hits == 0
    assert _ids(searcher.search(PhraseQuery("body", ["quick", "fox"], slop=1))) == {"1"}


def test_prefix_query(searcher):
    assert _ids(searcher.search(PrefixQuery("title", "bro"))) == {"1", "4"}


def test_wildcard_query(searcher):
    assert _ids(searcher.search(WildcardQuery("title", "f?x"))) == {"1", "5"}


def test_numeric_range_query(searcher):
    query = NumericRangeQuery("year", 2005, 2015, include_lower=True, include_upper=True)
    assert _ids(searcher.search(query)) == {"2", "3", "4"}


def test_numeric_range_exclusive(searcher):
    query = NumericRangeQuery("year", 2005, 2015, include_lower=False, include_upper=False)
    assert _ids(searcher.search(query)) == {"3"}


def test_numeric_range_open_ended(searcher):
    query = NumericRangeQuery("year", lower=2015, upper=None)
    assert _ids(searcher.search(query)) == {"4", "5"}


def test_top_k_limits_hits(searcher):
    result = searcher.search(MatchAllQuery(), top_k=2)
    assert len(result.hits) == 2
    assert result.total_hits == 5


# -- phrase_freq unit tests ---------------------------------------------
def test_phrase_freq_exact():
    assert phrase_freq([[1], [2]], slop=0) == 1
    assert phrase_freq([[1], [3]], slop=0) == 0
    assert phrase_freq([[1, 5], [2, 6]], slop=0) == 2


def test_phrase_freq_slop():
    assert phrase_freq([[1], [3]], slop=1) == 1
    assert phrase_freq([[1], [3]], slop=0) == 0
    assert phrase_freq([[1], [4]], slop=1) == 0


def test_phrase_freq_missing_term():
    assert phrase_freq([[1], []], slop=5) == 0


def test_phrase_freq_single_term():
    assert phrase_freq([[1, 4, 9]], slop=0) == 3
