"""Tests for the query-string parser."""

from __future__ import annotations

import pytest

from quarry.errors import QueryParseError
from quarry.qparser import parse_query, tokenize
from quarry.query import NumericRangeQuery, PhraseQuery, PrefixQuery, TermQuery
from quarry.searcher import Searcher


@pytest.fixture
def searcher(sample_index):
    return Searcher.open(sample_index)


def _ids(searcher, text):
    query = parse_query(text, searcher.schema)
    return {hit.doc_id for hit in searcher.search(query).hits}


# -- parsing shapes ------------------------------------------------------
def test_fielded_term(sample_schema):
    query = parse_query("title:brown", sample_schema)
    assert isinstance(query, TermQuery)
    assert query.field == "title" and query.term == "brown"


def test_bare_term_expands_to_default_fields(sample_schema):
    query = parse_query("fox", sample_schema)
    # (title:fox OR body:fox)
    assert "title:fox" in repr(query) and "body:fox" in repr(query)


def test_phrase_parsing(sample_schema):
    query = parse_query('title:"quick brown"', sample_schema)
    assert isinstance(query, PhraseQuery)
    assert query.terms == ["quick", "brown"]
    assert query.slop == 0


def test_phrase_slop_parsing(sample_schema):
    query = parse_query('body:"quick fox"~2', sample_schema)
    assert isinstance(query, PhraseQuery)
    assert query.slop == 2


def test_prefix_parsing(sample_schema):
    query = parse_query("title:bro*", sample_schema)
    assert isinstance(query, PrefixQuery)
    assert query.prefix == "bro"


def test_range_parsing(sample_schema):
    query = parse_query("year:[2005 TO 2015]", sample_schema)
    assert isinstance(query, NumericRangeQuery)
    assert query.lower == 2005 and query.upper == 2015
    assert query.include_lower and query.include_upper


def test_exclusive_range_parsing(sample_schema):
    query = parse_query("year:{2005 TO *}", sample_schema)
    assert query.include_lower is False
    assert query.upper is None


# -- end-to-end search behaviour ----------------------------------------
def test_search_bare_term(searcher):
    assert _ids(searcher, "fox") == {"1", "5"}


def test_search_default_or(searcher):
    assert _ids(searcher, "fox brown") == {"1", "4", "5"}


def test_search_required_terms(searcher):
    assert _ids(searcher, "+fox +brown") == {"1"}


def test_search_must_not(searcher):
    assert _ids(searcher, "running -dog") == {"3"}


def test_search_phrase(searcher):
    assert _ids(searcher, '"quick brown"') == {"1"}


def test_search_keyword_field(searcher):
    assert _ids(searcher, "tags:animals") == {"1", "2", "4"}


def test_search_numeric_exact(searcher):
    assert _ids(searcher, "year:2010") == {"3"}


def test_search_numeric_range(searcher):
    assert _ids(searcher, "year:[2005 TO 2015]") == {"2", "3", "4"}


def test_search_grouping(searcher):
    assert _ids(searcher, "(title:fox OR title:bear)") == {"1", "4", "5"}


def test_stemmed_query_matches(searcher):
    # 'running' stems to the same root as 'runs'/'runner' in doc 3
    assert "3" in _ids(searcher, "running")


# -- errors --------------------------------------------------------------
def test_unknown_field_errors(sample_schema):
    with pytest.raises(QueryParseError):
        parse_query("nope:value", sample_schema)


def test_range_on_non_numeric_errors(sample_schema):
    with pytest.raises(QueryParseError):
        parse_query("title:[a TO b]", sample_schema)


def test_unterminated_phrase_errors(sample_schema):
    with pytest.raises(QueryParseError):
        parse_query('title:"unterminated', sample_schema)


def test_empty_query_is_match_all(sample_schema):
    from quarry.query import MatchAllQuery

    assert isinstance(parse_query("   ", sample_schema), MatchAllQuery)


def test_tokenize_operators():
    kinds = [t.kind for t in tokenize("+a -b (c OR d)")]
    assert "PLUS" in kinds and "MINUS" in kinds and "OR" in kinds and "LP" in kinds
