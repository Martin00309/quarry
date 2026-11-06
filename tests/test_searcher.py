"""Integration tests for the searcher: ranking, sorting, facets, highlight."""

from __future__ import annotations

import pytest

from quarry.index import IndexWriter
from quarry.qparser import parse_query
from quarry.query import MatchAllQuery, TermQuery
from quarry.searcher import Searcher


@pytest.fixture
def searcher(sample_index):
    return Searcher.open(sample_index)


def test_results_sorted_by_score(searcher):
    result = searcher.search(parse_query("fox brown dog", searcher.schema))
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert result.max_score == scores[0]


def test_hits_carry_stored_fields(searcher):
    result = searcher.search(TermQuery("body", "fox"))
    hit = next(h for h in result.hits if h.doc_id == "1")
    assert hit.fields["title"] == "The Quick Brown Fox"
    assert hit.fields["year"] == 2001


def test_sort_by_numeric_ascending(searcher):
    result = searcher.search(MatchAllQuery(), sort=("year", False))
    assert [h.doc_id for h in result.hits] == ["1", "2", "3", "4", "5"]


def test_sort_by_numeric_descending(searcher):
    result = searcher.search(MatchAllQuery(), sort=("year", True))
    assert [h.doc_id for h in result.hits] == ["5", "4", "3", "2", "1"]


def test_faceting(searcher):
    result = searcher.search(MatchAllQuery(), facets=["tags"])
    counts = {fc.value: fc.count for fc in result.facets["tags"]}
    assert counts == {"animals": 3, "sports": 1, "news": 1}


def test_facet_on_unstored_field_errors(tmp_path, sample_docs):
    from quarry.errors import QueryError
    from quarry.schema import KeywordField, Schema, TextField

    schema = Schema(
        id="id",
        fields={
            "id": KeywordField(stored=True),
            "body": TextField(stored=True),
            "tags": KeywordField(stored=False),
        },
    )
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, schema) as writer:
        writer.add_document({"id": "1", "body": "x", "tags": "a"})
        writer.commit()
    searcher = Searcher.open(directory)
    with pytest.raises(QueryError):
        searcher.search(MatchAllQuery(), facets=["tags"])


def test_highlighting(searcher):
    result = searcher.search(TermQuery("body", "fox"), highlight=["body"])
    hit = next(h for h in result.hits if h.doc_id == "1")
    assert "<b>fox</b>" in hit.highlights["body"]


def test_explain_matches_score(searcher):
    query = TermQuery("body", "fox")
    result = searcher.search(query)
    hit = next(h for h in result.hits if h.doc_id == "1")
    explanation = searcher.explain(query, "1")
    assert explanation.value == pytest.approx(hit.score)
    assert "1" in explanation.description


def test_explain_non_match(searcher):
    explanation = searcher.explain(TermQuery("body", "fox"), "2")
    assert explanation.value == 0.0


def test_tfidf_similarity(sample_index):
    searcher = Searcher.open(sample_index, similarity="tfidf")
    result = searcher.search(TermQuery("body", "fox"))
    assert {h.doc_id for h in result.hits} == {"1", "5"}
    assert all(h.score > 0 for h in result.hits)


def test_multi_segment_matches_single_segment(index_factory, sample_docs):
    single = Searcher.open(index_factory(sample_docs))
    multi = Searcher.open(index_factory(sample_docs, commit_each=True))
    assert len(multi.reader.segments) == 5
    query_text = "fox brown"
    single_ids = {h.doc_id for h in single.search(parse_query(query_text, single.schema)).hits}
    multi_ids = {h.doc_id for h in multi.search(parse_query(query_text, multi.schema)).hits}
    assert single_ids == multi_ids == {"1", "4", "5"}


def test_deletion_reflected_in_search(sample_index):
    with IndexWriter.open(sample_index) as writer:
        writer.delete_document("1")
        writer.commit()
    searcher = Searcher.open(sample_index)
    assert {h.doc_id for h in searcher.search(TermQuery("body", "fox")).hits} == {"5"}


def test_num_docs_and_max_doc(searcher):
    assert searcher.reader.num_docs == 5
    assert searcher.reader.max_doc == 5
