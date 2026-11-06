"""Tests for snippet highlighting."""

from __future__ import annotations

from quarry.highlight import Highlighter
from quarry.schema import KeywordField, Schema, TextField


def _schema():
    return Schema(
        id="id",
        fields={
            "id": KeywordField(stored=True),
            "body": TextField(stored=True, analyzer_name="english"),
        },
    )


def test_marks_matching_terms():
    hl = Highlighter(_schema())
    snippet = hl.highlight("body", "A quick brown fox", [("body", "brown")])
    assert "<b>brown</b>" in snippet


def test_matches_via_stemming_but_marks_surface_form():
    hl = Highlighter(_schema())
    # query term 'run' (stemmed) should highlight the surface word 'running'
    snippet = hl.highlight("body", "She was running fast", [("body", "run")])
    assert "<b>running</b>" in snippet


def test_no_match_returns_none():
    hl = Highlighter(_schema())
    assert hl.highlight("body", "nothing to see here", [("body", "absent")]) is None
    # terms for a different field are ignored
    assert hl.highlight("body", "a quick fox", [("title", "fox")]) is None


def test_long_text_is_truncated_with_ellipsis():
    hl = Highlighter(_schema(), max_chars=30)
    text = "word " * 40 + "target " + "word " * 40
    snippet = hl.highlight("body", text, [("body", "target")])
    assert "<b>target</b>" in snippet
    assert "…" in snippet
    assert len(snippet) < len(text)


def test_multiple_occurrences_marked():
    hl = Highlighter(_schema())
    snippet = hl.highlight("body", "fox fox fox", [("body", "fox")])
    assert snippet.count("<b>fox</b>") == 3


def test_custom_tags():
    hl = Highlighter(_schema(), pre_tag="[", post_tag="]")
    snippet = hl.highlight("body", "a quick brown fox", [("body", "quick")])
    assert "[quick]" in snippet
