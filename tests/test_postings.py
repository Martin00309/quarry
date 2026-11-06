"""Tests for posting lists: encoding, cursors and the builder."""

from __future__ import annotations

import pytest

from quarry.postings import (
    NO_MORE_DOCS,
    Posting,
    PostingCursor,
    PostingsBuilder,
    decode_postings,
    encode_postings,
)


def test_encode_decode_without_positions():
    postings = [Posting(0, 2), Posting(5, 1), Posting(9, 3)]
    blob = encode_postings(postings, with_positions=False)
    decoded = decode_postings(blob, with_positions=False)
    assert decoded == postings


def test_encode_decode_with_positions():
    postings = [Posting(0, 2, (1, 4)), Posting(3, 1, (0,)), Posting(7, 3, (2, 5, 11))]
    blob = encode_postings(postings, with_positions=True)
    decoded = decode_postings(blob, with_positions=True)
    assert decoded == postings


def test_encode_rejects_unsorted():
    with pytest.raises(ValueError):
        encode_postings([Posting(5, 1), Posting(2, 1)], with_positions=False)


def test_encode_rejects_freq_position_mismatch():
    with pytest.raises(ValueError):
        encode_postings([Posting(0, 3, (1,))], with_positions=True)


def test_cursor_next_iteration():
    cursor = PostingCursor([Posting(1, 1), Posting(4, 2), Posting(8, 1)])
    assert cursor.next() == 1
    assert cursor.freq == 1
    assert cursor.next() == 4
    assert cursor.freq == 2
    assert cursor.next() == 8
    assert cursor.next() == NO_MORE_DOCS


def test_cursor_advance():
    cursor = PostingCursor([Posting(1, 1), Posting(4, 1), Posting(8, 1), Posting(12, 1)])
    assert cursor.advance(5) == 8
    assert cursor.advance(8) == 8  # already there
    assert cursor.advance(100) == NO_MORE_DOCS


def test_cursor_advance_to_exact():
    cursor = PostingCursor([Posting(2, 1), Posting(4, 1), Posting(6, 1)])
    assert cursor.advance(4) == 4
    assert cursor.positions == ()


def test_builder_with_positions():
    builder = PostingsBuilder(with_positions=True)
    builder.add("fox", 0, 3)
    builder.add("fox", 0, 7)
    builder.add("fox", 2, 1)
    builder.add("dog", 1, 0)
    assert builder.terms() == ["dog", "fox"]
    fox = builder.postings_for("fox")
    assert fox == [Posting(0, 2, (3, 7)), Posting(2, 1, (1,))]
    assert builder.total_term_freq("fox") == 3


def test_builder_without_positions():
    builder = PostingsBuilder(with_positions=False)
    for _ in range(3):
        builder.add("a", 0)
    builder.add("a", 5)
    assert builder.postings_for("a") == [Posting(0, 3), Posting(5, 1)]


def test_builder_sorts_positions():
    builder = PostingsBuilder(with_positions=True)
    builder.add("x", 0, 9)
    builder.add("x", 0, 2)
    assert builder.postings_for("x")[0].positions == (2, 9)
