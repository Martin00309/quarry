"""Tests for the stored-fields document store."""

from __future__ import annotations

import pytest

from quarry.errors import CorruptIndexError
from quarry.store import StoredReader, StoredWriter, encode_stored


def test_round_trip_random_access():
    docs = [{"id": "a", "n": 1}, {"id": "b", "title": "hello"}, {"id": "c", "tags": ["x", "y"]}]
    reader = StoredReader(encode_stored(docs))
    assert len(reader) == 3
    assert reader.get(0) == docs[0]
    assert reader.get(2) == docs[2]
    assert reader.get(1) == docs[1]  # out-of-order access works


def test_all_returns_in_order():
    docs = [{"id": str(i)} for i in range(5)]
    reader = StoredReader(encode_stored(docs))
    assert reader.all() == docs


def test_writer_assigns_sequential_ids():
    writer = StoredWriter()
    assert writer.add({"id": "a"}) == 0
    assert writer.add({"id": "b"}) == 1
    reader = StoredReader(writer.serialize())
    assert reader.get(1) == {"id": "b"}


def test_empty_store():
    reader = StoredReader(encode_stored([]))
    assert len(reader) == 0


def test_out_of_range_raises():
    reader = StoredReader(encode_stored([{"id": "a"}]))
    with pytest.raises(IndexError):
        reader.get(5)


def test_corruption_is_detected():
    blob = bytearray(encode_stored([{"id": "a", "body": "some text"}]))
    blob[0] ^= 0xFF  # flip a byte in the body
    with pytest.raises(CorruptIndexError):
        StoredReader(bytes(blob))


def test_unicode_values():
    docs = [{"id": "1", "title": "café ☕ résumé"}]
    reader = StoredReader(encode_stored(docs))
    assert reader.get(0)["title"] == "café ☕ résumé"
