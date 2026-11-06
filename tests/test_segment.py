"""Tests for the on-disk segment: write, read, deletions and integrity."""

from __future__ import annotations

import os

import pytest

from quarry.errors import CorruptIndexError
from quarry.postings import Posting, PostingsBuilder
from quarry.schema import KeywordField, NumericField, Schema, TextField
from quarry.segment import LiveDocs, SegmentData, SegmentReader, write_segment


def _schema():
    return Schema(
        id="id",
        fields={
            "id": KeywordField(stored=True),
            "body": TextField(stored=True),
            "year": NumericField(stored=True, numeric_type="int"),
        },
    )


def _build(tmp_path):
    schema = _schema()
    data = SegmentData(schema=schema, doc_count=2)
    pb = PostingsBuilder(with_positions=True)
    for pos, term in enumerate(["quick", "brown", "fox"]):
        pb.add(term, 0, pos)
    for pos, term in enumerate(["lazy", "brown", "dog"]):
        pb.add(term, 1, pos)
    data.postings["body"] = pb
    data.norms["body"] = {0: 3, 1: 3}
    data.field_sum_length["body"] = 6
    data.field_doc_count["body"] = 2
    data.numeric["year"] = {0: 2001.0, 1: 2005.0}
    data.stored = [
        {"id": "d0", "body": "quick brown fox", "year": 2001},
        {"id": "d1", "body": "lazy brown dog", "year": 2005},
    ]
    data.ids = ["d0", "d1"]
    write_segment(str(tmp_path), "_0", data)
    return SegmentReader(str(tmp_path), "_0", schema)


def test_segment_basic_reads(tmp_path):
    reader = _build(tmp_path)
    assert reader.doc_count == 2
    assert reader.doc_freq("body", "brown") == 2
    assert reader.doc_freq("body", "fox") == 1
    assert reader.doc_freq("body", "missing") == 0


def test_segment_postings(tmp_path):
    reader = _build(tmp_path)
    assert reader.postings("body", "brown") == [Posting(0, 1, (1,)), Posting(1, 1, (1,))]
    assert reader.postings("body", "fox") == [Posting(0, 1, (2,))]
    assert reader.postings("body", "nope") == []


def test_segment_norms_and_stats(tmp_path):
    reader = _build(tmp_path)
    assert reader.field_length("body", 0) == 3
    assert reader.field_sum_length("body") == 6
    assert reader.field_doc_count("body") == 2


def test_segment_numeric(tmp_path):
    reader = _build(tmp_path)
    assert reader.numeric_value("year", 0) == 2001.0
    assert reader.numeric_value("year", 1) == 2005.0


def test_segment_stored_and_ids(tmp_path):
    reader = _build(tmp_path)
    assert reader.stored(0)["body"] == "quick brown fox"
    assert reader.id_of(1) == "d1"
    assert reader.local_of_id("d0") == 0
    assert reader.local_of_id("ghost") is None


def test_segment_deletions_persist(tmp_path):
    reader = _build(tmp_path)
    assert reader.delete_by_id("d0") is True
    assert reader.delete_by_id("ghost") is False
    assert reader.num_deleted == 1
    assert reader.num_live == 1
    reader.flush_deletions()

    reopened = SegmentReader(str(tmp_path), "_0", _schema())
    assert reopened.is_deleted(0) is True
    assert reopened.num_live == 1


def test_segment_detects_corruption(tmp_path):
    _build(tmp_path)
    path = os.path.join(str(tmp_path), "_0.seg")
    blob = bytearray(open(path, "rb").read())
    blob[10] ^= 0xFF
    with open(path, "wb") as fh:
        fh.write(blob)
    with pytest.raises(CorruptIndexError):
        SegmentReader(str(tmp_path), "_0", _schema())


def test_live_docs_round_trip():
    live = LiveDocs(doc_count=10, deleted={2, 5})
    restored = LiveDocs.load(live.serialize())
    assert restored.doc_count == 10
    assert restored.is_deleted(2) and restored.is_deleted(5)
    assert not restored.is_deleted(0)
    assert restored.live_ids() == [0, 1, 3, 4, 6, 7, 8, 9]


def test_live_docs_rejects_bad_magic():
    with pytest.raises(CorruptIndexError):
        LiveDocs.load(b"XXXX garbage")
