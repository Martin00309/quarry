"""Tests for segment merging and the merge policy."""

from __future__ import annotations

import os

from quarry.index import IndexWriter, read_manifest
from quarry.merge import MergePolicy, maybe_merge, merge_segments
from quarry.qparser import parse_query
from quarry.searcher import Searcher
from quarry.segment import SegmentInfo


def test_merge_all_segments_preserves_results(index_factory, sample_docs):
    directory = index_factory(sample_docs, commit_each=True)
    assert len(read_manifest(directory).segments) == 5

    before = _search_ids(directory, "fox brown")
    with IndexWriter.open(directory) as writer:
        names = [s.name for s in writer.manifest.segments]
        merge_segments(writer, names)
    assert len(read_manifest(directory).segments) == 1

    after = _search_ids(directory, "fox brown")
    assert before == after == {"1", "4", "5"}


def _search_ids(directory, text):
    searcher = Searcher.open(directory)
    return {h.doc_id for h in searcher.search(parse_query(text, searcher.schema)).hits}


def test_merge_removes_old_segment_files(index_factory, sample_docs):
    directory = index_factory(sample_docs, commit_each=True)
    with IndexWriter.open(directory) as writer:
        names = [s.name for s in writer.manifest.segments]
        merge_segments(writer, names)
    for name in names:
        assert not os.path.exists(os.path.join(directory, f"{name}.seg"))


def test_merge_reclaims_deletions(index_factory, sample_docs):
    directory = index_factory(sample_docs)  # one segment, 5 docs
    with IndexWriter.open(directory) as writer:
        writer.delete_document("2")
        writer.delete_document("4")
        writer.commit()
        name = writer.manifest.segments[0].name
        merge_segments(writer, [name])

    manifest = read_manifest(directory)
    assert len(manifest.segments) == 1
    assert manifest.segments[0].doc_count == 3
    assert manifest.segments[0].num_deleted == 0
    reader = Searcher.open(directory).reader
    assert reader.num_docs == 3
    assert reader.document("2") is None
    assert reader.document("1") is not None


def test_merge_of_all_deleted_drops_segment(index_factory):
    directory = index_factory([{"id": "1", "title": "x", "body": "y", "tags": "t", "year": 1}])
    with IndexWriter.open(directory) as writer:
        writer.delete_document("1")
        writer.commit()
        name = writer.manifest.segments[0].name
        result = merge_segments(writer, [name])
    assert result is None
    assert read_manifest(directory).segments == []


def test_maybe_merge_reduces_segment_count(index_factory, sample_docs):
    directory = index_factory(sample_docs, commit_each=True)  # 5 segments
    with IndexWriter.open(directory) as writer:
        maybe_merge(writer, MergePolicy(max_segments=2, max_merge_at_once=10))
    assert len(read_manifest(directory).segments) <= 2


# -- policy unit tests ---------------------------------------------------
def test_policy_merges_when_too_many_segments():
    policy = MergePolicy(max_segments=8, max_merge_at_once=10)
    infos = [SegmentInfo(f"_{i}", doc_count=10) for i in range(9)]
    chosen = policy.find_merge(infos)
    assert chosen is not None
    assert len(chosen) == 9


def test_policy_limits_merge_at_once():
    policy = MergePolicy(max_segments=4, max_merge_at_once=3)
    infos = [SegmentInfo(f"_{i}", doc_count=i + 1) for i in range(10)]
    chosen = policy.find_merge(infos)
    assert len(chosen) == 3
    # smallest segments chosen
    assert set(chosen) == {"_0", "_1", "_2"}


def test_policy_merges_heavily_deleted_segment():
    policy = MergePolicy(max_segments=8, max_deleted_ratio=0.3)
    infos = [SegmentInfo("_0", doc_count=10, num_deleted=5)]
    assert policy.find_merge(infos) == ["_0"]


def test_policy_no_merge_needed():
    policy = MergePolicy(max_segments=8)
    infos = [SegmentInfo("_0", doc_count=10), SegmentInfo("_1", doc_count=10)]
    assert policy.find_merge(infos) is None
