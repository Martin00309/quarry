"""Tests for the index writer: commits, upserts, deletes and locking."""

from __future__ import annotations

import pytest

from quarry.errors import DocumentError, LockError, SchemaError
from quarry.index import IndexWriter, read_manifest, read_schema
from quarry.schema import KeywordField, Schema, TextField
from quarry.searcher import IndexReader


def _schema():
    return Schema(
        id="id",
        fields={"id": KeywordField(stored=True), "body": TextField(stored=True)},
    )


def test_create_persists_schema_and_manifest(tmp_path):
    directory = str(tmp_path / "idx")
    writer = IndexWriter.create(directory, _schema())
    writer.close()
    assert read_schema(directory).id == "id"
    assert read_manifest(directory).segments == []


def test_create_refuses_existing(tmp_path):
    directory = str(tmp_path / "idx")
    IndexWriter.create(directory, _schema()).close()
    with pytest.raises(SchemaError):
        IndexWriter.create(directory, _schema())


def test_add_and_commit(tmp_path):
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, _schema()) as writer:
        writer.add_document({"id": "1", "body": "hello world"})
        writer.add_document({"id": "2", "body": "goodbye world"})
        manifest = writer.commit()
    assert len(manifest.segments) == 1
    assert manifest.generation == 1
    reader = IndexReader(directory)
    assert reader.num_docs == 2


def test_empty_commit_is_noop(tmp_path):
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, _schema()) as writer:
        manifest = writer.commit()
    assert manifest.generation == 0
    assert manifest.segments == []


def test_upsert_replaces_document(tmp_path):
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, _schema()) as writer:
        writer.add_document({"id": "1", "body": "original text"})
        writer.commit()
        writer.add_document({"id": "1", "body": "updated text"})
        writer.commit()
    reader = IndexReader(directory)
    assert reader.num_docs == 1
    assert reader.document("1")["body"] == "updated text"


def test_delete_document(tmp_path):
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, _schema()) as writer:
        writer.add_document({"id": "1", "body": "keep me"})
        writer.add_document({"id": "2", "body": "remove me"})
        writer.commit()
        writer.delete_document("2")
        writer.commit()
    reader = IndexReader(directory)
    assert reader.num_docs == 1
    assert reader.document("2") is None
    assert reader.document("1") is not None


def test_multiple_commits_make_multiple_segments(tmp_path):
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, _schema()) as writer:
        for i in range(3):
            writer.add_document({"id": str(i), "body": f"doc {i}"})
            writer.commit()
    assert len(read_manifest(directory).segments) == 3


def test_validation_errors(tmp_path):
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, _schema()) as writer:
        with pytest.raises(DocumentError):
            writer.add_document({"body": "no id"})
        with pytest.raises(SchemaError):
            writer.add_document({"id": "1", "unknown": "field"})


def test_write_lock_is_exclusive(tmp_path):
    directory = str(tmp_path / "idx")
    writer = IndexWriter.create(directory, _schema())
    try:
        with pytest.raises(LockError):
            IndexWriter.open(directory)
    finally:
        writer.close()
    # lock released, reopen works
    IndexWriter.open(directory).close()


def test_buffered_count(tmp_path):
    directory = str(tmp_path / "idx")
    with IndexWriter.create(directory, _schema()) as writer:
        writer.add_document({"id": "1", "body": "a"})
        writer.add_document({"id": "2", "body": "b"})
        assert writer.buffered_count == 2
        writer.commit()
        assert writer.buffered_count == 0
