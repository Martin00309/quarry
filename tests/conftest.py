"""Shared fixtures: a sample schema, a sample corpus and an index factory."""

from __future__ import annotations

import pytest

from quarry.schema import KeywordField, NumericField, Schema, TextField


@pytest.fixture
def sample_schema() -> Schema:
    return Schema(
        id="id",
        fields={
            "id": KeywordField(stored=True),
            "title": TextField(stored=True, analyzer_name="english", boost=2.0),
            "body": TextField(stored=True, analyzer_name="english"),
            "tags": KeywordField(stored=True),
            "year": NumericField(stored=True, numeric_type="int"),
        },
    )


@pytest.fixture
def sample_docs():
    return [
        {
            "id": "1",
            "title": "The Quick Brown Fox",
            "body": "A quick brown fox jumps over the lazy dog",
            "tags": "animals",
            "year": 2001,
        },
        {
            "id": "2",
            "title": "Lazy Dogs Sleep",
            "body": "The lazy dog sleeps all day long in the warm sun",
            "tags": "animals",
            "year": 2005,
        },
        {
            "id": "3",
            "title": "Running Fast",
            "body": "She runs quickly and the runner keeps running every morning",
            "tags": "sports",
            "year": 2010,
        },
        {
            "id": "4",
            "title": "Brown Bears",
            "body": "Brown bears roam the forest looking for berries and fish",
            "tags": "animals",
            "year": 2015,
        },
        {
            "id": "5",
            "title": "Fox News Today",
            "body": "A short report about a fox that was seen near the quiet city park",
            "tags": "news",
            "year": 2020,
        },
    ]


@pytest.fixture
def index_factory(tmp_path, sample_schema):
    """Return a function that builds an index from docs and yields its path."""

    created = {"n": 0}

    def build(docs, schema=None, commit_each=False):
        from quarry.index import IndexWriter

        created["n"] += 1
        directory = str(tmp_path / f"index_{created['n']}")
        writer = IndexWriter.create(directory, schema or sample_schema)
        try:
            if commit_each:
                for doc in docs:
                    writer.add_document(doc)
                    writer.commit()
            else:
                writer.add_documents(docs)
                writer.commit()
        finally:
            writer.close()
        return directory

    return build


@pytest.fixture
def sample_index(index_factory, sample_docs):
    return index_factory(sample_docs)
