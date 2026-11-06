"""Quarry: an embeddable full-text search engine.

Quarry builds an inverted index over structured documents and answers ranked
queries against it. The public surface mirrors the way search libraries are
normally used:

- Define a :class:`~quarry.schema.Schema` describing your fields.
- Open an :class:`~quarry.index.IndexWriter`, add documents, and ``commit``.
- Open a :class:`~quarry.searcher.Searcher` and run queries.

The engine is pure Python and depends only on the standard library. On-disk
indexes are made of immutable *segments* that are periodically merged, a design
borrowed from Lucene.

Example
-------
>>> from quarry import Schema, TextField, KeywordField
>>> schema = Schema(id="id", fields={
...     "id": KeywordField(stored=True),
...     "title": TextField(stored=True),
...     "body": TextField(),
... })
>>> # writer = IndexWriter.create("./index", schema)
>>> # writer.add_document({"id": "1", "title": "Hello", "body": "world"})
>>> # writer.commit()
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import (
    CorruptIndexError,
    DocumentError,
    QuarryError,
    QueryParseError,
    SchemaError,
)
from .schema import KeywordField, NumericField, Schema, StoredField, TextField

__all__ = [
    "__version__",
    "QuarryError",
    "SchemaError",
    "DocumentError",
    "QueryParseError",
    "CorruptIndexError",
    "Schema",
    "TextField",
    "KeywordField",
    "NumericField",
    "StoredField",
]
