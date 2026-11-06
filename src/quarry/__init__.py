"""Quarry: an embeddable full-text search engine."""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import (
    CorruptIndexError,
    DocumentError,
    QuarryError,
    QueryParseError,
    SchemaError,
)

__all__ = [
    "__version__",
    "QuarryError",
    "SchemaError",
    "DocumentError",
    "QueryParseError",
    "CorruptIndexError",
]
