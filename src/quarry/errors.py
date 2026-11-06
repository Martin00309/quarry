"""Exception hierarchy for Quarry.

All errors raised deliberately by the library derive from :class:`QuarryError`,
so callers can catch everything the engine throws with a single ``except`` while
still being able to distinguish the specific failure categories below.
"""

from __future__ import annotations


class QuarryError(Exception):
    """Base class for every error raised by Quarry."""


class SchemaError(QuarryError):
    """A schema is malformed, or a document does not match the schema.

    Raised when a field type is misconfigured, when a document references an
    unknown field, or when a required field (such as the primary key) is
    missing or has the wrong type.
    """


class DocumentError(QuarryError):
    """A document could not be indexed or decoded.

    Raised for problems specific to a single document: a value that cannot be
    analyzed for its field, a duplicate primary key within a single commit, or
    a stored document that cannot be deserialized.
    """


class QueryParseError(QuarryError):
    """A query string could not be parsed.

    Carries the offending position in the source string when known, so a CLI
    can point at the character that tripped the parser.
    """

    def __init__(self, message: str, position: int | None = None) -> None:
        self.position = position
        if position is not None:
            message = f"{message} (at position {position})"
        super().__init__(message)


class QueryError(QuarryError):
    """A query is syntactically fine but cannot be executed.

    For example, a range query against a field that is not numeric, or a phrase
    query against a field indexed without positions.
    """


class CorruptIndexError(QuarryError):
    """On-disk index data failed an integrity check.

    Raised when a checksum does not match, when a segment references data past
    the end of a file, or when the manifest points at a segment that is missing.
    """


class LockError(QuarryError):
    """The index is already locked by another writer.

    Quarry allows a single writer per index at a time; a second writer that
    finds the lock file held raises this rather than corrupting the index.
    """
