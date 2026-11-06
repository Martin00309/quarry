"""Schema and field definitions.

A :class:`Schema` names the fields a document may contain and how each is
treated: whether it is analyzed into searchable terms, whether its original
value is stored for retrieval, and (for numeric fields) whether it supports
range filtering and sorting.

Field analyzers are referenced *by name* so a schema can be serialized to JSON
and persisted inside the index. Custom analyzers are supported by registering
them with :func:`quarry.analysis.register_analyzer` and naming them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .analysis import Analyzer, get_analyzer
from .errors import SchemaError


@dataclass
class Field:
    """Base class for all field types.

    Parameters
    ----------
    stored:
        Keep the original value so it can be returned with search hits.
    indexed:
        Build searchable index structures (terms or numeric columns) for this
        field. A field that is neither stored nor indexed is pointless and the
        schema rejects it.
    boost:
        Per-field score multiplier applied at query time.
    """

    stored: bool = False
    indexed: bool = True
    boost: float = 1.0

    #: Short tag used in (de)serialization; overridden by subclasses.
    kind: str = "field"

    # -- capability flags, overridden where appropriate ------------------
    @property
    def is_text(self) -> bool:
        return False

    @property
    def is_numeric(self) -> bool:
        return False

    @property
    def has_positions(self) -> bool:
        return False

    def analyzer(self) -> Optional[Analyzer]:
        """Return the analyzer for this field, or ``None`` if not analyzed."""
        return None

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "stored": self.stored,
            "indexed": self.indexed,
            "boost": self.boost,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Field":
        kind = data.get("kind")
        cls = _FIELD_KINDS.get(kind)
        if cls is None:
            raise SchemaError(f"unknown field kind: {kind!r}")
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Field":  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class TextField(Field):
    """Free text, analyzed into terms and ranked with the similarity model.

    ``positions`` must be on for the field to answer phrase queries. Turning it
    off makes the index smaller when only bag-of-words matching is needed.
    """

    analyzer_name: str = "standard"
    positions: bool = True
    kind: str = "text"

    @property
    def is_text(self) -> bool:
        return True

    @property
    def has_positions(self) -> bool:
        return self.positions

    def analyzer(self) -> Analyzer:
        return get_analyzer(self.analyzer_name)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(analyzer=self.analyzer_name, positions=self.positions)
        return data

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "TextField":
        return cls(
            stored=data.get("stored", False),
            indexed=data.get("indexed", True),
            boost=data.get("boost", 1.0),
            analyzer_name=data.get("analyzer", "standard"),
            positions=data.get("positions", True),
        )


@dataclass
class KeywordField(Field):
    """An un-analyzed string: matched exactly, and usable for faceting.

    The whole value becomes a single term, so ``KeywordField`` is right for
    ids, tags, enum-like values and file paths.
    """

    lowercase: bool = False
    kind: str = "keyword"

    @property
    def is_text(self) -> bool:
        return True

    def analyzer(self) -> Analyzer:
        from .analysis import KeywordTokenizer, lowercase_filter

        filters = [lowercase_filter] if self.lowercase else []
        return Analyzer(KeywordTokenizer(), filters)

    def normalize(self, value: str) -> str:
        return value.lower() if self.lowercase else value

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(lowercase=self.lowercase)
        return data

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "KeywordField":
        return cls(
            stored=data.get("stored", False),
            indexed=data.get("indexed", True),
            boost=data.get("boost", 1.0),
            lowercase=data.get("lowercase", False),
        )


@dataclass
class NumericField(Field):
    """An int or float value, stored in a per-segment column.

    Numeric fields are not tokenized; instead they power range filters
    (``price:[10 TO 50]``) and result sorting. ``numeric_type`` is either
    ``"int"`` or ``"float"``.
    """

    numeric_type: str = "float"
    kind: str = "numeric"

    def __post_init__(self) -> None:
        if self.numeric_type not in ("int", "float"):
            raise SchemaError(f"numeric_type must be 'int' or 'float', got {self.numeric_type!r}")

    @property
    def is_numeric(self) -> bool:
        return True

    def coerce(self, value: Any) -> float:
        try:
            return int(value) if self.numeric_type == "int" else float(value)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"value {value!r} is not a valid {self.numeric_type}") from exc

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(numeric_type=self.numeric_type)
        return data

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "NumericField":
        return cls(
            stored=data.get("stored", False),
            indexed=data.get("indexed", True),
            boost=data.get("boost", 1.0),
            numeric_type=data.get("numeric_type", "float"),
        )


@dataclass
class StoredField(Field):
    """A value that is only retrieved, never searched (e.g. a raw blob)."""

    indexed: bool = False
    kind: str = "stored"

    def __post_init__(self) -> None:
        # A stored-only field must at least be stored, or it does nothing.
        object.__setattr__(self, "stored", True)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "StoredField":
        return cls(boost=data.get("boost", 1.0))


_FIELD_KINDS = {
    "text": TextField,
    "keyword": KeywordField,
    "numeric": NumericField,
    "stored": StoredField,
}


class Schema:
    """The set of fields a document may have, plus the primary-key field.

    The ``id`` field names a :class:`KeywordField` that uniquely identifies a
    document. Re-adding a document with the same id replaces the old one, and
    deletes are expressed in terms of it.
    """

    def __init__(self, fields: Dict[str, Field], id: str = "id") -> None:
        if not fields:
            raise SchemaError("schema must define at least one field")
        for name, field in fields.items():
            if not name or not name.isidentifier():
                raise SchemaError(f"invalid field name: {name!r}")
            if not isinstance(field, Field):
                raise SchemaError(f"field {name!r} is not a Field instance")
            if not field.stored and not field.indexed:
                raise SchemaError(f"field {name!r} is neither stored nor indexed")
        if id not in fields:
            raise SchemaError(f"id field {id!r} is not defined in the schema")
        id_field = fields[id]
        if not isinstance(id_field, KeywordField):
            raise SchemaError(f"id field {id!r} must be a KeywordField")
        if not id_field.stored:
            raise SchemaError(f"id field {id!r} must be stored")

        self.fields = dict(fields)
        self.id = id

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    def field(self, name: str) -> Field:
        try:
            return self.fields[name]
        except KeyError:
            raise SchemaError(f"unknown field: {name!r}") from None

    def names(self) -> List[str]:
        return list(self.fields)

    def text_fields(self) -> List[str]:
        return [n for n, f in self.fields.items() if f.is_text and f.indexed]

    def numeric_fields(self) -> List[str]:
        return [n for n, f in self.fields.items() if f.is_numeric and f.indexed]

    def stored_fields(self) -> List[str]:
        return [n for n, f in self.fields.items() if f.stored]

    def default_search_fields(self) -> List[str]:
        """Free-text fields a bare (un-fielded) query term should search.

        Only :class:`TextField` fields qualify; keyword fields (including the
        id field) are matched exactly and are not part of default search.
        """
        return [n for n, f in self.fields.items() if isinstance(f, TextField) and f.indexed]

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "fields": {n: f.to_dict() for n, f in self.fields.items()}}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Schema":
        fields = {n: Field.from_dict(fd) for n, fd in data["fields"].items()}
        return Schema(fields=fields, id=data.get("id", "id"))
