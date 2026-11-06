"""Tests for schema and field definitions."""

from __future__ import annotations

import pytest

from quarry.errors import SchemaError
from quarry.schema import KeywordField, NumericField, Schema, StoredField, TextField


def test_schema_requires_id_field():
    with pytest.raises(SchemaError):
        Schema(fields={"title": TextField()}, id="id")


def test_id_field_must_be_stored_keyword():
    with pytest.raises(SchemaError):
        Schema(fields={"id": TextField(stored=True)}, id="id")
    with pytest.raises(SchemaError):
        Schema(fields={"id": KeywordField(stored=False)}, id="id")


def test_field_must_be_stored_or_indexed():
    with pytest.raises(SchemaError):
        Schema(
            fields={"id": KeywordField(stored=True), "x": TextField(stored=False, indexed=False)},
            id="id",
        )


def test_invalid_field_name_rejected():
    with pytest.raises(SchemaError):
        Schema(fields={"id": KeywordField(stored=True), "bad name": TextField()}, id="id")


def test_field_categorization(sample_schema):
    assert set(sample_schema.text_fields()) >= {"title", "body"}
    assert sample_schema.numeric_fields() == ["year"]
    assert set(sample_schema.stored_fields()) == {"id", "title", "body", "tags", "year"}
    # default search excludes keyword/id fields
    assert set(sample_schema.default_search_fields()) == {"title", "body"}


def test_numeric_field_type_validation():
    with pytest.raises(SchemaError):
        NumericField(numeric_type="decimal")


def test_numeric_coerce():
    f = NumericField(numeric_type="int")
    assert f.coerce("42") == 42
    with pytest.raises(SchemaError):
        f.coerce("not a number")


def test_stored_field_is_forced_stored():
    f = StoredField()
    assert f.stored is True
    assert f.indexed is False


def test_schema_serialization_round_trip(sample_schema):
    data = sample_schema.to_dict()
    restored = Schema.from_dict(data)
    assert restored.id == sample_schema.id
    assert restored.to_dict() == data
    assert isinstance(restored.field("title"), TextField)
    assert restored.field("title").analyzer_name == "english"
    assert restored.field("title").boost == 2.0
    assert restored.field("year").numeric_type == "int"


def test_keyword_lowercase_option():
    lowered = KeywordField(stored=True, lowercase=True)
    assert lowered.analyzer().terms("HELLO World") == ["hello world"]
    exact = KeywordField(stored=True, lowercase=False)
    assert exact.analyzer().terms("HELLO World") == ["HELLO World"]
