"""Tests for the command-line interface."""

from __future__ import annotations

import json

import pytest

from quarry.cli import EXIT_NOT_FOUND, EXIT_QUERY, main
from quarry.schema import KeywordField, NumericField, Schema, TextField


@pytest.fixture
def schema_file(tmp_path):
    schema = Schema(
        id="id",
        fields={
            "id": KeywordField(stored=True),
            "title": TextField(stored=True, analyzer_name="english"),
            "body": TextField(stored=True, analyzer_name="english"),
            "tags": KeywordField(stored=True),
            "year": NumericField(stored=True, numeric_type="int"),
        },
    )
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema.to_dict()))
    return str(path)


@pytest.fixture
def docs_file(tmp_path, sample_docs):
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in sample_docs))
    return str(path)


@pytest.fixture
def index_dir(tmp_path, schema_file, docs_file):
    directory = str(tmp_path / "idx")
    assert main(["init", directory, "--schema", schema_file]) == 0
    assert main(["add", directory, "--file", docs_file]) == 0
    return directory


def test_init_and_add(tmp_path, schema_file, docs_file, capsys):
    directory = str(tmp_path / "idx")
    assert main(["init", directory, "--schema", schema_file]) == 0
    assert main(["add", directory, "--file", docs_file]) == 0
    out = capsys.readouterr().out
    assert "indexed 5" in out


def test_search_human(index_dir, capsys):
    assert main(["search", index_dir, "fox"]) == 0
    out = capsys.readouterr().out
    assert "hit(s)" in out
    assert "Fox" in out or "fox" in out


def test_search_json(index_dir, capsys):
    assert main(["--json", "search", index_dir, "fox"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_hits"] == 2
    ids = {hit["id"] for hit in payload["hits"]}
    assert ids == {"1", "5"}


def test_search_with_highlight(index_dir, capsys):
    assert main(["--json", "search", index_dir, "body:fox", "--highlight", "body"]) == 0
    payload = json.loads(capsys.readouterr().out)
    hit = next(h for h in payload["hits"] if h["id"] == "1")
    assert "<b>fox</b>" in hit["highlights"]["body"]


def test_search_sorted(index_dir, capsys):
    assert main(["--json", "search", index_dir, "*:*", "--sort", "year:asc", "--top-k", "5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [h["id"] for h in payload["hits"]] == ["1", "2", "3", "4", "5"]


def test_search_facets(index_dir, capsys):
    assert main(["--json", "search", index_dir, "*:*", "--facet", "tags"]) == 0
    payload = json.loads(capsys.readouterr().out)
    counts = {fc["value"]: fc["count"] for fc in payload["facets"]["tags"]}
    assert counts == {"animals": 3, "sports": 1, "news": 1}


def test_delete(index_dir, capsys):
    assert main(["delete", index_dir, "--id", "1"]) == 0
    capsys.readouterr()  # discard the delete command's output
    assert main(["--json", "search", index_dir, "body:fox"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {h["id"] for h in payload["hits"]} == {"5"}


def test_stats(index_dir, capsys):
    assert main(["--json", "stats", index_dir]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["num_docs"] == 5
    assert "title" in payload["terms_per_field"]


def test_analyze(capsys):
    argv = ["--json", "analyze", "The", "cats", "are", "running", "--analyzer", "english"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    texts = [t["text"] for t in payload]
    assert "cat" in texts and "run" in texts


def test_inspect_terms(index_dir, capsys):
    assert main(["inspect", index_dir, "--terms", "title"]) == 0
    out = capsys.readouterr().out
    assert "fox" in out


def test_inspect_postings(index_dir, capsys):
    assert main(["inspect", index_dir, "--term-field", "body", "--term", "fox"]) == 0
    out = capsys.readouterr().out
    assert "freq=" in out


def test_merge_command(index_dir, capsys):
    assert main(["merge", index_dir, "--force"]) == 0
    capsys.readouterr()  # discard the merge command's output
    assert main(["--json", "stats", index_dir]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["segments"]) == 1


def test_query_error_exit_code(index_dir, capsys):
    assert main(["search", index_dir, "unknownfield:value"]) == EXIT_QUERY


def test_not_found_exit_code(tmp_path):
    assert main(["stats", str(tmp_path / "missing")]) == EXIT_NOT_FOUND
