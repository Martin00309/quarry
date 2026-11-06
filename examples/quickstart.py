"""A runnable end-to-end example.

Builds a small in-memory index in a temporary directory, runs a few queries, and
prints the results. Run it with::

    python examples/quickstart.py
"""

from __future__ import annotations

import tempfile

from quarry import KeywordField, NumericField, Schema, TextField
from quarry.index import IndexWriter
from quarry.qparser import parse_query
from quarry.searcher import Searcher

DOCUMENTS = [
    {
        "id": "1",
        "title": "The Quick Brown Fox",
        "body": "A quick brown fox jumps over the lazy dog.",
        "tags": "animals",
        "year": 2001,
    },
    {
        "id": "2",
        "title": "Running Fast",
        "body": "She runs quickly and the runner keeps running every morning.",
        "tags": "sports",
        "year": 2010,
    },
    {
        "id": "3",
        "title": "Brown Bears of the Forest",
        "body": "Brown bears roam the forest looking for berries and fish.",
        "tags": "animals",
        "year": 2015,
    },
]


def build(directory: str) -> None:
    schema = Schema(
        id="id",
        fields={
            "id": KeywordField(stored=True),
            "title": TextField(stored=True, analyzer_name="english", boost=2.0),
            "body": TextField(stored=True, analyzer_name="english"),
            "tags": KeywordField(stored=True),
            "year": NumericField(stored=True, numeric_type="int"),
        },
    )
    with IndexWriter.create(directory, schema) as writer:
        writer.add_documents(DOCUMENTS)
        writer.commit()


def run_query(searcher: Searcher, text: str) -> None:
    query = parse_query(text, searcher.schema)
    result = searcher.search(query, top_k=5, highlight=["body"])
    print(f"\nquery: {text!r}  ({result.total_hits} hit(s))")
    for rank, hit in enumerate(result.hits, 1):
        snippet = hit.highlights.get("body", hit.fields["body"])
        print(f"  {rank}. [{hit.score:.3f}] {hit.fields['title']}")
        print(f"       {snippet}")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        build(directory)
        searcher = Searcher.open(directory)
        # 'running' matches 'runs'/'runner' in doc 2 thanks to stemming
        run_query(searcher, "running")
        # phrase query
        run_query(searcher, '"brown fox"')
        # boolean + range
        run_query(searcher, "brown AND year:[2010 TO 2020]")
        # faceting
        faceted = searcher.search(parse_query("*:*", searcher.schema), facets=["tags"])
        print("\nfacet tags:")
        for fc in faceted.facets["tags"]:
            print(f"  {fc.value}: {fc.count}")


if __name__ == "__main__":
    main()
