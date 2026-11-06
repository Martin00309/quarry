# Quarry

**An embeddable full-text search engine for the terminal.** Quarry builds an
inverted index over your documents and answers ranked queries against it — the
kind of lexical search that powers site search, log search and "find in my
notes", implemented from scratch in pure Python with no third-party
dependencies.

Think of it as a very small Lucene: a schema, an analyzer pipeline, an
append-only index made of immutable segments, BM25 ranking, a compact query
language, and a CLI to drive it all.

> Status: alpha. The on-disk format is versioned but not yet stable across
> releases.

## Why Quarry

- **Real information retrieval, not a `LIKE '%term%'` wrapper.** Tokenization,
  stemming, an inverted index with positional postings, BM25 scoring and phrase
  queries — the actual machinery of a search engine.
- **Embeddable.** Use it as a library from Python, or as a CLI over JSONL.
- **No dependencies.** Standard library only, so it runs anywhere Python does.
- **Inspectable.** Every layer — analysis, postings, segments, scoring — has a
  small, readable module and a way to look inside it from the command line.

## Install

```bash
git clone https://github.com/Martin00309/quarry
cd quarry
pip install -e .
```

Quarry targets Python 3.9+.

## Quickstart (CLI)

Describe your fields in a schema file:

```json
{
  "id": "id",
  "fields": {
    "id":    {"kind": "keyword", "stored": true},
    "title": {"kind": "text", "stored": true, "analyzer": "english", "boost": 2.0},
    "body":  {"kind": "text", "stored": true, "analyzer": "english"},
    "tags":  {"kind": "keyword", "stored": true},
    "year":  {"kind": "numeric", "stored": true, "numeric_type": "int"}
  }
}
```

Create the index, feed it documents as JSONL, and search:

```bash
quarry init ./notes --schema schema.json

cat <<'EOF' | quarry add ./notes --file -
{"id": "1", "title": "The Quick Brown Fox", "body": "a quick brown fox jumps over the lazy dog", "tags": "animals", "year": 2001}
{"id": "2", "title": "Running Fast", "body": "she runs quickly and the runner keeps running", "tags": "sports", "year": 2010}
EOF

quarry search ./notes 'quick +fox'
quarry search ./notes '"quick brown"' --highlight body
quarry search ./notes 'running' --explain          # 'running' matches 'runs' via stemming
quarry search ./notes 'year:[2000 TO 2005]' --sort year:desc
quarry search ./notes 'tags:animals' --facet tags --json
```

## Quickstart (library)

```python
from quarry import Schema, TextField, KeywordField, NumericField
from quarry.index import IndexWriter
from quarry.searcher import Searcher
from quarry.qparser import parse_query

schema = Schema(id="id", fields={
    "id":    KeywordField(stored=True),
    "title": TextField(stored=True, analyzer_name="english", boost=2.0),
    "body":  TextField(stored=True, analyzer_name="english"),
})

with IndexWriter.create("./index", schema) as writer:
    writer.add_document({"id": "1", "title": "Hello world", "body": "a tiny example"})
    writer.commit()

searcher = Searcher.open("./index")
query = parse_query("hello", searcher.schema)
for hit in searcher.search(query, top_k=10).hits:
    print(hit.score, hit.doc_id, hit.fields["title"])
```

## The query language

| Syntax | Meaning |
| --- | --- |
| `quick brown` | either term (default operator is OR) |
| `+quick +brown` | both terms required |
| `title:quick` | term in a specific field |
| `-draft` | term must not appear |
| `"quick brown"` | exact phrase |
| `"quick brown"~2` | phrase with a slop of 2 |
| `quick*` | prefix query |
| `qu?ck` | wildcard (`?` one char, `*` any run) |
| `price:[10 TO 50]` | inclusive numeric range (`{ }` for exclusive) |
| `year:{2000 TO *}` | open-ended range |
| `(a OR b) AND c` | grouping with explicit `AND`/`OR`/`NOT` |

Bare terms search every text field. Terms run through the target field's
analyzer, so with the English analyzer a search for `Running` matches an indexed
`run`.

## Fields and analyzers

- **`TextField`** — free text, analyzed into terms and ranked. Positional by
  default, so it can answer phrase queries.
- **`KeywordField`** — an un-analyzed string matched exactly; ideal for ids,
  tags and enums, and usable for faceting.
- **`NumericField`** — int/float stored in a per-segment column; powers range
  filters and sorting.
- **`StoredField`** — retrieved with hits but never searched.

Built-in analyzers: `simple`, `standard` (stop-words removed), `english`
(stop-words + Porter stemming), `whitespace`, `keyword`. Inspect any of them:

```bash
quarry analyze "The cats are running quickly" --analyzer english
#   1  cat  [4:8]
#   3  run  [13:20]
#   4  quickli  [21:28]
```

## How it works

Quarry is organized as a stack of small layers, each in its own module:

```
analysis   tokenizer + filters (lowercase, stop-words, Porter stemmer)
schema     field types, (de)serialized with the index
postings   delta + varint compressed posting lists with positions
segment    immutable on-disk segment (term dict, postings, store, norms, numerics)
index      IndexWriter: buffering, upserts, deletes, atomic commits, a manifest
merge      tiered merge policy + deletion-reclaiming compaction
similarity BM25 (default) and TF-IDF with collection statistics
query      document-at-a-time scorers (term, boolean, phrase, prefix, range)
qparser    the query language above, compiled to the query AST
searcher   multi-segment search, sorting, faceting, highlighting, explain
```

An index is an **append-only set of immutable segments**. Each commit writes a
new segment and atomically publishes a new manifest; deletions are tombstones in
a side file; merging combines segments and drops deleted documents. A reader
sees a consistent snapshot and never blocks a writer. The on-disk layout is
documented in [docs/format.md](docs/format.md).

## Development

```bash
pip install -e .
pip install pytest ruff
pytest          # 250+ tests
ruff check .
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what's planned (proximity scoring, fuzzy terms,
multi-valued fields, persisted term-dictionary indexes, concurrent writers).

## License

MIT — see [LICENSE](LICENSE).
