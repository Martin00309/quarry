# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Text analysis pipeline: regex/whitespace/keyword tokenizers and lowercase,
  accent-folding, stop-word, length and Porter-stemmer token filters, exposed
  as the `simple`, `standard`, `english`, `whitespace` and `keyword` analyzers.
- Schema with `TextField`, `KeywordField`, `NumericField` and `StoredField`,
  serialized alongside the index.
- Inverted index with delta + varint compressed positional posting lists.
- Immutable, checksummed on-disk segments (term dictionary, postings, stored
  fields, length norms, numeric columns) with a deletion side-file.
- `IndexWriter` with buffered adds, upsert-by-id, deletes, atomic commits and a
  generation-counted manifest, guarded by a write lock.
- Tiered merge policy with deletion-reclaiming compaction.
- BM25 (default) and TF-IDF similarity with collection statistics and score
  explanations.
- Query model and document-at-a-time scorers: term, boolean (must/should/
  must-not with minimum-should-match), phrase (with slop), prefix, wildcard,
  numeric range and match-all.
- A Lucene-style query-string parser.
- Multi-segment searcher with sorting, faceting, highlighting and `explain`.
- `quarry` CLI: `init`, `add`, `delete`, `search`, `merge`, `stats`, `inspect`
  and `analyze`, with human-readable and `--json` output.

[Unreleased]: https://github.com/Martin00309/quarry/commits/main
