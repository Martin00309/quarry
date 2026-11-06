# Roadmap

Quarry is usable today for indexing and ranked retrieval, but there is plenty of
work ahead. Rough order of priority.

## Near term
- [ ] **Persisted term-dictionary index.** Terms are read fully into memory on
      open; add an on-disk sorted index (or FST) so large segments open lazily.
- [ ] **Block-skip postings.** Store skip pointers in the postings so
      `advance()` can jump without decoding a whole list.
- [ ] **Fuzzy term queries.** `term~1` edit-distance matching against the term
      dictionary (currently `~N` on a single term is parsed but treated as the
      term itself).
- [ ] **Proximity scoring.** Score sloppy phrases by how tight the match is,
      not just whether it matched.

## Mid term
- [ ] **Multi-valued fields.** Let a document hold a list of values for a field
      (e.g. several tags) with correct positions and facets.
- [ ] **Term-range queries on text fields**, not just numeric ranges.
- [ ] **Configurable merge scheduling**, including a background/auto-merge hook
      on the writer rather than an explicit call.
- [ ] **Synonym and n-gram token filters.**
- [ ] **Stored-field compression** for the document store.

## Longer term
- [ ] **Concurrent writers** via a lock-coordinated segment hand-off.
- [ ] **Near-real-time readers** that refresh from the manifest without a full
      reopen.
- [ ] **A richer `explain`** that walks boolean/phrase/range queries exactly.
- [ ] **Pluggable language analyzers** beyond English.

## Known limitations
- The whole term dictionary of each open segment lives in memory.
- Wildcard and prefix queries scan every term in a field per segment.
- Numeric range and faceting scan the matching documents linearly.
- A single writer at a time (enforced by a lock file).
