# On-disk format

This document describes the bytes Quarry writes. The format is versioned
(`version = 1`) but not yet guaranteed stable across releases.

All integers are little-endian. "uvarint" is an unsigned LEB128 variable-length
integer; "varint" is a zig-zag-mapped signed varint. A "string" is a uvarint
byte length followed by that many UTF-8 bytes. These primitives live in
[`coding.py`](../src/quarry/coding.py).

## Directory layout

An index is a directory:

```
schema.json      the serialized Schema (written once at creation)
manifest.json    the live segment list + generation counters (rewritten atomically)
write.lock       present while a writer is open; holds the writer's pid
_N.seg           an immutable segment (N is an integer, assigned in order)
_N.del           optional deletion bitset for segment _N
```

`manifest.json` and `write.lock` make writes safe: a commit writes the new
`.seg` file first, then atomically replaces `manifest.json`, so a reader that
opened the previous manifest keeps a consistent view.

### manifest.json

```json
{
  "version": 1,
  "generation": 7,
  "next_segment": 4,
  "segments": [
    {"name": "_0", "doc_count": 100, "num_deleted": 3},
    {"name": "_3", "doc_count": 50,  "num_deleted": 0}
  ]
}
```

- `generation` increments on every change-bearing commit.
- `next_segment` is the counter used to name new segments (`_0`, `_1`, …).

## Segment container (`_N.seg`)

A segment is a container of named **sections** with a checksummed footer:

```
body         section payloads, concatenated
directory    uvarint count, then per section: string name, u64 offset, u64 length
footer       u64 directory_offset, u32 crc32(body + directory), 4-byte magic "QSEG"
```

The footer is a fixed 16 bytes. On open, Quarry reads the footer, verifies the
magic and the CRC over everything before the footer, then reads the directory to
locate each section. The sections are:

| Section | Contents |
| --- | --- |
| `terms` | the term dictionary (per field, front-coded sorted terms → term info) |
| `postings` | every term's posting list, concatenated |
| `stored` | the stored-fields document store |
| `norms` | per text field, each document's indexed length |
| `numeric` | per numeric field, a dense column of values |
| `meta` | a small JSON blob (doc count, field stats, the id list) |

### `terms` — the term dictionary

```
uvarint   num_fields
repeat num_fields:
    string   field_name
    uvarint  num_terms
    repeat num_terms (terms are sorted; front-coded against the previous term):
        uvarint  shared_prefix_len     # chars shared with the previous term
        string   suffix                # the rest of this term
        uvarint  doc_freq              # documents containing the term
        uvarint  total_term_freq       # total occurrences across documents
        uvarint  postings_offset       # offset into the `postings` section
        uvarint  postings_length       # byte length of this term's postings
```

Front-coding exploits the sorted order: `connect`, `connected`, `connecting`
share the `connect` prefix and store only the differing suffix.

### `postings` — posting lists

Each term's posting list is a self-describing blob (see
[`postings.py`](../src/quarry/postings.py)):

```
uvarint   count                        # number of postings
repeat count:
    uvarint  doc_delta                 # doc id minus the previous doc id
    uvarint  freq                      # term frequency in this document
    if the field has positions:
        uvarint  position_delta  × freq # position minus the previous position
```

Document ids within a list are strictly increasing, so delta-encoding keeps them
small; the same delta trick is applied to positions within a document.

### `stored` — the document store

```
body     per doc (in local-id order): a length-prefixed JSON object
offsets  one u64 per doc: the byte offset of that doc's body entry
footer   u64 offsets_start, u32 doc_count, u32 crc32(body + offsets)
```

The offset table gives O(1) random access to any document's stored fields; the
CRC detects truncation or corruption.

### `norms` — field lengths

```
uvarint   num_fields
repeat num_fields:
    string   field_name
    uvarint  length  × doc_count       # indexed token count per local doc
```

Field lengths feed BM25's length normalization.

### `numeric` — numeric columns

```
uvarint   num_fields
repeat num_fields:
    string   field_name
    string   numeric_type              # "int" or "float"
    repeat doc_count:
        uvarint  present               # 1 if this doc has a value, else 0
        if present: 8-byte IEEE-754 double
```

### `meta`

A JSON object:

```json
{
  "name": "_0",
  "version": 1,
  "doc_count": 100,
  "field_sum_length": {"title": 420, "body": 3100},
  "field_doc_count": {"title": 100, "body": 100},
  "ids": ["doc-1", "doc-2", "..."]
}
```

`field_sum_length` and `field_doc_count` give the per-field average length used
for scoring; `ids` maps each local document id to its primary key.

## Deletions (`_N.del`)

Deletes are tombstones, written to a small side file only once a segment has at
least one deleted document:

```
magic    4-byte "QDEL"
uvarint  doc_count
uvarint  num_deleted
uvarint  local_id  × num_deleted       # sorted ascending
u32      crc32(everything above)
```

Merging a segment rewrites it without its deleted documents and removes the
`.del` file, reclaiming the space.
