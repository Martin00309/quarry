"""The stored-fields document store.

Each segment keeps a column of *stored* field values so that a search can
return the original data for a hit without re-reading the source. The store is
a flat blob with a footer that gives random access by local document id:

    body         per doc, in id order: a length-prefixed JSON object
    offsets      one u64 per doc: the byte offset of that doc's body entry
    footer       u64 offsets_start, u32 doc_count, u32 crc32(everything above)

The CRC covers the body and the offset table, so a truncated or corrupted
store is detected when the segment is opened.
"""

from __future__ import annotations

import json
from typing import List

from .coding import ByteReader, ByteWriter, crc32
from .errors import CorruptIndexError
from .models import Document

_FOOTER = 16  # u64 + u32 + u32


def encode_stored(docs: List[Document]) -> bytes:
    """Serialize per-document stored fields (``docs[i]`` for local id ``i``)."""
    writer = ByteWriter()
    offsets: List[int] = []
    for doc in docs:
        offsets.append(writer.tell())
        payload = json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8")
        writer.bytes(payload)
    offsets_start = writer.tell()
    for off in offsets:
        writer.u64(off)
    body = writer.getvalue()
    footer = ByteWriter()
    footer.u64(offsets_start)
    footer.u32(len(offsets))
    footer.u32(crc32(body))
    return body + footer.getvalue()


class StoredReader:
    """Random-access reader over a :func:`encode_stored` blob."""

    def __init__(self, blob: bytes) -> None:
        if len(blob) < _FOOTER:
            raise CorruptIndexError("stored-fields blob is too small")
        footer = ByteReader(blob, len(blob) - _FOOTER)
        self._offsets_start = footer.u64()
        self._count = footer.u32()
        expected_crc = footer.u32()
        body = blob[: len(blob) - _FOOTER]
        if crc32(body) != expected_crc:
            raise CorruptIndexError("stored-fields checksum mismatch")
        self._blob = blob

    def __len__(self) -> int:
        return self._count

    def get(self, local_doc: int) -> Document:
        """Return the stored fields for ``local_doc`` (0-based)."""
        if not 0 <= local_doc < self._count:
            raise IndexError(f"doc {local_doc} out of range (0..{self._count - 1})")
        table = ByteReader(self._blob, self._offsets_start + local_doc * 8)
        offset = table.u64()
        reader = ByteReader(self._blob, offset)
        payload = reader.bytes()
        return json.loads(payload.decode("utf-8"))

    def all(self) -> List[Document]:
        return [self.get(i) for i in range(self._count)]


class StoredWriter:
    """Accumulates stored documents in local-id order."""

    def __init__(self) -> None:
        self._docs: List[Document] = []

    def add(self, stored: Document) -> int:
        local = len(self._docs)
        self._docs.append(dict(stored))
        return local

    def __len__(self) -> int:
        return len(self._docs)

    def serialize(self) -> bytes:
        return encode_stored(self._docs)

    def documents(self) -> List[Document]:
        return list(self._docs)
