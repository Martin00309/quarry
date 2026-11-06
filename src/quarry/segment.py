"""Immutable on-disk segments.

An index is a set of segments. Each segment is written once and never mutated
(except for its deletion bitset), which makes writes append-only and reads
lock-free. A segment lives in two files inside the index directory:

``<name>.seg``
    The immutable payload: a container of named *sections* with a checksummed
    footer. Sections are the term dictionary, the postings, the stored fields,
    the length norms, the numeric columns and a small JSON metadata blob.
``<name>.del``
    An optional, mutable bitset of deleted local document ids. Deleting a
    document just flips a bit here; the space is reclaimed when the segment is
    merged.

Container layout of ``<name>.seg``::

    body         section payloads, concatenated
    directory    uvarint count, then per section: name, u64 offset, u64 length
    footer       u64 directory_offset, u32 crc32(body+directory), 4-byte "QSEG"

The term dictionary front-codes its sorted terms (shared-prefix + suffix) and
is read fully into memory on open, so term lookups are dictionary-fast.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .coding import ByteReader, ByteWriter, crc32
from .errors import CorruptIndexError
from .models import Document, TermInfo
from .postings import Posting, PostingsBuilder, decode_postings, encode_postings
from .schema import Schema
from .store import StoredReader, encode_stored

_MAGIC = b"QSEG"
_VERSION = 1
_FOOTER = struct.calcsize("<QI") + len(_MAGIC)  # u64 + u32 + magic
_DOUBLE = struct.Struct("<d")


# -- container (de)serialization ----------------------------------------
def _assemble(sections: List[Tuple[str, bytes]]) -> bytes:
    body = ByteWriter()
    directory: List[Tuple[str, int, int]] = []
    for name, payload in sections:
        directory.append((name, body.tell(), len(payload)))
        body.raw(payload)
    dir_writer = ByteWriter()
    dir_writer.uvarint(len(directory))
    for name, offset, length in directory:
        dir_writer.string(name).u64(offset).u64(length)
    pre_footer = body.getvalue() + dir_writer.getvalue()
    footer = ByteWriter()
    footer.u64(len(body))
    footer.u32(crc32(pre_footer))
    footer.raw(_MAGIC)
    return pre_footer + footer.getvalue()


def _parse_container(blob: bytes) -> Tuple[bytes, Dict[str, Tuple[int, int]]]:
    if len(blob) < _FOOTER or blob[-len(_MAGIC) :] != _MAGIC:
        raise CorruptIndexError("segment is missing its magic footer")
    footer = ByteReader(blob, len(blob) - _FOOTER)
    dir_offset = footer.u64()
    expected_crc = footer.u32()
    pre_footer = blob[: len(blob) - _FOOTER]
    if crc32(pre_footer) != expected_crc:
        raise CorruptIndexError("segment checksum mismatch")
    reader = ByteReader(blob, dir_offset)
    count = reader.uvarint()
    sections: Dict[str, Tuple[int, int]] = {}
    for _ in range(count):
        name = reader.string()
        offset = reader.u64()
        length = reader.u64()
        sections[name] = (offset, length)
    return blob, sections


# -- in-memory representation used while building -----------------------
@dataclass
class SegmentData:
    """Everything needed to write one segment, assembled in memory first."""

    schema: Schema
    doc_count: int = 0
    postings: Dict[str, PostingsBuilder] = field(default_factory=dict)
    norms: Dict[str, Dict[int, int]] = field(default_factory=dict)
    field_sum_length: Dict[str, int] = field(default_factory=dict)
    field_doc_count: Dict[str, int] = field(default_factory=dict)
    numeric: Dict[str, Dict[int, float]] = field(default_factory=dict)
    stored: List[Document] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)


# -- term dictionary -----------------------------------------------------
def _encode_terms(
    data: SegmentData,
) -> Tuple[bytes, bytes]:
    """Return ``(terms_section, postings_section)``.

    Postings for every term are concatenated into one section; the term
    dictionary stores each term's offset/length within it.
    """
    postings = ByteWriter()
    terms = ByteWriter()
    text_fields = [f for f in data.schema.text_fields() if f in data.postings]
    terms.uvarint(len(text_fields))
    for field_name in text_fields:
        builder = data.postings[field_name]
        with_positions = data.schema.field(field_name).has_positions
        term_list = builder.terms()
        terms.string(field_name)
        terms.uvarint(len(term_list))
        prev = ""
        for term in term_list:
            plist = builder.postings_for(term)
            offset = postings.tell()
            blob = encode_postings(plist, with_positions)
            postings.raw(blob)
            shared = _shared_prefix(prev, term)
            terms.uvarint(shared)
            terms.string(term[shared:])
            terms.uvarint(len(plist))  # doc_freq
            terms.uvarint(sum(p.freq for p in plist))  # total_term_freq
            terms.uvarint(offset)
            terms.uvarint(len(blob))
            prev = term
    return terms.getvalue(), postings.getvalue()


def _shared_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _decode_terms(blob: bytes, offset: int, length: int) -> Dict[str, Dict[str, TermInfo]]:
    reader = ByteReader(blob, offset)
    out: Dict[str, Dict[str, TermInfo]] = {}
    num_fields = reader.uvarint()
    for _ in range(num_fields):
        field_name = reader.string()
        num_terms = reader.uvarint()
        field_terms: Dict[str, TermInfo] = {}
        prev = ""
        for _ in range(num_terms):
            shared = reader.uvarint()
            suffix = reader.string()
            term = prev[:shared] + suffix
            doc_freq = reader.uvarint()
            total_tf = reader.uvarint()
            p_offset = reader.uvarint()
            p_length = reader.uvarint()
            field_terms[term] = TermInfo(doc_freq, total_tf, p_offset, p_length)
            prev = term
        out[field_name] = field_terms
    return out


# -- norms / numeric columns --------------------------------------------
def _encode_norms(data: SegmentData) -> bytes:
    writer = ByteWriter()
    fields = [f for f in data.schema.text_fields() if f in data.norms]
    writer.uvarint(len(fields))
    for field_name in fields:
        lengths = data.norms[field_name]
        writer.string(field_name)
        for local in range(data.doc_count):
            writer.uvarint(lengths.get(local, 0))
    return writer.getvalue()


def _decode_norms(blob: bytes, offset: int, doc_count: int) -> Dict[str, List[int]]:
    reader = ByteReader(blob, offset)
    out: Dict[str, List[int]] = {}
    num_fields = reader.uvarint()
    for _ in range(num_fields):
        field_name = reader.string()
        out[field_name] = [reader.uvarint() for _ in range(doc_count)]
    return out


def _encode_numeric(data: SegmentData) -> bytes:
    writer = ByteWriter()
    fields = [f for f in data.schema.numeric_fields() if f in data.numeric]
    writer.uvarint(len(fields))
    for field_name in fields:
        values = data.numeric[field_name]
        ntype = data.schema.field(field_name).numeric_type
        writer.string(field_name)
        writer.string(ntype)
        for local in range(data.doc_count):
            if local in values:
                writer.uvarint(1)
                writer.raw(_DOUBLE.pack(float(values[local])))
            else:
                writer.uvarint(0)
    return writer.getvalue()


def _decode_numeric(blob: bytes, offset: int, doc_count: int) -> Dict[str, List[Optional[float]]]:
    reader = ByteReader(blob, offset)
    out: Dict[str, List[Optional[float]]] = {}
    num_fields = reader.uvarint()
    for _ in range(num_fields):
        field_name = reader.string()
        reader.string()  # numeric_type, informational
        column: List[Optional[float]] = []
        for _ in range(doc_count):
            present = reader.uvarint()
            column.append(_DOUBLE.unpack(reader.raw(8))[0] if present else None)
        out[field_name] = column
    return out


# -- writer --------------------------------------------------------------
def write_segment(directory: str, name: str, data: SegmentData) -> "SegmentInfo":
    """Serialize ``data`` to ``<directory>/<name>.seg`` and return its info."""
    terms_section, postings_section = _encode_terms(data)
    stored_section = encode_stored(data.stored)
    norms_section = _encode_norms(data)
    numeric_section = _encode_numeric(data)
    meta = {
        "name": name,
        "version": _VERSION,
        "doc_count": data.doc_count,
        "field_sum_length": data.field_sum_length,
        "field_doc_count": data.field_doc_count,
        "ids": data.ids,
    }
    meta_section = json.dumps(meta, ensure_ascii=False).encode("utf-8")

    blob = _assemble(
        [
            ("terms", terms_section),
            ("postings", postings_section),
            ("stored", stored_section),
            ("norms", norms_section),
            ("numeric", numeric_section),
            ("meta", meta_section),
        ]
    )
    path = os.path.join(directory, f"{name}.seg")
    _atomic_write(path, blob)
    return SegmentInfo(name=name, doc_count=data.doc_count, num_deleted=0)


def _atomic_write(path: str, blob: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@dataclass
class SegmentInfo:
    """Lightweight record of a segment in the manifest."""

    name: str
    doc_count: int
    num_deleted: int = 0

    @property
    def num_live(self) -> int:
        return self.doc_count - self.num_deleted

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "doc_count": self.doc_count, "num_deleted": self.num_deleted}

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "SegmentInfo":
        return SegmentInfo(
            name=str(d["name"]),
            doc_count=int(d["doc_count"]),  # type: ignore[arg-type]
            num_deleted=int(d.get("num_deleted", 0)),  # type: ignore[arg-type]
        )


# -- live docs (deletions) ----------------------------------------------
class LiveDocs:
    """A growable bitset marking which local docs are still alive.

    Persisted as ``<name>.del`` only when at least one doc is deleted, so a
    fresh segment needs no extra file.
    """

    _DEL_MAGIC = b"QDEL"

    def __init__(self, doc_count: int, deleted: Optional[set] = None) -> None:
        self.doc_count = doc_count
        self._deleted = set(deleted or ())

    @property
    def num_deleted(self) -> int:
        return len(self._deleted)

    def is_deleted(self, local: int) -> bool:
        return local in self._deleted

    def delete(self, local: int) -> bool:
        if 0 <= local < self.doc_count and local not in self._deleted:
            self._deleted.add(local)
            return True
        return False

    def live_ids(self) -> List[int]:
        return [i for i in range(self.doc_count) if i not in self._deleted]

    def serialize(self) -> bytes:
        writer = ByteWriter()
        writer.raw(self._DEL_MAGIC)
        writer.uvarint(self.doc_count)
        writer.uvarint(len(self._deleted))
        for local in sorted(self._deleted):
            writer.uvarint(local)
        body = writer.getvalue()
        return body + struct.pack("<I", crc32(body))

    @classmethod
    def load(cls, blob: bytes) -> "LiveDocs":
        if len(blob) < 4 or blob[:4] != cls._DEL_MAGIC:
            raise CorruptIndexError("deletions file is missing its magic")
        body = blob[:-4]
        (expected,) = struct.unpack("<I", blob[-4:])
        if crc32(body) != expected:
            raise CorruptIndexError("deletions checksum mismatch")
        reader = ByteReader(body, len(cls._DEL_MAGIC))
        doc_count = reader.uvarint()
        count = reader.uvarint()
        deleted = {reader.uvarint() for _ in range(count)}
        return cls(doc_count, deleted)


# -- reader --------------------------------------------------------------
class SegmentReader:
    """Read-only view of a segment, with its deletion bitset applied."""

    def __init__(self, directory: str, name: str, schema: Schema) -> None:
        self.directory = directory
        self.name = name
        self.schema = schema
        with open(os.path.join(directory, f"{name}.seg"), "rb") as fh:
            self._blob = fh.read()
        _, self._sections = _parse_container(self._blob)
        meta_off, _ = self._sections["meta"]
        meta_len = self._sections["meta"][1]
        self.meta = json.loads(self._blob[meta_off : meta_off + meta_len].decode("utf-8"))
        self.doc_count: int = self.meta["doc_count"]
        self._ids: List[str] = self.meta["ids"]
        self._id_to_local = {doc_id: i for i, doc_id in enumerate(self._ids)}
        self._field_sum_length: Dict[str, int] = self.meta.get("field_sum_length", {})
        self._field_doc_count: Dict[str, int] = self.meta.get("field_doc_count", {})

        terms_off, terms_len = self._sections["terms"]
        self._terms = _decode_terms(self._blob, terms_off, terms_len)
        self._postings_base = self._sections["postings"][0]

        norms_off, _ = self._sections["norms"]
        self._norms = _decode_norms(self._blob, norms_off, self.doc_count)
        numeric_off, _ = self._sections["numeric"]
        self._numeric = _decode_numeric(self._blob, numeric_off, self.doc_count)

        stored_off, stored_len = self._sections["stored"]
        self._stored = StoredReader(self._blob[stored_off : stored_off + stored_len])

        self.live_docs = self._load_live_docs()

    def _load_live_docs(self) -> LiveDocs:
        del_path = os.path.join(self.directory, f"{self.name}.del")
        if os.path.exists(del_path):
            with open(del_path, "rb") as fh:
                return LiveDocs.load(fh.read())
        return LiveDocs(self.doc_count)

    # -- deletions -------------------------------------------------------
    def is_deleted(self, local: int) -> bool:
        return self.live_docs.is_deleted(local)

    @property
    def num_deleted(self) -> int:
        return self.live_docs.num_deleted

    @property
    def num_live(self) -> int:
        return self.doc_count - self.live_docs.num_deleted

    def delete_by_id(self, doc_id: str) -> bool:
        local = self._id_to_local.get(doc_id)
        if local is None:
            return False
        return self.live_docs.delete(local)

    def flush_deletions(self) -> None:
        if self.live_docs.num_deleted:
            path = os.path.join(self.directory, f"{self.name}.del")
            _atomic_write(path, self.live_docs.serialize())

    # -- terms & postings ------------------------------------------------
    def field_terms(self, field_name: str) -> Dict[str, TermInfo]:
        return self._terms.get(field_name, {})

    def term_info(self, field_name: str, term: str) -> Optional[TermInfo]:
        return self._terms.get(field_name, {}).get(term)

    def postings(self, field_name: str, term: str) -> List[Posting]:
        info = self.term_info(field_name, term)
        if info is None:
            return []
        start = self._postings_base + info.postings_offset
        blob = self._blob[start : start + info.postings_length]
        with_positions = self.schema.field(field_name).has_positions
        return decode_postings(blob, with_positions)

    # -- statistics ------------------------------------------------------
    def doc_freq(self, field_name: str, term: str) -> int:
        info = self.term_info(field_name, term)
        return info.doc_freq if info else 0

    def field_length(self, field_name: str, local: int) -> int:
        column = self._norms.get(field_name)
        return column[local] if column else 0

    def field_sum_length(self, field_name: str) -> int:
        return self._field_sum_length.get(field_name, 0)

    def field_doc_count(self, field_name: str) -> int:
        return self._field_doc_count.get(field_name, 0)

    # -- numeric ---------------------------------------------------------
    def numeric_value(self, field_name: str, local: int) -> Optional[float]:
        column = self._numeric.get(field_name)
        return column[local] if column else None

    # -- stored & ids ----------------------------------------------------
    def stored(self, local: int) -> Document:
        return self._stored.get(local)

    def id_of(self, local: int) -> str:
        return self._ids[local]

    def local_of_id(self, doc_id: str) -> Optional[int]:
        return self._id_to_local.get(doc_id)

    def all_ids(self) -> List[str]:
        return list(self._ids)
