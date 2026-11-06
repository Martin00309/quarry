"""The index writer and its on-disk manifest.

An index is a directory containing:

``schema.json``
    The serialized :class:`~quarry.schema.Schema`, written once at creation.
``manifest.json``
    The list of live segments and the generation counter. It is rewritten
    atomically on every commit, so a reader always sees a consistent set.
``write.lock``
    Held for the lifetime of a writer. A second writer that finds it raises
    :class:`~quarry.errors.LockError` rather than corrupting the index.
``_N.seg`` / ``_N.del``
    The segments themselves (see :mod:`quarry.segment`).

Documents added between commits are buffered in memory keyed by their primary
id, which gives natural upsert semantics: adding an id that already exists
replaces it. On :meth:`commit`, buffered deletions are applied to existing
segments and the buffer is flushed into a new segment.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Set

from .analysis import Token
from .errors import DocumentError, LockError, SchemaError
from .models import Document
from .schema import Schema
from .segment import SegmentData, SegmentInfo, SegmentReader, write_segment

_MANIFEST = "manifest.json"
_SCHEMA = "schema.json"
_LOCK = "write.lock"


class Manifest:
    """The set of live segments plus the counters that name new ones."""

    def __init__(
        self,
        generation: int = 0,
        next_segment: int = 0,
        segments: Optional[List[SegmentInfo]] = None,
    ) -> None:
        self.generation = generation
        self.next_segment = next_segment
        self.segments: List[SegmentInfo] = segments or []

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "generation": self.generation,
            "next_segment": self.next_segment,
            "segments": [s.to_dict() for s in self.segments],
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Manifest":
        segments = [SegmentInfo.from_dict(s) for s in d.get("segments", [])]  # type: ignore[union-attr]
        return Manifest(
            generation=int(d.get("generation", 0)),  # type: ignore[arg-type]
            next_segment=int(d.get("next_segment", 0)),  # type: ignore[arg-type]
            segments=segments,
        )


def _write_json_atomic(path: str, data: object) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_manifest(directory: str) -> Manifest:
    path = os.path.join(directory, _MANIFEST)
    if not os.path.exists(path):
        return Manifest()
    with open(path, encoding="utf-8") as fh:
        return Manifest.from_dict(json.load(fh))


def read_schema(directory: str) -> Schema:
    path = os.path.join(directory, _SCHEMA)
    with open(path, encoding="utf-8") as fh:
        return Schema.from_dict(json.load(fh))


class IndexWriter:
    """Adds, updates and deletes documents, committing in batches."""

    def __init__(self, directory: str, schema: Schema, manifest: Manifest) -> None:
        self.directory = directory
        self.schema = schema
        self.manifest = manifest
        self._buffer: Dict[str, Document] = {}
        self._pending_deletes: Set[str] = set()
        self._lock_path = os.path.join(directory, _LOCK)
        self._acquire_lock()
        self._closed = False

    # -- construction ----------------------------------------------------
    @classmethod
    def create(cls, directory: str, schema: Schema, *, exist_ok: bool = False) -> "IndexWriter":
        """Create a fresh index directory and return a writer for it."""
        if os.path.exists(os.path.join(directory, _MANIFEST)) and not exist_ok:
            raise SchemaError(f"an index already exists at {directory!r}")
        os.makedirs(directory, exist_ok=True)
        _write_json_atomic(os.path.join(directory, _SCHEMA), schema.to_dict())
        manifest = Manifest()
        _write_json_atomic(os.path.join(directory, _MANIFEST), manifest.to_dict())
        return cls(directory, schema, manifest)

    @classmethod
    def open(cls, directory: str) -> "IndexWriter":
        """Open an existing index for writing."""
        if not os.path.exists(os.path.join(directory, _MANIFEST)):
            raise SchemaError(f"no index found at {directory!r}")
        schema = read_schema(directory)
        manifest = read_manifest(directory)
        return cls(directory, schema, manifest)

    # -- locking ---------------------------------------------------------
    def _acquire_lock(self) -> None:
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LockError(f"index at {self.directory!r} is locked by another writer") from None
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)

    def _release_lock(self) -> None:
        try:
            os.remove(self._lock_path)
        except FileNotFoundError:
            pass

    # -- document mutations ----------------------------------------------
    def add_document(self, doc: Document) -> None:
        """Add (or replace) a document. The id field must be present."""
        self._validate(doc)
        doc_id = str(doc[self.schema.id])
        self._buffer[doc_id] = dict(doc)
        # Replacing any already-committed copy of this id.
        self._pending_deletes.add(doc_id)

    def add_documents(self, docs) -> int:
        count = 0
        for doc in docs:
            self.add_document(doc)
            count += 1
        return count

    def delete_document(self, doc_id: str) -> None:
        """Mark a document for deletion by its primary id."""
        doc_id = str(doc_id)
        self._buffer.pop(doc_id, None)
        self._pending_deletes.add(doc_id)

    def _validate(self, doc: Document) -> None:
        id_field = self.schema.id
        if id_field not in doc or doc[id_field] in (None, ""):
            raise DocumentError(f"document is missing required id field {id_field!r}")
        for name in doc:
            if name not in self.schema:
                raise SchemaError(f"document has unknown field {name!r}")

    # -- commit ----------------------------------------------------------
    @property
    def buffered_count(self) -> int:
        return len(self._buffer)

    def commit(self) -> Manifest:
        """Flush buffered changes into a new segment and publish a manifest."""
        self._check_open()
        changed = False

        if self._pending_deletes:
            changed |= self._apply_deletions()

        if self._buffer:
            info = self._flush_segment()
            self.manifest.segments.append(info)
            changed = True

        self._buffer.clear()
        self._pending_deletes.clear()

        if changed:
            self.manifest.generation += 1
            self._publish()
        return self.manifest

    def _apply_deletions(self) -> bool:
        changed = False
        for info in self.manifest.segments:
            reader = SegmentReader(self.directory, info.name, self.schema)
            removed = 0
            for doc_id in self._pending_deletes:
                if reader.delete_by_id(doc_id):
                    removed += 1
            if removed:
                reader.flush_deletions()
                info.num_deleted = reader.num_deleted
                changed = True
        return changed

    def _flush_segment(self) -> SegmentInfo:
        data = SegmentData(schema=self.schema)
        for field_name in self.schema.text_fields():
            with_positions = self.schema.field(field_name).has_positions
            from .postings import PostingsBuilder

            data.postings[field_name] = PostingsBuilder(with_positions=with_positions)
            data.norms[field_name] = {}
            data.field_sum_length.setdefault(field_name, 0)
            data.field_doc_count.setdefault(field_name, 0)
        for field_name in self.schema.numeric_fields():
            data.numeric[field_name] = {}

        for doc in self._buffer.values():
            self._add_to_segment(data, doc)

        name = f"_{self.manifest.next_segment}"
        self.manifest.next_segment += 1
        return write_segment(self.directory, name, data)

    def _add_to_segment(self, data: SegmentData, doc: Document) -> None:
        local = data.doc_count
        stored: Document = {}
        for field_name, field in self.schema.fields.items():
            if field_name not in doc:
                continue
            value = doc[field_name]
            if field.indexed and field.is_text:
                tokens: List[Token] = field.analyzer().analyze(_as_text(value))
                for tok in tokens:
                    data.postings[field_name].add(tok.text, local, tok.position)
                length = len(tokens)
                data.norms[field_name][local] = length
                data.field_sum_length[field_name] += length
                if length:
                    data.field_doc_count[field_name] += 1
            if field.indexed and field.is_numeric:
                data.numeric[field_name][local] = field.coerce(value)
            if field.stored:
                stored[field_name] = value
        data.stored.append(stored)
        data.ids.append(str(doc[self.schema.id]))
        data.doc_count += 1

    def _publish(self) -> None:
        _write_json_atomic(os.path.join(self.directory, _MANIFEST), self.manifest.to_dict())

    def replace_segments(self, old_names: List[str], new_info: Optional[SegmentInfo]) -> Manifest:
        """Swap ``old_names`` out of the manifest for ``new_info`` (used by merge)."""
        self._check_open()
        remaining = [s for s in self.manifest.segments if s.name not in set(old_names)]
        if new_info is not None:
            remaining.append(new_info)
        self.manifest.segments = remaining
        self.manifest.generation += 1
        self._publish()
        return self.manifest

    # -- lifecycle -------------------------------------------------------
    def _check_open(self) -> None:
        if self._closed:
            raise LockError("writer is closed")

    def close(self) -> None:
        if not self._closed:
            self._release_lock()
            self._closed = True

    def __enter__(self) -> "IndexWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else str(value)
