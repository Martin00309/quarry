"""Segment merging.

Every commit writes a new segment, so an index accumulates them over time. Too
many segments slows search (each query touches all of them) and deleted
documents keep taking space until the segment that holds them is rewritten.
Merging solves both: it combines several segments into one, dropping deleted
documents and renumbering the survivors into a dense id space.

A :class:`MergePolicy` decides *what* to merge; :func:`merge_segments` does the
work and swaps the result into the manifest through the writer. Merges are safe
to run after a commit: they only read already-published segments and publish a
new manifest atomically, so a concurrent reader keeps using its old snapshot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .index import IndexWriter
from .postings import PostingsBuilder
from .segment import SegmentData, SegmentInfo, SegmentReader, write_segment


@dataclass
class MergePolicy:
    """Decides which segments to merge.

    Two triggers: too many segments (merge the smallest together to cut their
    number) and a single segment whose deletions exceed ``max_deleted_ratio``
    (rewrite it to reclaim the space).
    """

    max_segments: int = 8
    max_merge_at_once: int = 10
    max_deleted_ratio: float = 0.3

    def find_merge(self, infos: List[SegmentInfo]) -> Optional[List[str]]:
        if len(infos) > self.max_segments:
            ordered = sorted(infos, key=lambda i: i.num_live)
            chosen = ordered[: self.max_merge_at_once]
            return [i.name for i in chosen]
        for info in infos:
            if info.doc_count and info.num_deleted / info.doc_count > self.max_deleted_ratio:
                return [info.name]
        return None


def merge_segments(writer: IndexWriter, names: List[str]) -> Optional[SegmentInfo]:
    """Merge the named segments into one and publish the result.

    Returns the new :class:`SegmentInfo`, or ``None`` if every document in the
    merged segments was deleted (in which case the segments are simply removed).
    """
    schema = writer.schema
    readers = [SegmentReader(writer.directory, name, schema) for name in names]

    data = SegmentData(schema=schema)
    for field_name in schema.text_fields():
        with_positions = schema.field(field_name).has_positions
        data.postings[field_name] = PostingsBuilder(with_positions=with_positions)
        data.norms[field_name] = {}
        data.field_sum_length[field_name] = 0
        data.field_doc_count[field_name] = 0
    for field_name in schema.numeric_fields():
        data.numeric[field_name] = {}

    for reader in readers:
        mapping = _copy_live_docs(reader, data, schema)
        _copy_postings(reader, data, schema, mapping)

    if data.doc_count == 0:
        _finish(writer, names, None)
        return None

    name = f"_{writer.manifest.next_segment}"
    writer.manifest.next_segment += 1
    info = write_segment(writer.directory, name, data)
    _finish(writer, names, info)
    return info


def _copy_live_docs(reader: SegmentReader, data: SegmentData, schema) -> dict:
    mapping = {}
    for old_local in range(reader.doc_count):
        if reader.is_deleted(old_local):
            continue
        new_local = data.doc_count
        mapping[old_local] = new_local
        data.stored.append(reader.stored(old_local))
        data.ids.append(reader.id_of(old_local))
        for field_name in schema.text_fields():
            length = reader.field_length(field_name, old_local)
            data.norms[field_name][new_local] = length
            data.field_sum_length[field_name] += length
            if length:
                data.field_doc_count[field_name] += 1
        for field_name in schema.numeric_fields():
            value = reader.numeric_value(field_name, old_local)
            if value is not None:
                data.numeric[field_name][new_local] = value
        data.doc_count += 1
    return mapping


def _copy_postings(reader: SegmentReader, data: SegmentData, schema, mapping: dict) -> None:
    for field_name in schema.text_fields():
        with_positions = schema.field(field_name).has_positions
        builder = data.postings[field_name]
        for term in reader.field_terms(field_name):
            for posting in reader.postings(field_name, term):
                new_local = mapping.get(posting.doc)
                if new_local is None:
                    continue
                if with_positions:
                    for pos in posting.positions:
                        builder.add(term, new_local, pos)
                else:
                    for _ in range(posting.freq):
                        builder.add(term, new_local)


def _finish(writer: IndexWriter, old_names: List[str], info: Optional[SegmentInfo]) -> None:
    writer.replace_segments(old_names, info)
    for name in old_names:
        _remove_segment_files(writer.directory, name)


def _remove_segment_files(directory: str, name: str) -> None:
    for suffix in (".seg", ".del"):
        path = os.path.join(directory, f"{name}{suffix}")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def maybe_merge(writer: IndexWriter, policy: Optional[MergePolicy] = None) -> int:
    """Run merges until the policy is satisfied; return the number performed."""
    policy = policy or MergePolicy()
    merges = 0
    while True:
        names = policy.find_merge(writer.manifest.segments)
        if not names:
            return merges
        merge_segments(writer, names)
        merges += 1
