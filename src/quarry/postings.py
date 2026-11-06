"""Posting lists: the per-term inverted lists at the core of the index.

A *posting* records that a term appears in a document: the local document id,
how many times it occurs in the field (the term frequency), and -- when the
field is indexed with positions -- the ordinal positions of each occurrence.

On disk a posting list is a self-describing blob:

    count                      uvarint   number of postings
    repeated `count` times:
        doc_delta              uvarint   doc id minus the previous doc id
        freq                   uvarint   term frequency in this document
        positions (optional)   uvarint*  `freq` position deltas, if positions on

Document ids within a list are strictly increasing, so delta encoding keeps the
bytes small; the same trick is used for the positions inside one document. The
reader decodes a blob into a :class:`PostingCursor`, a document-at-a-time
iterator supporting ``next`` and ``advance(target)`` for conjunctive queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .coding import ByteReader, ByteWriter

#: Sentinel id meaning "the cursor is exhausted"; larger than any real doc id.
NO_MORE_DOCS = (1 << 63) - 1


@dataclass
class Posting:
    """One (doc, freq, positions) entry in a posting list."""

    doc: int
    freq: int
    positions: Tuple[int, ...] = ()


def encode_postings(postings: Sequence[Posting], with_positions: bool) -> bytes:
    """Serialize ``postings`` (already sorted by doc id) to a blob."""
    writer = ByteWriter()
    writer.uvarint(len(postings))
    prev_doc = 0
    for p in postings:
        if p.doc < prev_doc:
            raise ValueError("postings must be sorted by increasing doc id")
        writer.uvarint(p.doc - prev_doc)
        prev_doc = p.doc
        writer.uvarint(p.freq)
        if with_positions:
            if len(p.positions) != p.freq:
                raise ValueError("position count must equal term frequency")
            prev_pos = 0
            for pos in p.positions:
                writer.uvarint(pos - prev_pos)
                prev_pos = pos
    return writer.getvalue()


def decode_postings(blob: bytes, with_positions: bool) -> List[Posting]:
    """Inverse of :func:`encode_postings`."""
    reader = ByteReader(blob)
    count = reader.uvarint()
    out: List[Posting] = []
    doc = 0
    for _ in range(count):
        doc += reader.uvarint()
        freq = reader.uvarint()
        if with_positions:
            positions = []
            pos = 0
            for _ in range(freq):
                pos += reader.uvarint()
                positions.append(pos)
            out.append(Posting(doc, freq, tuple(positions)))
        else:
            out.append(Posting(doc, freq))
    return out


class PostingCursor:
    """A document-at-a-time cursor over a decoded posting list.

    The cursor starts *before* the first posting; call :meth:`next` or
    :meth:`advance` to position it. :attr:`doc` is :data:`NO_MORE_DOCS` once
    exhausted.
    """

    __slots__ = ("_postings", "_i", "doc_freq")

    def __init__(self, postings: List[Posting]) -> None:
        self._postings = postings
        self._i = -1
        self.doc_freq = len(postings)

    @property
    def doc(self) -> int:
        if 0 <= self._i < len(self._postings):
            return self._postings[self._i].doc
        return NO_MORE_DOCS

    @property
    def freq(self) -> int:
        if 0 <= self._i < len(self._postings):
            return self._postings[self._i].freq
        return 0

    @property
    def positions(self) -> Tuple[int, ...]:
        if 0 <= self._i < len(self._postings):
            return self._postings[self._i].positions
        return ()

    def next(self) -> int:
        """Advance to the next posting and return its doc id."""
        self._i += 1
        return self.doc

    def advance(self, target: int) -> int:
        """Advance to the first doc id >= ``target`` and return it.

        Uses a binary search from the current position, which keeps conjunctive
        queries cheap when one list is much shorter than another.
        """
        lo = max(self._i, 0)
        hi = len(self._postings)
        postings = self._postings
        while lo < hi:
            mid = (lo + hi) // 2
            if postings[mid].doc < target:
                lo = mid + 1
            else:
                hi = mid
        self._i = lo
        return self.doc


class PostingsBuilder:
    """In-memory accumulator used while a segment is being built.

    Call :meth:`add` once per occurrence of a term in a document (passing the
    position for positional fields), then :meth:`finish` to get posting lists
    sorted by doc id, ready to encode.
    """

    def __init__(self, with_positions: bool = True) -> None:
        self.with_positions = with_positions
        # term -> doc -> list of positions (or a running count when no positions)
        self._terms: Dict[str, Dict[int, List[int]]] = {}
        self._freqs: Dict[str, Dict[int, int]] = {}

    def add(self, term: str, doc: int, position: int = 0) -> None:
        if self.with_positions:
            self._terms.setdefault(term, {}).setdefault(doc, []).append(position)
        else:
            per_doc = self._freqs.setdefault(term, {})
            per_doc[doc] = per_doc.get(doc, 0) + 1

    def terms(self) -> List[str]:
        source = self._terms if self.with_positions else self._freqs
        return sorted(source)

    def postings_for(self, term: str) -> List[Posting]:
        if self.with_positions:
            per_doc = self._terms.get(term, {})
            return [
                Posting(doc, len(positions), tuple(sorted(positions)))
                for doc, positions in sorted(per_doc.items())
            ]
        per_doc = self._freqs.get(term, {})
        return [Posting(doc, freq) for doc, freq in sorted(per_doc.items())]

    def total_term_freq(self, term: str) -> int:
        return sum(p.freq for p in self.postings_for(term))


@dataclass
class FieldPostings:
    """Grouping of a field's builder with its length bookkeeping.

    ``doc_lengths`` maps a local doc id to the number of indexed tokens in the
    field for that document, which the similarity model needs for length
    normalization.
    """

    builder: PostingsBuilder
    doc_lengths: Dict[int, int] = field(default_factory=dict)
    total_tokens: int = 0
