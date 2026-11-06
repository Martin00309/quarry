"""Public data types returned by searches.

These are deliberately thin: a :class:`Hit` is one ranked document, and a
:class:`SearchResult` is an ordered page of hits plus the totals a UI needs to
render "showing 1-10 of 42".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

#: A document is just a mapping of field name to value.
Document = Dict[str, Any]


@dataclass
class Hit:
    """One document in a result set.

    Attributes
    ----------
    doc_id:
        The document's primary-key value (the schema ``id`` field).
    score:
        Relevance score under the active similarity model. Higher is better.
        Filter-only matches that are not scored carry ``0.0``.
    fields:
        Stored field values for the document (empty if nothing was stored or
        none were requested).
    highlights:
        Optional field-name to snippet mapping produced by the highlighter.
    """

    doc_id: str
    score: float
    fields: Document = field(default_factory=dict)
    highlights: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.doc_id, "score": self.score, "fields": self.fields}
        if self.highlights:
            out["highlights"] = self.highlights
        return out


@dataclass
class SearchResult:
    """An ordered page of hits and the counts around it."""

    hits: List[Hit]
    total_hits: int
    max_score: float = 0.0
    facets: Dict[str, List["FacetCount"]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "total_hits": self.total_hits,
            "max_score": self.max_score,
            "hits": [h.to_dict() for h in self.hits],
        }
        if self.facets:
            out["facets"] = {
                name: [fc.to_dict() for fc in counts] for name, counts in self.facets.items()
            }
        return out


@dataclass
class FacetCount:
    """A single value/count pair from a faceted aggregation."""

    value: str
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "count": self.count}


@dataclass
class Explanation:
    """A human-readable breakdown of how a score was computed."""

    value: float
    description: str
    details: List["Explanation"] = field(default_factory=list)

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [f"{pad}{self.value:.6f} = {self.description}"]
        for child in self.details:
            lines.append(child.render(indent + 1))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "description": self.description,
            "details": [d.to_dict() for d in self.details],
        }


@dataclass
class TermInfo:
    """Where a term's postings live inside a segment and how big they are."""

    doc_freq: int
    total_term_freq: int
    postings_offset: int
    postings_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_freq": self.doc_freq,
            "total_term_freq": self.total_term_freq,
            "postings_offset": self.postings_offset,
            "postings_length": self.postings_length,
        }
