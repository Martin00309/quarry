"""Reading an index and running searches over it.

:class:`IndexReader` opens the manifest and every live segment, and exposes the
collection-wide statistics (document frequency, average field length) that
correct scoring needs. :class:`Searcher` runs a query across all segments,
merges the per-segment matches into a single ranked result, and handles
sorting, faceting, highlighting and score explanation.

A reader sees a consistent snapshot: it reads the manifest once at open time, so
a concurrent writer committing new segments does not disturb it.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .errors import QueryError
from .index import read_manifest, read_schema
from .models import Explanation, FacetCount, Hit, SearchResult
from .postings import NO_MORE_DOCS
from .query import Query, SearchContext
from .schema import Schema
from .segment import SegmentReader
from .similarity import BM25Similarity, Similarity, get_similarity


class IndexReader:
    """A read-only snapshot of an index: its schema, segments and statistics."""

    def __init__(self, directory: str) -> None:
        if not os.path.exists(os.path.join(directory, "manifest.json")):
            raise FileNotFoundError(f"no index found at {directory!r}")
        self.directory = directory
        self.schema: Schema = read_schema(directory)
        manifest = read_manifest(directory)
        self.segments: List[SegmentReader] = [
            SegmentReader(directory, info.name, self.schema) for info in manifest.segments
        ]
        self._doc_freq_cache: Dict[Tuple[str, str], int] = {}
        self._avg_len_cache: Dict[str, float] = {}

    # -- collection statistics ------------------------------------------
    @property
    def num_docs(self) -> int:
        """Number of live (non-deleted) documents across all segments."""
        return sum(seg.num_live for seg in self.segments)

    @property
    def max_doc(self) -> int:
        """Total documents including deleted ones."""
        return sum(seg.doc_count for seg in self.segments)

    def doc_freq(self, field_name: str, term: str) -> int:
        key = (field_name, term)
        cached = self._doc_freq_cache.get(key)
        if cached is None:
            cached = sum(seg.doc_freq(field_name, term) for seg in self.segments)
            self._doc_freq_cache[key] = cached
        return cached

    def avg_field_length(self, field_name: str) -> float:
        cached = self._avg_len_cache.get(field_name)
        if cached is None:
            total_len = sum(seg.field_sum_length(field_name) for seg in self.segments)
            total_docs = sum(seg.field_doc_count(field_name) for seg in self.segments)
            cached = (total_len / total_docs) if total_docs else 0.0
            self._avg_len_cache[field_name] = cached
        return cached

    def document(self, doc_id: str) -> Optional[Dict]:
        """Return the stored fields for a live document, or ``None``."""
        for seg in self.segments:
            local = seg.local_of_id(doc_id)
            if local is not None and not seg.is_deleted(local):
                return seg.stored(local)
        return None

    def close(self) -> None:
        self.segments = []


# A sort spec: (field name or "score", reverse). "score" sorts by relevance.
Sort = Tuple[str, bool]


class Searcher:
    """Executes queries against an :class:`IndexReader`."""

    def __init__(self, reader: IndexReader, similarity: Optional[Similarity] = None) -> None:
        self.reader = reader
        self.similarity = similarity or BM25Similarity()

    @classmethod
    def open(cls, directory: str, similarity: str = "bm25") -> "Searcher":
        return cls(IndexReader(directory), get_similarity(similarity))

    @property
    def schema(self) -> Schema:
        return self.reader.schema

    def _context(self) -> SearchContext:
        return SearchContext(
            similarity=self.similarity,
            num_docs=max(self.reader.num_docs, 1),
            doc_freq_fn=self.reader.doc_freq,
            avg_len_fn=self.reader.avg_field_length,
            schema=self.reader.schema,
        )

    # -- search ----------------------------------------------------------
    def search(
        self,
        query: Query,
        top_k: int = 10,
        sort: Optional[Sort] = None,
        facets: Optional[List[str]] = None,
        highlight: Optional[List[str]] = None,
    ) -> SearchResult:
        ctx = self._context()
        facets = facets or []
        for fname in facets:
            if not self.schema.field(fname).stored:
                raise QueryError(f"cannot facet on {fname!r}: field is not stored")

        collected: List[Tuple[SegmentReader, int, float]] = []
        facet_counts: Dict[str, Counter] = {f: Counter() for f in facets}
        total = 0
        for seg in self.reader.segments:
            scorer = query.scorer(seg, ctx)
            doc = scorer.next()
            while doc != NO_MORE_DOCS:
                if not seg.is_deleted(doc):
                    total += 1
                    collected.append((seg, doc, scorer.score()))
                    if facets:
                        stored = seg.stored(doc)
                        for fname in facets:
                            value = stored.get(fname)
                            if value is not None:
                                facet_counts[fname][str(value)] += 1
                doc = scorer.next()

        ordered = self._order(collected, sort)
        top = ordered[:top_k]
        hits = [self._make_hit(seg, local, score, query, highlight) for seg, local, score in top]
        max_score = max((h.score for h in hits), default=0.0)
        result = SearchResult(hits=hits, total_hits=total, max_score=max_score)
        if facets:
            result.facets = {
                fname: [FacetCount(v, c) for v, c in counts.most_common()]
                for fname, counts in facet_counts.items()
            }
        return result

    def _order(
        self, collected: List[Tuple[SegmentReader, int, float]], sort: Optional[Sort]
    ) -> List[Tuple[SegmentReader, int, float]]:
        if sort is None or sort[0] == "score":
            reverse = True if sort is None else sort[1]
            return sorted(collected, key=lambda t: t[2], reverse=reverse)
        field_name, reverse = sort
        self.schema.field(field_name)  # validate existence

        def key(item: Tuple[SegmentReader, int, float]):
            seg, local, _ = item
            value = seg.numeric_value(field_name, local)
            # Missing values sort last regardless of direction.
            missing = value is None
            return (missing != reverse, value if value is not None else 0.0)

        return sorted(collected, key=key, reverse=reverse)

    def _make_hit(
        self,
        seg: SegmentReader,
        local: int,
        score: float,
        query: Query,
        highlight: Optional[List[str]],
    ) -> Hit:
        stored = seg.stored(local)
        hit = Hit(doc_id=seg.id_of(local), score=score, fields=stored)
        if highlight:
            from .highlight import Highlighter

            hl = Highlighter(self.schema)
            terms = query.extract_terms()
            for fname in highlight:
                text = stored.get(fname)
                if isinstance(text, str):
                    snippet = hl.highlight(fname, text, terms)
                    if snippet:
                        hit.highlights[fname] = snippet
        return hit

    # -- explanation -----------------------------------------------------
    def explain(self, query: Query, doc_id: str) -> Explanation:
        ctx = self._context()
        for seg in self.reader.segments:
            local = seg.local_of_id(doc_id)
            if local is None or seg.is_deleted(local):
                continue
            scorer = query.scorer(seg, ctx)
            if scorer.advance(local) != local:
                return Explanation(0.0, f"{doc_id!r} does not match the query")
            total = scorer.score()
            details = self._explain_terms(seg, local, query, ctx)
            return Explanation(total, f"score {total:.6f} for document {doc_id!r}", details)
        return Explanation(0.0, f"{doc_id!r} is not in the index")

    def _explain_terms(self, seg, local, query, ctx) -> List[Explanation]:
        details: List[Explanation] = []
        for fname, term in query.extract_terms():
            postings = {p.doc: p for p in seg.postings(fname, term)}
            posting = postings.get(local)
            if posting is None:
                continue
            length = seg.field_length(fname, local)
            doc_freq = ctx.doc_freq(fname, term)
            boost = self.schema.field(fname).boost
            weight = ctx.similarity.term_weight(doc_freq, ctx.collection_stats(fname), boost)
            child = weight.explain(posting.freq, length)
            details.append(Explanation(child.value, f"term {fname}:{term}", child.details))
        return details


def open_searcher(directory: str, similarity: str = "bm25") -> Searcher:
    return Searcher.open(directory, similarity)
