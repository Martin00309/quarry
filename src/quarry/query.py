"""The query model and its execution.

Queries are an abstract syntax tree (:class:`TermQuery`, :class:`BooleanQuery`,
:class:`PhraseQuery`, and so on). Execution is document-at-a-time: each query
builds a :class:`Scorer` for a given segment, and a scorer is an iterator over
matching local document ids that can also report a relevance score.

Scorers compose: a boolean query's scorer is built from the scorers of its
clauses (a conjunction for ``MUST``, a disjunction for ``SHOULD``, an exclusion
for ``MUST_NOT``). Scored leaf queries (``TermQuery``, ``PhraseQuery``) use the
collection's similarity; multi-term queries (prefix, wildcard, range) use a
constant score, the conventional "constant score rewrite".

A query never reaches across segments itself; the :class:`~quarry.searcher.Searcher`
runs each segment and merges the results. Collection-wide statistics (document
frequency, average field length) are supplied through the ``context`` object so
that idf is computed once over the whole index, not per segment.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .postings import NO_MORE_DOCS, PostingCursor
from .similarity import CollectionStatistics


# -- execution context ---------------------------------------------------
class SearchContext:
    """Collection-wide statistics and settings needed to build scorers.

    Provided by the searcher. ``doc_freq`` and ``avg_field_length`` are global
    across all live segments, which is what correct idf weighting requires.
    """

    def __init__(self, similarity, num_docs, doc_freq_fn, avg_len_fn, schema) -> None:
        self.similarity = similarity
        self.num_docs = num_docs
        self._doc_freq_fn = doc_freq_fn
        self._avg_len_fn = avg_len_fn
        self.schema = schema

    def doc_freq(self, field_name: str, term: str) -> int:
        return self._doc_freq_fn(field_name, term)

    def avg_field_length(self, field_name: str) -> float:
        return self._avg_len_fn(field_name)

    def collection_stats(self, field_name: str) -> CollectionStatistics:
        return CollectionStatistics(self.num_docs, self.avg_field_length(field_name))


# -- scorers -------------------------------------------------------------
class Scorer:
    """Iterator over matching local doc ids, with a score for the current one."""

    doc_id: int = -1

    def next(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def advance(self, target: int) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def score(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class EmptyScorer(Scorer):
    """A scorer that matches nothing."""

    def __init__(self) -> None:
        self.doc_id = NO_MORE_DOCS

    def next(self) -> int:
        return NO_MORE_DOCS

    def advance(self, target: int) -> int:
        return NO_MORE_DOCS

    def score(self) -> float:
        return 0.0


class TermScorer(Scorer):
    """Scores a single term using the active similarity."""

    def __init__(self, cursor: PostingCursor, weight, segment, field_name: str) -> None:
        self._cursor = cursor
        self._weight = weight
        self._segment = segment
        self._field = field_name
        self.doc_id = -1

    def next(self) -> int:
        self.doc_id = self._cursor.next()
        return self.doc_id

    def advance(self, target: int) -> int:
        self.doc_id = self._cursor.advance(target)
        return self.doc_id

    def score(self) -> float:
        length = self._segment.field_length(self._field, self.doc_id)
        return self._weight.score(self._cursor.freq, length)


class ConstantScorer(Scorer):
    """Emits a fixed score for every doc in a cursor (multi-term rewrite)."""

    def __init__(self, cursor: PostingCursor, boost: float) -> None:
        self._cursor = cursor
        self._boost = boost
        self.doc_id = -1

    def next(self) -> int:
        self.doc_id = self._cursor.next()
        return self.doc_id

    def advance(self, target: int) -> int:
        self.doc_id = self._cursor.advance(target)
        return self.doc_id

    def score(self) -> float:
        return self._boost


class MatchAllScorer(Scorer):
    """Matches every document in the segment (deletions filtered downstream)."""

    def __init__(self, doc_count: int, boost: float) -> None:
        self._doc_count = doc_count
        self._boost = boost
        self.doc_id = -1

    def next(self) -> int:
        self.doc_id = 0 if self.doc_id < 0 else self.doc_id + 1
        if self.doc_id >= self._doc_count:
            self.doc_id = NO_MORE_DOCS
        return self.doc_id

    def advance(self, target: int) -> int:
        self.doc_id = max(target, 0)
        if self.doc_id >= self._doc_count:
            self.doc_id = NO_MORE_DOCS
        return self.doc_id

    def score(self) -> float:
        return self._boost


class RangeScorer(Scorer):
    """Matches docs whose numeric column value falls in a range."""

    def __init__(self, matches: List[int], boost: float) -> None:
        self._matches = matches
        self._i = -1
        self._boost = boost
        self.doc_id = -1

    def next(self) -> int:
        self._i += 1
        self.doc_id = self._matches[self._i] if self._i < len(self._matches) else NO_MORE_DOCS
        return self.doc_id

    def advance(self, target: int) -> int:
        self._i = bisect.bisect_left(self._matches, target)
        self.doc_id = self._matches[self._i] if self._i < len(self._matches) else NO_MORE_DOCS
        return self.doc_id

    def score(self) -> float:
        return self._boost


class DisjunctionScorer(Scorer):
    """Union of sub-scorers; matches a doc shared by >= ``min_match`` of them."""

    def __init__(self, subs: Sequence[Scorer], min_match: int = 1) -> None:
        self._subs = list(subs)
        self._min = max(1, min_match)
        for s in self._subs:
            s.next()
        self.doc_id = -1

    def next(self) -> int:
        cur = self.doc_id
        if cur != -1:
            for s in self._subs:
                if s.doc_id == cur:
                    s.next()
        return self._settle()

    def advance(self, target: int) -> int:
        for s in self._subs:
            if s.doc_id < target:
                s.advance(target)
        return self._settle()

    def _settle(self) -> int:
        while True:
            m = min((s.doc_id for s in self._subs), default=NO_MORE_DOCS)
            if m == NO_MORE_DOCS:
                self.doc_id = NO_MORE_DOCS
                return self.doc_id
            if sum(1 for s in self._subs if s.doc_id == m) >= self._min:
                self.doc_id = m
                return m
            for s in self._subs:
                if s.doc_id == m:
                    s.next()

    def score(self) -> float:
        cur = self.doc_id
        return sum(s.score() for s in self._subs if s.doc_id == cur)


class ConjunctionScorer(Scorer):
    """Intersection of sub-scorers; score is the sum of all sub-scores."""

    def __init__(self, subs: Sequence[Scorer]) -> None:
        self._subs = list(subs)
        for s in self._subs:
            s.next()
        self.doc_id = -1

    def next(self) -> int:
        if self.doc_id == NO_MORE_DOCS:
            return NO_MORE_DOCS
        if self.doc_id != -1:
            self._subs[0].advance(self.doc_id + 1)
        return self._align()

    def advance(self, target: int) -> int:
        if self.doc_id == NO_MORE_DOCS:
            return NO_MORE_DOCS
        self._subs[0].advance(target)
        return self._align()

    def _align(self) -> int:
        lead = self._subs[0]
        while True:
            target = lead.doc_id
            if target == NO_MORE_DOCS:
                self.doc_id = NO_MORE_DOCS
                return self.doc_id
            ok = True
            for s in self._subs:
                if s is lead:
                    continue
                d = s.doc_id
                if d < target:
                    d = s.advance(target)
                if d > target:
                    lead = s
                    ok = False
                    break
            if ok:
                self.doc_id = target
                return target

    def score(self) -> float:
        return sum(s.score() for s in self._subs)


class ReqOptScorer(Scorer):
    """Required scorer whose score is augmented by any optional scorers."""

    def __init__(self, required: Scorer, optional: Sequence[Scorer]) -> None:
        self._req = required
        self._opt = list(optional)
        for s in self._opt:
            s.next()
        self.doc_id = -1

    def next(self) -> int:
        self.doc_id = self._req.next()
        return self.doc_id

    def advance(self, target: int) -> int:
        self.doc_id = self._req.advance(target)
        return self.doc_id

    def score(self) -> float:
        total = self._req.score()
        for o in self._opt:
            if o.doc_id < self.doc_id:
                o.advance(self.doc_id)
            if o.doc_id == self.doc_id:
                total += o.score()
        return total


class ExclusionScorer(Scorer):
    """Positive scorer minus any doc matched by the excluded scorer."""

    def __init__(self, positive: Scorer, excluded: Scorer) -> None:
        self._pos = positive
        self._excl = excluded
        self._excl.next()
        self.doc_id = -1

    def _is_excluded(self, doc: int) -> bool:
        if self._excl.doc_id < doc:
            self._excl.advance(doc)
        return self._excl.doc_id == doc

    def next(self) -> int:
        d = self._pos.next()
        while d != NO_MORE_DOCS and self._is_excluded(d):
            d = self._pos.next()
        self.doc_id = d
        return d

    def advance(self, target: int) -> int:
        d = self._pos.advance(target)
        while d != NO_MORE_DOCS and self._is_excluded(d):
            d = self._pos.next()
        self.doc_id = d
        return d

    def score(self) -> float:
        return self._pos.score()


class PhraseScorer(Scorer):
    """Matches docs where the phrase terms occur in order within ``slop``."""

    def __init__(self, cursors, weights, segment, field_name, slop) -> None:
        self._cursors = cursors
        self._weights = weights
        self._segment = segment
        self._field = field_name
        self._slop = slop
        for c in self._cursors:
            c.next()
        self._freq = 0
        self.doc_id = -1

    def next(self) -> int:
        start = 0 if self.doc_id < 0 else self.doc_id + 1
        return self._find(start)

    def advance(self, target: int) -> int:
        return self._find(target)

    def _find(self, start: int) -> int:
        while True:
            doc = _next_common_doc(self._cursors, start)
            if doc == NO_MORE_DOCS:
                self.doc_id = NO_MORE_DOCS
                self._freq = 0
                return self.doc_id
            freq = phrase_freq([list(c.positions) for c in self._cursors], self._slop)
            if freq > 0:
                self.doc_id = doc
                self._freq = freq
                return doc
            start = doc + 1

    def score(self) -> float:
        length = self._segment.field_length(self._field, self.doc_id)
        return sum(w.score(self._freq, length) for w in self._weights)


def _next_common_doc(cursors, start: int) -> int:
    for c in cursors:
        if c.doc < start:
            c.advance(start)
    while True:
        docs = [c.doc for c in cursors]
        if NO_MORE_DOCS in docs:
            return NO_MORE_DOCS
        m = max(docs)
        if all(d == m for d in docs):
            return m
        for c in cursors:
            if c.doc < m:
                c.advance(m)


def phrase_freq(position_lists: List[List[int]], slop: int) -> int:
    """Count in-order phrase matches across per-term position lists.

    With ``slop == 0`` the positions must be exactly consecutive. With a
    positive slop, terms may be spread apart as long as the total gap (the sum
    of missing positions between consecutive terms) does not exceed ``slop``.
    """
    if not position_lists or any(not pl for pl in position_lists):
        return 0
    if len(position_lists) == 1:
        return len(position_lists[0])
    if slop == 0:
        common = {p - 0 for p in position_lists[0]}
        for i in range(1, len(position_lists)):
            common &= {p - i for p in position_lists[i]}
            if not common:
                return 0
        return len(common)
    count = 0
    for p0 in position_lists[0]:
        prev = p0
        cost = 0
        ok = True
        for i in range(1, len(position_lists)):
            pl = position_lists[i]
            j = bisect.bisect_right(pl, prev)
            if j >= len(pl):
                ok = False
                break
            cost += pl[j] - prev - 1
            prev = pl[j]
        if ok and cost <= slop:
            count += 1
    return count


# -- query AST -----------------------------------------------------------
class Query:
    """Base class for all query nodes."""

    def scorer(self, segment, ctx: SearchContext) -> Scorer:  # pragma: no cover - interface
        raise NotImplementedError

    def extract_terms(self) -> List[Tuple[str, str]]:
        return []


@dataclass
class MatchAllQuery(Query):
    boost: float = 1.0

    def scorer(self, segment, ctx: SearchContext) -> Scorer:
        return MatchAllScorer(segment.doc_count, self.boost)

    def __repr__(self) -> str:
        return "*:*"


@dataclass
class TermQuery(Query):
    field: str
    term: str
    boost: float = 1.0

    def scorer(self, segment, ctx: SearchContext) -> Scorer:
        info = segment.term_info(self.field, self.term)
        if info is None:
            return EmptyScorer()
        doc_freq = ctx.doc_freq(self.field, self.term)
        field_boost = self.boost * ctx.schema.field(self.field).boost
        weight = ctx.similarity.term_weight(doc_freq, ctx.collection_stats(self.field), field_boost)
        cursor = PostingCursor(segment.postings(self.field, self.term))
        return TermScorer(cursor, weight, segment, self.field)

    def extract_terms(self) -> List[Tuple[str, str]]:
        return [(self.field, self.term)]

    def __repr__(self) -> str:
        return f"{self.field}:{self.term}"


@dataclass
class PhraseQuery(Query):
    field: str
    terms: List[str]
    slop: int = 0
    boost: float = 1.0

    def scorer(self, segment, ctx: SearchContext) -> Scorer:
        if not self.terms:
            return EmptyScorer()
        cursors = []
        weights = []
        for term in self.terms:
            info = segment.term_info(self.field, term)
            if info is None:
                return EmptyScorer()
            cursors.append(PostingCursor(segment.postings(self.field, term)))
            doc_freq = ctx.doc_freq(self.field, term)
            field_boost = self.boost * ctx.schema.field(self.field).boost
            weights.append(
                ctx.similarity.term_weight(doc_freq, ctx.collection_stats(self.field), field_boost)
            )
        return PhraseScorer(cursors, weights, segment, self.field, self.slop)

    def extract_terms(self) -> List[Tuple[str, str]]:
        return [(self.field, t) for t in self.terms]

    def __repr__(self) -> str:
        quoted = " ".join(self.terms)
        return f'{self.field}:"{quoted}"' + (f"~{self.slop}" if self.slop else "")


@dataclass
class PrefixQuery(Query):
    field: str
    prefix: str
    boost: float = 1.0

    def _expand(self, segment) -> List[str]:
        return [t for t in segment.field_terms(self.field) if t.startswith(self.prefix)]

    def scorer(self, segment, ctx: SearchContext) -> Scorer:
        subs = [
            ConstantScorer(PostingCursor(segment.postings(self.field, term)), self.boost)
            for term in self._expand(segment)
        ]
        return DisjunctionScorer(subs) if subs else EmptyScorer()

    def __repr__(self) -> str:
        return f"{self.field}:{self.prefix}*"


@dataclass
class WildcardQuery(Query):
    field: str
    pattern: str
    boost: float = 1.0

    def _regex(self) -> "re.Pattern[str]":
        out = ["^"]
        for ch in self.pattern:
            if ch == "*":
                out.append(".*")
            elif ch == "?":
                out.append(".")
            else:
                out.append(re.escape(ch))
        out.append("$")
        return re.compile("".join(out))

    def scorer(self, segment, ctx: SearchContext) -> Scorer:
        rx = self._regex()
        subs = [
            ConstantScorer(PostingCursor(segment.postings(self.field, term)), self.boost)
            for term in segment.field_terms(self.field)
            if rx.match(term)
        ]
        return DisjunctionScorer(subs) if subs else EmptyScorer()

    def __repr__(self) -> str:
        return f"{self.field}:{self.pattern}"


@dataclass
class NumericRangeQuery(Query):
    field: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    include_lower: bool = True
    include_upper: bool = True
    boost: float = 1.0

    def _in_range(self, value: float) -> bool:
        if self.lower is not None:
            if value < self.lower or (value == self.lower and not self.include_lower):
                return False
        if self.upper is not None:
            if value > self.upper or (value == self.upper and not self.include_upper):
                return False
        return True

    def scorer(self, segment, ctx: SearchContext) -> Scorer:
        matches = []
        for local in range(segment.doc_count):
            value = segment.numeric_value(self.field, local)
            if value is not None and self._in_range(value):
                matches.append(local)
        return RangeScorer(matches, self.boost) if matches else EmptyScorer()

    def __repr__(self) -> str:
        lo = "*" if self.lower is None else self.lower
        hi = "*" if self.upper is None else self.upper
        return f"{self.field}:[{lo} TO {hi}]"


class Occur:
    """Clause roles inside a boolean query."""

    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"


@dataclass
class Clause:
    query: Query
    occur: str = Occur.SHOULD


@dataclass
class BooleanQuery(Query):
    clauses: List[Clause] = field(default_factory=list)
    minimum_should_match: int = 0
    boost: float = 1.0

    def add(self, query: Query, occur: str = Occur.SHOULD) -> "BooleanQuery":
        self.clauses.append(Clause(query, occur))
        return self

    def _by(self, occur: str) -> List[Query]:
        return [c.query for c in self.clauses if c.occur == occur]

    def scorer(self, segment, ctx: SearchContext) -> Scorer:
        must = [q.scorer(segment, ctx) for q in self._by(Occur.MUST)]
        should = [q.scorer(segment, ctx) for q in self._by(Occur.SHOULD)]
        must_not = [q.scorer(segment, ctx) for q in self._by(Occur.MUST_NOT)]

        if any(isinstance(s, EmptyScorer) for s in must):
            return EmptyScorer()

        if must:
            required: Scorer = must[0] if len(must) == 1 else ConjunctionScorer(must)
            if should:
                required = ReqOptScorer(required, should)
        elif should:
            min_match = self.minimum_should_match or 1
            if min_match > len(should):
                return EmptyScorer()
            required = should[0] if len(should) == 1 and min_match == 1 else DisjunctionScorer(
                should, min_match
            )
        elif must_not:
            required = MatchAllScorer(segment.doc_count, self.boost)
        else:
            return EmptyScorer()

        for excl in must_not:
            if not isinstance(excl, EmptyScorer):
                required = ExclusionScorer(required, excl)
        return required

    def extract_terms(self) -> List[Tuple[str, str]]:
        terms: List[Tuple[str, str]] = []
        for clause in self.clauses:
            if clause.occur != Occur.MUST_NOT:
                terms.extend(clause.query.extract_terms())
        return terms

    def __repr__(self) -> str:
        sign = {Occur.MUST: "+", Occur.SHOULD: "", Occur.MUST_NOT: "-"}
        return "(" + " ".join(f"{sign[c.occur]}{c.query!r}" for c in self.clauses) + ")"
