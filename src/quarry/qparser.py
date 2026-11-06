"""A small query language, parsed into the query AST.

The syntax is a familiar subset of the Lucene query language:

===========================  =================================================
``quick brown``              either term (default operator is OR)
``+quick +brown``            both terms required
``title:quick``              term in a specific field
``-draft``                   term must not appear
``"quick brown"``            exact phrase
``"quick brown"~2``          phrase with a slop of 2
``quick*``                   prefix query
``qu?ck``                    wildcard (``?`` one char, ``*`` any run)
``price:[10 TO 50]``         inclusive numeric range (``{`` / ``}`` exclusive)
``year:{2000 TO *}``         open-ended range
``(a OR b) AND c``           grouping and explicit AND/OR/NOT operators
===========================  =================================================

Bare terms search every indexed text field. Terms are run through the target
field's analyzer, so a query for ``Running`` matches an indexed ``run`` when the
field uses the English analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .errors import QueryParseError
from .query import (
    BooleanQuery,
    Clause,
    MatchAllQuery,
    NumericRangeQuery,
    Occur,
    PhraseQuery,
    PrefixQuery,
    Query,
    TermQuery,
    WildcardQuery,
)
from .schema import KeywordField, Schema

_PUNCT = {
    "(": "LP",
    ")": "RP",
    "[": "LB",
    "]": "RB",
    "{": "LBR",
    "}": "RBR",
    ":": "COLON",
    "~": "TILDE",
}
_KEYWORDS = {"AND", "OR", "NOT", "TO"}


@dataclass
class _Tok:
    kind: str
    value: str
    pos: int


def tokenize(text: str) -> List[_Tok]:
    tokens: List[_Tok] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 1
            if j >= n:
                raise QueryParseError("unterminated phrase", i)
            tokens.append(_Tok("PHRASE", text[i + 1 : j], i))
            i = j + 1
            continue
        if c in _PUNCT:
            tokens.append(_Tok(_PUNCT[c], c, i))
            i += 1
            continue
        if c in "+-":
            # A leading +/- is an operator when it starts a token (preceded by
            # the start, whitespace or an open paren) and is followed by a term.
            boundary = i == 0 or text[i - 1].isspace() or text[i - 1] == "("
            nxt = text[i + 1] if i + 1 < n else ""
            if boundary and nxt and not nxt.isspace() and nxt not in ")]}":
                tokens.append(_Tok("PLUS" if c == "+" else "MINUS", c, i))
                i += 1
                continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in _PUNCT and text[j] != '"':
            j += 1
        word = text[i:j]
        up = word.upper()
        tokens.append(_Tok(up if up in _KEYWORDS else "WORD", word, i))
        i = j
    return tokens


class QueryParser:
    """Parse query strings against a schema into :class:`Query` trees."""

    def __init__(
        self,
        schema: Schema,
        default_fields: Optional[List[str]] = None,
        default_occur: str = Occur.SHOULD,
    ) -> None:
        self.schema = schema
        self.default_fields = default_fields or schema.default_search_fields()
        if not self.default_fields:
            raise QueryParseError("schema has no searchable text fields")
        self.default_occur = default_occur
        self._tokens: List[_Tok] = []
        self._i = 0

    # -- parsing ---------------------------------------------------------
    def parse(self, text: str) -> Query:
        if text.strip() in ("*", "*:*"):
            return MatchAllQuery()
        self._tokens = tokenize(text)
        self._i = 0
        if not self._tokens:
            return MatchAllQuery()
        query = self._parse_expr()
        if not self._at_end():
            raise QueryParseError(f"unexpected token {self._cur().value!r}", self._cur().pos)
        return query

    def _parse_expr(self) -> Query:
        clauses: List[Clause] = []
        self._parse_clause_into(clauses, conj=None)
        while not self._at_end() and not self._peek("RP"):
            conj = None
            if self._peek("AND"):
                self._next()
                conj = "AND"
            elif self._peek("OR"):
                self._next()
                conj = "OR"
            self._parse_clause_into(clauses, conj)
        if len(clauses) == 1 and clauses[0].occur in (Occur.SHOULD, Occur.MUST):
            return clauses[0].query
        return BooleanQuery(clauses)

    def _parse_clause_into(self, clauses: List[Clause], conj: Optional[str]) -> None:
        mod = self._read_modifier()
        query = self._parse_subclause()
        if query is None:
            return
        if mod == "+":
            occur = Occur.MUST
        elif mod == "-":
            occur = Occur.MUST_NOT
        elif conj == "AND":
            occur = Occur.MUST
            if clauses and clauses[-1].occur == Occur.SHOULD:
                clauses[-1].occur = Occur.MUST
        elif conj == "OR":
            occur = Occur.SHOULD
        else:
            occur = self.default_occur
        clauses.append(Clause(query, occur))

    def _read_modifier(self) -> Optional[str]:
        if self._peek("PLUS"):
            self._next()
            return "+"
        if self._peek("MINUS") or self._peek("NOT"):
            self._next()
            return "-"
        return None

    def _parse_subclause(self) -> Optional[Query]:
        if self._peek("LP"):
            self._next()
            inner = self._parse_expr()
            self._expect("RP")
            return inner

        field = None
        if self._peek("WORD") and self._peek_at(1, "COLON"):
            name = self._cur().value
            if name not in self.schema:
                raise QueryParseError(f"unknown field {name!r}", self._cur().pos)
            field = name
            self._next()
            self._next()

        if self._peek("PHRASE"):
            return self._phrase(field)
        if self._peek("LB") or self._peek("LBR"):
            return self._range(field)
        if self._peek("WORD"):
            return self._term(field)
        raise QueryParseError("expected a term, phrase, range or group", self._pos())

    # -- leaf builders ---------------------------------------------------
    def _phrase(self, field: Optional[str]) -> Query:
        raw = self._cur().value
        self._next()
        slop = 0
        if self._peek("TILDE"):
            self._next()
            slop = self._read_int()
        fields = [field] if field else self.default_fields
        subqueries = []
        for fname in fields:
            self._require_text(fname)
            terms = self.schema.field(fname).analyzer().terms(raw)
            if terms:
                subqueries.append(PhraseQuery(fname, terms, slop))
        return self._any_of(subqueries)

    def _range(self, field: Optional[str]) -> Query:
        if field is None:
            raise QueryParseError("a range query must name a field", self._pos())
        self._require_numeric(field)
        open_tok = self._next()  # LB or LBR
        include_lower = open_tok.kind == "LB"
        lower = self._read_bound()
        if not self._peek("TO"):
            raise QueryParseError("expected 'TO' in range", self._pos())
        self._next()
        upper = self._read_bound()
        close = self._next()
        if close.kind not in ("RB", "RBR"):
            raise QueryParseError("unterminated range", close.pos)
        include_upper = close.kind == "RB"
        return NumericRangeQuery(field, lower, upper, include_lower, include_upper)

    def _term(self, field: Optional[str]) -> Optional[Query]:
        word = self._cur().value
        self._next()
        if self._peek("TILDE"):  # fuzzy distance is accepted but treated as the term
            self._next()
            if self._peek("WORD"):
                self._next()
        fields = [field] if field else self.default_fields
        subqueries: List[Query] = []
        for fname in fields:
            built = self._term_for_field(fname, word)
            if built is not None:
                subqueries.append(built)
        return self._any_of(subqueries)

    def _term_for_field(self, fname: str, word: str) -> Optional[Query]:
        field_obj = self.schema.field(fname)
        if field_obj.is_numeric:
            value = field_obj.coerce(word)
            return NumericRangeQuery(fname, value, value, True, True)
        if not field_obj.is_text:
            return None
        if "*" in word or "?" in word:
            pattern = self._normalize_wildcard(fname, word)
            if word.endswith("*") and "*" not in word[:-1] and "?" not in word:
                return PrefixQuery(fname, pattern[:-1])
            return WildcardQuery(fname, pattern)
        terms = field_obj.analyzer().terms(word)
        if not terms:
            return None
        if len(terms) == 1:
            return TermQuery(fname, terms[0])
        return BooleanQuery([Clause(TermQuery(fname, t), Occur.SHOULD) for t in terms])

    def _normalize_wildcard(self, fname: str, word: str) -> str:
        field_obj = self.schema.field(fname)
        if isinstance(field_obj, KeywordField) and not field_obj.lowercase:
            return word
        return word.lower()

    def _any_of(self, subqueries: List[Query]) -> Query:
        if not subqueries:
            return BooleanQuery([])  # matches nothing
        if len(subqueries) == 1:
            return subqueries[0]
        return BooleanQuery([Clause(q, Occur.SHOULD) for q in subqueries])

    # -- low-level helpers ----------------------------------------------
    def _read_bound(self) -> Optional[float]:
        tok = self._next()
        if tok.kind != "WORD":
            raise QueryParseError("expected a range bound", tok.pos)
        if tok.value == "*":
            return None
        try:
            return float(tok.value)
        except ValueError:
            raise QueryParseError(f"invalid numeric bound {tok.value!r}", tok.pos) from None

    def _read_int(self) -> int:
        tok = self._next()
        if tok.kind != "WORD" or not tok.value.isdigit():
            raise QueryParseError("expected an integer after '~'", tok.pos)
        return int(tok.value)

    def _require_text(self, fname: str) -> None:
        if not self.schema.field(fname).is_text:
            raise QueryParseError(f"field {fname!r} is not a text field", self._pos())

    def _require_numeric(self, fname: str) -> None:
        if not self.schema.field(fname).is_numeric:
            raise QueryParseError(f"range requires a numeric field, {fname!r} is not", self._pos())

    def _cur(self) -> _Tok:
        return self._tokens[self._i]

    def _pos(self) -> Optional[int]:
        return self._tokens[self._i].pos if self._i < len(self._tokens) else None

    def _at_end(self) -> bool:
        return self._i >= len(self._tokens)

    def _peek(self, kind: str) -> bool:
        return self._i < len(self._tokens) and self._tokens[self._i].kind == kind

    def _peek_at(self, offset: int, kind: str) -> bool:
        j = self._i + offset
        return j < len(self._tokens) and self._tokens[j].kind == kind

    def _next(self) -> _Tok:
        tok = self._tokens[self._i]
        self._i += 1
        return tok

    def _expect(self, kind: str) -> _Tok:
        if not self._peek(kind):
            raise QueryParseError(f"expected {kind}", self._pos())
        return self._next()


def parse_query(
    text: str,
    schema: Schema,
    default_fields: Optional[List[str]] = None,
    default_occur: str = Occur.SHOULD,
) -> Query:
    """Convenience wrapper around :class:`QueryParser`."""
    return QueryParser(schema, default_fields, default_occur).parse(text)
