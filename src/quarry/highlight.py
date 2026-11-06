"""Snippet generation with term highlighting.

Given a stored field value and the terms a query matched, the highlighter
returns a short passage of the text centred on the densest cluster of matches,
with each matching term wrapped in marker tags. It re-runs the field's analyzer
so that the same stemming/normalization used at index time decides what counts
as a match, but it marks the *original* surface form in the text.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .analysis import Token
from .schema import Schema


class Highlighter:
    """Produce highlighted snippets for a field."""

    def __init__(
        self,
        schema: Schema,
        pre_tag: str = "<b>",
        post_tag: str = "</b>",
        max_chars: int = 160,
        ellipsis: str = "…",
    ) -> None:
        self.schema = schema
        self.pre = pre_tag
        self.post = post_tag
        self.max_chars = max_chars
        self.ellipsis = ellipsis

    def highlight(
        self, field_name: str, text: str, terms: Iterable[Tuple[str, str]]
    ) -> Optional[str]:
        field = self.schema.field(field_name)
        if not field.is_text:
            return None
        wanted = {term for fname, term in terms if fname == field_name}
        if not wanted:
            return None
        tokens: List[Token] = field.analyzer().analyze(text)
        matched = [tok for tok in tokens if tok.text in wanted]
        if not matched:
            return None

        anchor = self._best_anchor(matched)
        frag_start = self._snap_left(text, anchor)
        frag_end = self._snap_right(text, min(len(text), anchor + self.max_chars))

        pieces: List[str] = []
        cursor = frag_start
        for tok in matched:
            if tok.start < cursor or tok.end > frag_end:
                continue
            pieces.append(text[cursor : tok.start])
            pieces.append(self.pre)
            pieces.append(text[tok.start : tok.end])
            pieces.append(self.post)
            cursor = tok.end
        pieces.append(text[cursor:frag_end])

        snippet = "".join(pieces).strip()
        if frag_start > 0:
            snippet = self.ellipsis + snippet
        if frag_end < len(text):
            snippet = snippet + self.ellipsis
        return snippet

    def _best_anchor(self, matched: List[Token]) -> int:
        best_start = matched[0].start
        best_count = -1
        for anchor in matched:
            window_end = anchor.start + self.max_chars
            count = sum(1 for other in matched if anchor.start <= other.start < window_end)
            if count > best_count:
                best_count = count
                best_start = anchor.start
        return best_start

    @staticmethod
    def _snap_left(text: str, pos: int) -> int:
        i = pos
        while i > 0 and not text[i - 1].isspace():
            i -= 1
        return i

    @staticmethod
    def _snap_right(text: str, pos: int) -> int:
        i = pos
        while i < len(text) and not text[i].isspace():
            i += 1
        return i
