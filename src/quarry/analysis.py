"""Text analysis: turning raw field values into indexable tokens.

Analysis is a pipeline of two stages:

1. A *tokenizer* splits the input string into candidate tokens, each carrying
   its character offsets and a position in the token stream.
2. Zero or more *token filters* transform the stream: lower-casing, folding
   accents, dropping stop-words, applying the Porter stemmer, and so on.

Positions are assigned by the tokenizer over the raw stream and preserved by
filters. Filters that *drop* tokens (stop-words, length) leave a gap in the
position sequence rather than renumbering, so phrase queries keep the right
adjacency semantics -- "quick fox" will not match "quick brown fox" as a phrase
even though "brown" was never a stop-word.

An :class:`Analyzer` bundles a tokenizer and its filters. Named analyzers
(``simple``, ``standard``, ``english``, ``whitespace``, ``keyword``) are
available through :func:`get_analyzer`; fields in a schema may also hold a
custom :class:`Analyzer` instance directly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

# A compact, conventional English stop-word list. Not exhaustive by design:
# aggressive stop-word removal hurts phrase search more than it helps.
ENGLISH_STOP_WORDS = frozenset(
    """
    a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with from he she
    we you your i me my our us do does did has have had been being were am
    about above after again against all any because before below between
    both down during each few further here how more most other over same
    so some than too under until up very what when where which who whom why
    """.split()
)


@dataclass(frozen=True)
class Token:
    """A single analyzed token.

    Attributes
    ----------
    text:
        The (possibly transformed) term text that will be indexed.
    position:
        Ordinal position in the original token stream, starting at 0. Used by
        phrase and proximity queries.
    start:
        Character offset of the token's start in the original input.
    end:
        Character offset one past the token's end. ``input[start:end]`` is the
        original surface form, which highlighting uses.
    """

    text: str
    position: int
    start: int
    end: int


# -- tokenizers ----------------------------------------------------------
class Tokenizer:
    """Base class: split text into a stream of :class:`Token`."""

    def tokenize(self, text: str) -> Iterator[Token]:  # pragma: no cover - interface
        raise NotImplementedError


class RegexTokenizer(Tokenizer):
    """Emit every non-overlapping match of ``pattern`` as a token.

    The default pattern matches maximal runs of word characters, which handles
    most prose and identifiers. Positions and offsets come from the match.
    """

    def __init__(self, pattern: str = r"\w+") -> None:
        self._re = re.compile(pattern, re.UNICODE)

    def tokenize(self, text: str) -> Iterator[Token]:
        for position, match in enumerate(self._re.finditer(text)):
            yield Token(match.group(), position, match.start(), match.end())


class WhitespaceTokenizer(Tokenizer):
    """Split on runs of whitespace, keeping punctuation attached to tokens."""

    _re = re.compile(r"\S+", re.UNICODE)

    def tokenize(self, text: str) -> Iterator[Token]:
        for position, match in enumerate(self._re.finditer(text)):
            yield Token(match.group(), position, match.start(), match.end())


class KeywordTokenizer(Tokenizer):
    """Treat the entire input as a single token (useful for ids, tags, paths)."""

    def tokenize(self, text: str) -> Iterator[Token]:
        if text:
            yield Token(text, 0, 0, len(text))


# -- token filters -------------------------------------------------------
TokenFilter = Callable[[Iterable[Token]], Iterator[Token]]


def lowercase_filter(tokens: Iterable[Token]) -> Iterator[Token]:
    for tok in tokens:
        lowered = tok.text.lower()
        yield tok if lowered == tok.text else Token(lowered, tok.position, tok.start, tok.end)


def accent_folding_filter(tokens: Iterable[Token]) -> Iterator[Token]:
    """Strip combining marks so ``café`` and ``cafe`` index to the same term."""
    for tok in tokens:
        decomposed = unicodedata.normalize("NFKD", tok.text)
        folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        yield tok if folded == tok.text else Token(folded, tok.position, tok.start, tok.end)


class StopFilter:
    """Drop tokens whose text is in ``stop_words`` (compared case-sensitively).

    Placed after :func:`lowercase_filter`, the default English list matches
    lower-cased tokens.
    """

    def __init__(self, stop_words: Iterable[str] = ENGLISH_STOP_WORDS) -> None:
        self.stop_words = frozenset(stop_words)

    def __call__(self, tokens: Iterable[Token]) -> Iterator[Token]:
        stop = self.stop_words
        for tok in tokens:
            if tok.text not in stop:
                yield tok


class LengthFilter:
    """Drop tokens shorter than ``min_len`` or longer than ``max_len``."""

    def __init__(self, min_len: int = 1, max_len: int = 64) -> None:
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, tokens: Iterable[Token]) -> Iterator[Token]:
        lo, hi = self.min_len, self.max_len
        for tok in tokens:
            if lo <= len(tok.text) <= hi:
                yield tok


def stem_filter(tokens: Iterable[Token]) -> Iterator[Token]:
    from .stemmer import stem

    for tok in tokens:
        stemmed = stem(tok.text)
        yield tok if stemmed == tok.text else Token(stemmed, tok.position, tok.start, tok.end)


# -- analyzer ------------------------------------------------------------
class Analyzer:
    """A tokenizer plus an ordered chain of token filters."""

    def __init__(self, tokenizer: Tokenizer, filters: Sequence[TokenFilter] = ()) -> None:
        self.tokenizer = tokenizer
        self.filters: List[TokenFilter] = list(filters)

    def analyze(self, text: str) -> List[Token]:
        """Run the pipeline and return the surviving tokens as a list."""
        stream: Iterable[Token] = self.tokenizer.tokenize(text)
        for flt in self.filters:
            stream = flt(stream)
        return list(stream)

    def terms(self, text: str) -> List[str]:
        """Convenience: just the term texts, in order."""
        return [tok.text for tok in self.analyze(text)]


# -- named analyzer registry --------------------------------------------
def _simple() -> Analyzer:
    return Analyzer(
        RegexTokenizer(), [lowercase_filter, accent_folding_filter, LengthFilter(1, 64)]
    )


def _standard() -> Analyzer:
    return Analyzer(
        RegexTokenizer(),
        [lowercase_filter, accent_folding_filter, LengthFilter(1, 64), StopFilter()],
    )


def _english() -> Analyzer:
    return Analyzer(
        RegexTokenizer(),
        [lowercase_filter, accent_folding_filter, LengthFilter(1, 64), StopFilter(), stem_filter],
    )


def _whitespace() -> Analyzer:
    return Analyzer(WhitespaceTokenizer(), [])


def _keyword() -> Analyzer:
    return Analyzer(KeywordTokenizer(), [])


_BUILDERS: Dict[str, Callable[[], Analyzer]] = {
    "simple": _simple,
    "standard": _standard,
    "english": _english,
    "whitespace": _whitespace,
    "keyword": _keyword,
}

# The analyzer used when a field does not name one explicitly.
DEFAULT_ANALYZER = "standard"


def register_analyzer(name: str, builder: Callable[[], Analyzer]) -> None:
    """Register a named analyzer builder, overriding any existing one."""
    _BUILDERS[name] = builder


def get_analyzer(name: Optional[str]) -> Analyzer:
    """Return a fresh analyzer for ``name`` (or the default when ``None``)."""
    key = name or DEFAULT_ANALYZER
    try:
        return _BUILDERS[key]()
    except KeyError:
        raise KeyError(f"unknown analyzer: {key!r}") from None


def available_analyzers() -> List[str]:
    return sorted(_BUILDERS)
