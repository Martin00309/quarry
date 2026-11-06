"""Tests for the analysis pipeline."""

from __future__ import annotations

from quarry.analysis import (
    Analyzer,
    KeywordTokenizer,
    LengthFilter,
    RegexTokenizer,
    StopFilter,
    WhitespaceTokenizer,
    accent_folding_filter,
    get_analyzer,
    lowercase_filter,
    stem_filter,
)


def test_regex_tokenizer_offsets_and_positions():
    tokens = list(RegexTokenizer().tokenize("Hello, world!"))
    assert [t.text for t in tokens] == ["Hello", "world"]
    assert [t.position for t in tokens] == [0, 1]
    assert tokens[0].start == 0 and tokens[0].end == 5
    assert tokens[1].start == 7 and tokens[1].end == 12


def test_whitespace_tokenizer_keeps_punctuation():
    tokens = list(WhitespaceTokenizer().tokenize("foo-bar baz!"))
    assert [t.text for t in tokens] == ["foo-bar", "baz!"]


def test_keyword_tokenizer_is_single_token():
    tokens = list(KeywordTokenizer().tokenize("New York City"))
    assert len(tokens) == 1
    assert tokens[0].text == "New York City"


def test_lowercase_filter():
    analyzer = Analyzer(RegexTokenizer(), [lowercase_filter])
    assert analyzer.terms("Hello WORLD") == ["hello", "world"]


def test_accent_folding():
    analyzer = Analyzer(RegexTokenizer(), [lowercase_filter, accent_folding_filter])
    assert analyzer.terms("café RÉSUMÉ") == ["cafe", "resume"]


def test_stop_filter_removes_but_keeps_positions():
    analyzer = Analyzer(RegexTokenizer(), [lowercase_filter, StopFilter()])
    tokens = analyzer.analyze("the quick brown fox")
    assert [t.text for t in tokens] == ["quick", "brown", "fox"]
    # 'the' was position 0 and is dropped; remaining positions keep their gap.
    assert [t.position for t in tokens] == [1, 2, 3]


def test_length_filter():
    analyzer = Analyzer(RegexTokenizer(), [LengthFilter(min_len=3, max_len=5)])
    assert analyzer.terms("a be cat hello enormous") == ["cat", "hello"]


def test_stem_filter():
    analyzer = Analyzer(RegexTokenizer(), [lowercase_filter, stem_filter])
    assert analyzer.terms("running runner runs") == ["run", "runner", "run"]


def test_english_analyzer_pipeline():
    analyzer = get_analyzer("english")
    # stop-words removed, remaining terms stemmed
    assert analyzer.terms("The cats are running quickly") == ["cat", "run", "quickli"]


def test_standard_vs_simple_stopwords():
    simple = get_analyzer("simple")
    standard = get_analyzer("standard")
    assert "the" in simple.terms("the dog")
    assert "the" not in standard.terms("the dog")


def test_keyword_analyzer_passthrough():
    analyzer = get_analyzer("keyword")
    assert analyzer.terms("Hello World") == ["Hello World"]


def test_phrase_positions_survive_transforms():
    analyzer = get_analyzer("english")
    tokens = analyzer.analyze("the quick brown fox")
    by_text = {t.text: t.position for t in tokens}
    # quick and brown remain adjacent after stop-word removal
    assert by_text["brown"] - by_text["quick"] == 1
