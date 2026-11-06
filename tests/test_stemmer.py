"""Tests for the Porter stemmer against known reference outputs."""

from __future__ import annotations

import pytest

from quarry.stemmer import stem

# A sampling of the canonical Porter-stemmer vocabulary/output pairs.
CASES = {
    "caresses": "caress",
    "ponies": "poni",
    "ties": "ti",
    "caress": "caress",
    "cats": "cat",
    "feed": "feed",
    "agreed": "agre",
    "plastered": "plaster",
    "motoring": "motor",
    "sing": "sing",
    "conflated": "conflat",
    "troubling": "troubl",
    "sized": "size",
    "hopping": "hop",
    "falling": "fall",
    "hissing": "hiss",
    "fizzed": "fizz",
    "failing": "fail",
    "filing": "file",
    "happy": "happi",
    "relational": "relat",
    "conditional": "condit",
    "rational": "ration",
    "valenci": "valenc",
    "digitizer": "digit",
    "conformabli": "conform",
    "radicalli": "radic",
    "vietnamization": "vietnam",
    "predication": "predic",
    "operator": "oper",
    "feudalism": "feudal",
    "decisiveness": "decis",
    "hopefulness": "hope",
    "callousness": "callous",
    "formaliti": "formal",
    "sensitiviti": "sensit",
    "sensibiliti": "sensibl",
    "triplicate": "triplic",
    "formative": "form",
    "formalize": "formal",
    "electriciti": "electr",
    "electrical": "electr",
    "hopeful": "hope",
    "goodness": "good",
    "revival": "reviv",
    "allowance": "allow",
    "inference": "infer",
    "airliner": "airlin",
    "gyroscopic": "gyroscop",
    "adjustable": "adjust",
    "defensible": "defens",
    "irritant": "irrit",
    "replacement": "replac",
    "adjustment": "adjust",
    "dependent": "depend",
    "adoption": "adopt",
    "homologou": "homolog",
    "communism": "commun",
    "activate": "activ",
    "angulariti": "angular",
    "homologous": "homolog",
    "effective": "effect",
    "bowdlerize": "bowdler",
    "probate": "probat",
    "rate": "rate",
    "cease": "ceas",
    "controll": "control",
    "roll": "roll",
}


@pytest.mark.parametrize("word,expected", list(CASES.items()))
def test_porter_reference_cases(word, expected):
    assert stem(word) == expected


def test_short_words_unchanged():
    assert stem("a") == "a"
    assert stem("be") == "be"
    assert stem("i") == "i"


def test_morphological_variants_conflate():
    assert stem("connect") == stem("connected") == stem("connecting") == stem("connection")
