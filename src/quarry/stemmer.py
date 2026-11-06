"""The Porter stemming algorithm.

This is a from-scratch implementation of M. F. Porter's 1980 suffix-stripping
algorithm ("An algorithm for suffix stripping", *Program* 14(3), pp. 130-137).
Stemming conflates morphological variants of a word to a common root so that a
query for ``connect`` also matches ``connection``, ``connected`` and
``connecting``.

The algorithm is a fixed sequence of rule-based steps. It is intentionally not
a dictionary lookup: it is fast, deterministic and language-specific to
English. The output "stem" is not always a real word (``ponies`` -> ``poni``),
which is fine, because the same transformation is applied at index and query
time.

Only :func:`stem` is public; everything else is an implementation detail of the
rule engine.
"""

from __future__ import annotations

_VOWELS = frozenset("aeiou")


class _Stemmer:
    """Mutable cursor over a single word while the rules rewrite it."""

    __slots__ = ("b", "k", "j")

    def __init__(self, word: str) -> None:
        self.b = list(word)
        self.k = len(word) - 1  # index of the last character under consideration
        self.j = 0  # general-purpose offset used by the rules

    def _is_consonant(self, i: int) -> bool:
        ch = self.b[i]
        if ch in _VOWELS:
            return False
        if ch == "y":
            # 'y' is a consonant only when preceded by a vowel-like position.
            return True if i == 0 else not self._is_consonant(i - 1)
        return True

    def _measure(self) -> int:
        """Count the number of consonant-vowel-consonant sequences in b[0..j]."""
        n = 0
        i = 0
        # Skip a leading run of consonants.
        while True:
            if i > self.j:
                return n
            if not self._is_consonant(i):
                break
            i += 1
        i += 1
        while True:
            # Each VC pair after the initial state increments the measure.
            while True:
                if i > self.j:
                    return n
                if self._is_consonant(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._is_consonant(i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self) -> bool:
        return any(not self._is_consonant(i) for i in range(self.j + 1))

    def _double_consonant(self, i: int) -> bool:
        if i < 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._is_consonant(i)

    def _cvc(self, i: int) -> bool:
        """True if b[i-2..i] is consonant-vowel-consonant and the last is not w/x/y.

        This test is used to restore a final ``e`` on short words.
        """
        if i < 2:
            return False
        if not self._is_consonant(i) or self._is_consonant(i - 1) or not self._is_consonant(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, suffix: str) -> bool:
        length = len(suffix)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1 : self.k + 1] != list(suffix):
            return False
        self.j = self.k - length
        return True

    def _set_to(self, suffix: str) -> None:
        self.b[self.j + 1 : self.k + 1] = list(suffix)
        self.k = self.j + len(suffix)

    def _replace(self, suffix: str) -> None:
        """Replace the matched suffix with ``suffix`` when the stem has m > 0."""
        if self._measure() > 0:
            self._set_to(suffix)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._set_to("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._measure() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowel_in_stem():
            self.k = self.j
            if self._ends("at"):
                self._set_to("ate")
            elif self._ends("bl"):
                self._set_to("ble")
            elif self._ends("iz"):
                self._set_to("ize")
            elif self._double_consonant(self.k):
                if self.b[self.k] not in "lsz":
                    self.k -= 1
            elif self._measure() == 1 and self._cvc(self.k):
                self._set_to("e")

    def _step1c(self) -> None:
        if self._ends("y") and self._vowel_in_stem():
            self.b[self.k] = "i"

    def _step2(self) -> None:
        ch = self.b[self.k - 1]
        if ch == "a":
            if self._ends("ational"):
                self._replace("ate")
            elif self._ends("tional"):
                self._replace("tion")
        elif ch == "c":
            if self._ends("enci"):
                self._replace("ence")
            elif self._ends("anci"):
                self._replace("ance")
        elif ch == "e":
            if self._ends("izer"):
                self._replace("ize")
        elif ch == "l":
            if self._ends("bli"):
                self._replace("ble")
            elif self._ends("alli"):
                self._replace("al")
            elif self._ends("entli"):
                self._replace("ent")
            elif self._ends("eli"):
                self._replace("e")
            elif self._ends("ousli"):
                self._replace("ous")
        elif ch == "o":
            if self._ends("ization"):
                self._replace("ize")
            elif self._ends("ation"):
                self._replace("ate")
            elif self._ends("ator"):
                self._replace("ate")
        elif ch == "s":
            if self._ends("alism"):
                self._replace("al")
            elif self._ends("iveness"):
                self._replace("ive")
            elif self._ends("fulness"):
                self._replace("ful")
            elif self._ends("ousness"):
                self._replace("ous")
        elif ch == "t":
            if self._ends("aliti"):
                self._replace("al")
            elif self._ends("iviti"):
                self._replace("ive")
            elif self._ends("biliti"):
                self._replace("ble")
        elif ch == "g":
            if self._ends("logi"):
                self._replace("log")

    def _step3(self) -> None:
        ch = self.b[self.k]
        if ch == "e":
            if self._ends("icate"):
                self._replace("ic")
            elif self._ends("ative"):
                self._replace("")
            elif self._ends("alize"):
                self._replace("al")
        elif ch == "i":
            if self._ends("iciti"):
                self._replace("ic")
        elif ch == "l":
            if self._ends("ical"):
                self._replace("ic")
            elif self._ends("ful"):
                self._replace("")
        elif ch == "s":
            if self._ends("ness"):
                self._replace("")

    def _step4(self) -> None:
        ch = self.b[self.k - 1]
        matched = False
        if ch == "a":
            matched = self._ends("al")
        elif ch == "c":
            matched = self._ends("ance") or self._ends("ence")
        elif ch == "e":
            matched = self._ends("er")
        elif ch == "i":
            matched = self._ends("ic")
        elif ch == "l":
            matched = self._ends("able") or self._ends("ible")
        elif ch == "n":
            matched = (
                self._ends("ant")
                or self._ends("ement")
                or self._ends("ment")
                or self._ends("ent")
            )
        elif ch == "o":
            matched = (self._ends("ion") and self.j >= 0 and self.b[self.j] in "st") or self._ends(
                "ou"
            )
        elif ch == "s":
            matched = self._ends("ism")
        elif ch == "t":
            matched = self._ends("ate") or self._ends("iti")
        elif ch == "u":
            matched = self._ends("ous")
        elif ch == "v":
            matched = self._ends("ive")
        elif ch == "z":
            matched = self._ends("ize")
        if matched and self._measure() > 1:
            self.k = self.j

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            m = self._measure()
            if m > 1 or (m == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._double_consonant(self.k) and self._measure() > 1:
            self.k -= 1

    def result(self) -> str:
        return "".join(self.b[: self.k + 1])


def stem(word: str) -> str:
    """Return the Porter stem of a single lowercase token.

    Words of two characters or fewer are returned unchanged. The input is
    expected to be lowercase ASCII letters; callers that may hold other
    characters should normalize first (the analysis pipeline does).
    """
    if len(word) <= 2:
        return word
    s = _Stemmer(word)
    s._step1ab()
    s._step1c()
    s._step2()
    s._step3()
    s._step4()
    s._step5()
    return s.result()
