"""Scoring-time text normalisation for Hebrew ASR.

WER is computed on whitespace-separated tokens, so punctuation attached to a word
makes the WHOLE word count as an error: predicting `לומר` where the reference says
`לומר,` is scored exactly as badly as predicting the wrong word. On our dev set 20.9%
of reference words carry punctuation, so raw WER conflates "got the word wrong" with
"got the comma wrong" -- and we only care about the former.

This module is scoring-only. It never rewrites a manifest and never touches training
targets: the model still learns to emit punctuation, we just stop charging it for
punctuation when measuring how well it heard the words.
"""
from __future__ import annotations

import re
import unicodedata

# Hebrew combining marks: niqqud + cantillation, EXCLUDING the punctuation code points
# that share the block (U+05BE MAQAF, U+05C0 PASEQ, U+05C3 SOF PASUQ, U+05C6 NUN HAFUKHA).
_NIQQUD = re.compile("[֑-ׇֽֿׁׂׅׄ]")
_WS = re.compile(r"\s+")

# Quote marks are NOT punctuation in Hebrew when they sit inside a word: gershayim marks
# an acronym (צה"ל) and geresh marks a modified consonant (ג'ון = "John"). Removing them
# merges genuinely different words, so word-internal quotes are KEPT and only edge quotes
# are stripped, where they really are quotation marks ("שלום" -> שלום).
# Hebrew gershayim/geresh fold onto their ASCII twins so a reference written ״ and a
# hypothesis written " are not scored as a mismatch.
_QUOTE_FOLD = {
    "״": '"',   # HEBREW PUNCTUATION GERSHAYIM
    "“": '"',   # LEFT DOUBLE QUOTATION MARK
    "”": '"',   # RIGHT DOUBLE QUOTATION MARK
    "׳": "'",   # HEBREW PUNCTUATION GERESH
    "‘": "'",   # LEFT SINGLE QUOTATION MARK
    "’": "'",   # RIGHT SINGLE QUOTATION MARK
}
_INNER_QUOTE = re.compile(r"(?<=\w)([\"'])(?=\w)")
# Private-use code points: never present in real transcripts, so parking word-internal
# quotes here hides them from the punctuation sweep with no risk of collision.
_PLACEHOLDER = {'"': "", "'": ""}


def normalize(text: str, *, strip_niqqud: bool = True) -> str:
    """Lowercase, drop punctuation/symbols, optionally drop niqqud, collapse spaces.

    Word-internal quotes survive: `צה"ל` stays `צה"ל` and is NOT equated with `צהל`.
    """
    text = unicodedata.normalize("NFC", text or "")
    if strip_niqqud:
        text = _NIQQUD.sub("", text)
    for src, dst in _QUOTE_FOLD.items():
        text = text.replace(src, dst)
    # Repeat until stable so every quote in a multi-quote token is parked (ר"ה"ב):
    # each pass consumes one, because a matched quote is no longer a \w boundary.
    for _ in range(4):
        parked = _INNER_QUOTE.sub(lambda m: _PLACEHOLDER[m.group(1)], text)
        if parked == text:
            break
        text = parked
    # Category P* (punctuation) and S* (symbols) -> space, so `word,word` splits rather
    # than fusing into one token.
    text = "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text)
    for ch, placeholder in _PLACEHOLDER.items():
        text = text.replace(placeholder, ch)
    return _WS.sub(" ", text).strip().lower()


def wer(references: list[str], hypotheses: list[str], *, normalized: bool = True) -> tuple[float, int, int]:
    """Return (wer, edit_distance, reference_word_count) over a corpus."""
    errors = 0
    words = 0
    for ref, hyp in zip(references, hypotheses):
        if normalized:
            ref, hyp = normalize(ref), normalize(hyp)
        r = ref.split()
        h = hyp.split()
        errors += _levenshtein(r, h)
        words += len(r)
    return (errors / words if words else 0.0), errors, words


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Word-level edit distance, O(len(b)) memory."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj))
        prev = cur
    return prev[-1]
