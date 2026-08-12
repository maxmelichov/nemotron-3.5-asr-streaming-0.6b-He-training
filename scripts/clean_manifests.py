#!/usr/bin/env python3
"""Drop rows whose transcript contains script the model should never be asked to emit.

The builder's text check only required *one* Hebrew character, so a row could still carry
Arabic, Cyrillic, CJK, emoji or control characters alongside it. Those are unlearnable
targets: the audio is Hebrew/English speech, the label is not.

Policy, deliberately conservative:

  * Rows are **dropped**, not stripped. Removing a foreign word would leave the audio
    saying something the transcript no longer contains -- a silent audio/text mismatch,
    which is worse for training than losing the row.
  * Cosmetic damage is repaired first (curly quotes, zero-width marks, NBSP), so rows are
    not thrown away over a typographic apostrophe.

Usage:
  python scripts/clean_manifests.py data/manifests/train.json --in-place
  python scripts/clean_manifests.py data/manifests/train.json --report-only
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# Code-point ranges we are willing to train on. Everything else is a script the model
# should never emit for Hebrew/English audio.
ALLOWED_RANGES = (
    (0x0590, 0x05FF),   # Hebrew: letters, niqqud, geresh/gershayim
    (0x0020, 0x007E),   # printable ASCII: English, digits, punctuation
    (0x00A0, 0x00BF),   # NBSP-adjacent punctuation, degree, guillemets
    (0x2010, 0x2027),   # dashes, quotes, ellipsis
    (0x20AA, 0x20AA),   # shekel
    (0x20AC, 0x20AC),   # euro
)
WHITESPACE = {0x09, 0x0A, 0x0D}
# Zero-width and bidi marks: invisible, and they poison tokenisation.
STRIP_CODEPOINTS = set(range(0x200B, 0x2010)) | set(range(0x202A, 0x202F)) | {0xFEFF}
QUOTE_MAP = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x00A0: " ", 0x2007: " ", 0x202F: " ",
}
HEBREW = re.compile("[" + chr(0x0590) + "-" + chr(0x05FF) + "]")


def allowed(char: str) -> bool:
    code = ord(char)
    if code in WHITESPACE:
        return True
    return any(low <= code <= high for low, high in ALLOWED_RANGES)


def repair(text: str) -> str:
    """Fix cosmetic damage so rows are not dropped over typography."""
    text = unicodedata.normalize("NFC", text)
    out = []
    for char in text:
        code = ord(char)
        if code in STRIP_CODEPOINTS:
            continue
        if code in QUOTE_MAP:
            out.append(QUOTE_MAP[code])
            continue
        if code < 0x20 and code not in WHITESPACE:
            continue
        out.append(char)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def offending(text: str) -> list[str]:
    return sorted({c for c in text if not allowed(c)})


def classify(char: str) -> str:
    try:
        return unicodedata.name(char).split()[0]
    except ValueError:
        return "U+%04X" % ord(char)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--min-chars", type=int, default=2)
    args = ap.parse_args()

    kept: list[str] = []
    dropped: Counter = Counter()
    samples: dict[str, str] = {}
    total = repaired = 0

    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            original = row.get("text", "")
            text = repair(original)
            if text != original:
                repaired += 1

            if len(text) < args.min_chars:
                dropped["too_short"] += 1
                continue
            if not HEBREW.search(text):
                dropped["no_hebrew"] += 1
                samples.setdefault("no_hebrew", text[:70])
                continue
            bad = offending(text)
            if bad:
                label = "script:" + classify(bad[0])
                dropped[label] += 1
                samples.setdefault(label, text[:70])
                continue

            row["text"] = text
            kept.append(json.dumps(row, ensure_ascii=False))

    removed = total - len(kept)
    print(f"{args.manifest.name}: {total:,} rows -> {len(kept):,} kept, {removed:,} dropped "
          f"({100 * removed / max(total, 1):.2f}%), {repaired:,} repaired")
    for reason, count in dropped.most_common(12):
        print(f"  {reason:30s} {count:>9,}  {samples.get(reason, '')}")

    if args.report_only:
        return
    target = args.manifest if args.in_place else (args.out or args.manifest.with_suffix(".clean.json"))
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
