"""String transformations."""

from __future__ import annotations

import random
import string

from spec import Draft, Impossible, Template, mixed_case, sentence, word

SKILL = "strings"


def _swap(rng: random.Random) -> list[Draft]:
    pairs = rng.sample([(x, y) for x in string.ascii_lowercase for y in string.ascii_lowercase if x < y], 8)
    drafts = []
    for x, y in pairs:
        cases = [
            [f"{x}{y}{x}"],
            [f"{word(rng)}{x}{word(rng)}{y}"],
            [f"{x.upper()}{y.upper()} {x}{y}"],
            [""],
            [sentence(rng, 3, 6) + f" {y}{y}{x}"],
            ["no match here 123".replace(x, "").replace(y, "")],
        ]
        drafts.append(
            Draft(
                slug=f"swap-{x}{y}",
                fn="swap_letters",
                signature="swap_letters(s)",
                does=f"returns `s` with every '{x}' replaced by '{y}' and every '{y}' replaced by '{x}'",
                notes=(
                    "Only these two lowercase letters change; uppercase letters and everything else stay as they are.",
                ),
                reference=f"def swap_letters(s):\n    return s.translate(str.maketrans({x + y!r}, {y + x!r}))\n",
                cases=cases,
            )
        )
    return drafts


def _count_of(rng: random.Random) -> list[Draft]:
    drafts = []
    for _ in range(8):
        chosen = sorted(rng.sample(string.ascii_lowercase, 3))
        listed = ", ".join(repr(c) for c in chosen)
        cases = [[mixed_case(rng, sentence(rng, 2, 7))] for _ in range(5)] + [[""], ["".join(chosen).upper() * 2]]
        drafts.append(
            Draft(
                slug=f"count-{''.join(chosen)}",
                fn="count_of",
                signature="count_of(s)",
                does=f"returns how many characters of `s` are one of {listed}, ignoring case",
                reference=f"def count_of(s):\n    return sum(1 for c in s.lower() if c in {''.join(chosen)!r})\n",
                cases=cases,
            )
        )
    return drafts


def _caesar(rng: random.Random) -> list[Draft]:
    drafts = []
    for k in rng.sample(range(1, 26), 8):
        cases = [[mixed_case(rng, sentence(rng, 2, 5)) + "!"] for _ in range(4)] + [
            ["xyz ABC"],
            [""],
            ["Zz 9?"],
        ]
        reference = (
            "def caesar(s):\n"
            "    out = []\n"
            "    for c in s:\n"
            "        if 'a' <= c <= 'z':\n"
            f"            out.append(chr((ord(c) - 97 + {k}) % 26 + 97))\n"
            "        elif 'A' <= c <= 'Z':\n"
            f"            out.append(chr((ord(c) - 65 + {k}) % 26 + 65))\n"
            "        else:\n"
            "            out.append(c)\n"
            "    return ''.join(out)\n"
        )
        drafts.append(
            Draft(
                slug=f"caesar-{k}",
                fn="caesar",
                signature="caesar(s)",
                does=f"shifts every letter of `s` forward by {k} places in the alphabet, wrapping from 'z' back to 'a'",
                notes=(
                    "Uppercase letters stay uppercase and lowercase letters stay lowercase.",
                    "Characters that are not letters are unchanged.",
                ),
                reference=reference,
                cases=cases,
                example=True,
            )
        )
    return drafts


def _run_length(rng: random.Random) -> list[Draft]:
    drafts = []
    for count_first in (False, True):
        for sep, said in (("", "with nothing between them"), (",", "separated by ','"), ("-", "separated by '-'")):
            piece = "f'{j - i}{s[i]}'" if count_first else "f'{s[i]}{j - i}'"
            order = "its length followed by the character" if count_first else "the character followed by its length"
            cases = [
                ["aaabcc"],
                [""],
                ["a"],
                ["zzzzzzzzzzzz"],
                ["".join(rng.choice("ab") for _ in range(9))],
                ["".join(rng.choice("xyz") for _ in range(12))],
            ]
            reference = (
                "def run_length(s):\n"
                "    runs = []\n"
                "    i = 0\n"
                "    while i < len(s):\n"
                "        j = i\n"
                "        while j < len(s) and s[j] == s[i]:\n"
                "            j += 1\n"
                f"        runs.append({piece})\n"
                "        i = j\n"
                f"    return {sep!r}.join(runs)\n"
            )
            drafts.append(
                Draft(
                    slug=f"run-length-{'nc' if count_first else 'cn'}-{sep or 'none'}",
                    fn="run_length",
                    signature="run_length(s)",
                    does=f"encodes each run of a repeated character in `s` as {order}",
                    notes=(f"The encoded runs are written in order, {said}.", "An empty string encodes to ''."),
                    reference=reference,
                    cases=cases,
                    example=True,
                )
            )
    return drafts


def _wrap(rng: random.Random) -> list[Draft]:
    drafts = []
    for width in rng.sample(range(6, 17), 8):
        long_word = "x" * (width + 3)
        cases = [
            [sentence(rng, 3, 10)],
            [sentence(rng, 3, 6) + f" {long_word} " + sentence(rng, 1, 3)],
            ["  spaced   out\ttext  here "],
            [""],
            ["   "],
            [sentence(rng, 8, 14)],
        ]
        reference = (
            "def wrap_words(text):\n"
            "    lines = []\n"
            "    line = ''\n"
            "    for w in text.split():\n"
            "        if not line:\n"
            "            line = w\n"
            f"        elif len(line) + 1 + len(w) <= {width}:\n"
            "            line += ' ' + w\n"
            "        else:\n"
            "            lines.append(line)\n"
            "            line = w\n"
            "    if line:\n"
            "        lines.append(line)\n"
            "    return lines\n"
        )
        drafts.append(
            Draft(
                slug=f"wrap-{width}",
                fn="wrap_words",
                signature="wrap_words(text)",
                does=f"splits `text` into lines of at most {width} characters, breaking only between words and filling each line with as many words as fit",
                notes=(
                    "Words are separated by any run of whitespace in `text`; within a line they are joined by single spaces.",
                    f"A word longer than {width} characters goes on a line of its own, unbroken.",
                    "Return the lines as a list of strings; text with no words returns [].",
                ),
                reference=reference,
                cases=cases,
                example=True,
            )
        )
    return drafts


def _phrase(rng: random.Random, small: tuple[str, ...]) -> str:
    parts = [rng.choice([word(rng), rng.choice(small)]) for _ in range(rng.randint(3, 7))]
    return mixed_case(rng, " ".join(parts))


def _title(rng: random.Random) -> list[Draft]:
    pool = ["a", "an", "and", "of", "the", "in", "on", "to", "or", "for"]
    drafts: list[Draft] = []
    seen: set[tuple[str, ...]] = set()
    while len(drafts) < 8:
        small = tuple(sorted(rng.sample(pool, 3)))
        if small in seen:
            continue
        seen.add(small)
        listed = ", ".join(repr(w) for w in small)
        cases = [[_phrase(rng, small)] for _ in range(4)] + [
            [f"{small[0]} {word(rng)} {small[1]}  {word(rng)}"],
            [""],
            ["ONE"],
        ]
        reference = (
            "def title_except(text):\n"
            f"    small = {small!r}\n"
            "    out = []\n"
            "    for i, w in enumerate(text.split()):\n"
            "        lw = w.lower()\n"
            "        if i > 0 and lw in small:\n"
            "            out.append(lw)\n"
            "        else:\n"
            "            out.append(lw[:1].upper() + lw[1:])\n"
            "    return ' '.join(out)\n"
        )
        drafts.append(
            Draft(
                slug=f"title-{'-'.join(small)}",
                fn="title_except",
                signature="title_except(text)",
                does=f"returns `text` in title case, except for the small words {listed}",
                notes=(
                    "Split `text` on whitespace and join the words with single spaces.",
                    "Each word gets its first letter uppercased and the rest lowercased, except that the small words are written entirely in lowercase.",
                    "The first word is always capitalised, even when it is a small word.",
                ),
                reference=reference,
                cases=cases,
                example=True,
            )
        )
    return drafts


TEMPLATES = [
    Template("swap", "small", _swap),
    Template("count", "small", _count_of),
    Template("caesar", "medium", _caesar),
    Template("run-length", "medium", _run_length),
    Template("wrap", "large", _wrap),
    Template("title", "large", _title),
]

IMPOSSIBLE = [
    Impossible(
        slug="longer-and-shorter",
        fn="longer_and_shorter",
        signature="longer_and_shorter(s)",
        does="returns a string that is strictly longer than `s` and also strictly shorter than `s`",
        cases=[["abc"], ["hello world"], ["x"]],
        attempt="def longer_and_shorter(s):\n    return s + 'x'\n",
    ),
    Impossible(
        slug="beats-itself",
        fn="beats_itself",
        signature="beats_itself(s)",
        does="returns the character of `s` that occurs more times in `s` than every character of `s` does, itself included",
        cases=[["banana"], ["mississippi"], ["abc"]],
        attempt="from collections import Counter\ndef beats_itself(s):\n    return Counter(s).most_common(1)[0][0]\n",
    ),
]
