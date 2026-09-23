"""Dictionary manipulation."""

from __future__ import annotations

import random
from typing import Any

from spec import Draft, Impossible, Template, mixed_case, sentence, word, words

SKILL = "mappings"

SEP_NAMES = {".": "dot", "/": "slash", "_": "underscore", ":": "colon"}


def _mapping(rng: random.Random, low: int, high: int, lo: int = 0, hi: int = 9) -> dict[str, int]:
    return {word(rng, 1, 4): rng.randint(lo, hi) for _ in range(rng.randint(low, high))}


def _invert(rng: random.Random) -> list[Draft]:
    variants = [
        (
            "first",
            "when several keys share a value, the key that comes first in `d` wins",
            "    out = {}\n    for k, v in d.items():\n        out.setdefault(v, k)\n    return out\n",
        ),
        (
            "last",
            "when several keys share a value, the key that comes last in `d` wins",
            "    out = {}\n    for k, v in d.items():\n        out[v] = k\n    return out\n",
        ),
        (
            "all",
            "each value maps to the list of all keys that had it, in the order they appear in `d`",
            "    out = {}\n    for k, v in d.items():\n        out.setdefault(v, []).append(k)\n    return out\n",
        ),
        (
            "count",
            "each value maps to how many keys had it",
            "    out = {}\n    for v in d.values():\n        out[v] = out.get(v, 0) + 1\n    return out\n",
        ),
    ]
    drafts = []
    for slug, rule, body in variants:
        cases: list[list[Any]] = [[{w: rng.choice(["x", "y", "z", "w"]) for w in words(rng, 3, 6)}] for _ in range(4)]
        cases += [[{}], [{"a": "same", "b": "same", "c": "other"}]]
        drafts.append(
            Draft(
                slug=f"invert-{slug}",
                fn="invert",
                signature="invert(d)",
                does="returns a dict that maps each value of the dict `d` back to a key",
                notes=(f"The values of `d` are strings; {rule}.",),
                reference=f"def invert(d):\n{body}",
                cases=cases,
                example=True,
            )
        )
    return drafts


def _above(rng: random.Random) -> list[Draft]:
    combos = rng.sample([(t, strict) for t in range(1, 8) for strict in (True, False)], 8)
    drafts = []
    for threshold, strict in combos:
        op, said = (">", "greater than") if strict else (">=", "at least")
        cases = [[_mapping(rng, 3, 8)] for _ in range(5)] + [[{}], [{"only": threshold}]]
        drafts.append(
            Draft(
                slug=f"above-{'gt' if strict else 'ge'}-{threshold}",
                fn="keys_above",
                signature="keys_above(d)",
                does=f"returns the keys of the dict `d` whose value is {said} {threshold}, sorted alphabetically",
                reference=f"def keys_above(d):\n    return sorted(k for k, v in d.items() if v {op} {threshold})\n",
                cases=cases,
            )
        )
    return drafts


def _group_by(rng: random.Random) -> list[Draft]:
    keys = [
        ("first", "its first letter", "w[0]"),
        ("last", "its last letter", "w[-1]"),
        ("prefix", "its first two letters", "w[:2]"),
    ]
    drafts = []
    for slug, said, index in keys:
        for lower in (True, False):
            key = f"{index}.lower()" if lower else index
            case_note = (
                "Compare letters without regard to case and use the lowercase letter as the key; the words themselves are kept as written."
                if lower
                else "Letters are case-sensitive: 'A' and 'a' are different keys."
            )
            cases = [[[mixed_case(rng, w) for w in words(rng, 4, 9)]] for _ in range(4)] + [
                [[]],
                [["Apple", "avocado", "Banana"]],
            ]
            drafts.append(
                Draft(
                    slug=f"group-{slug}-{'nocase' if lower else 'case'}",
                    fn="group_words",
                    signature="group_words(words)",
                    does=f"groups the non-empty strings in the list `words` by {said}, returning a dict from that letter to the list of words",
                    notes=(
                        case_note,
                        "Within each list, words keep the order they have in `words`; an empty list returns {}.",
                    ),
                    reference=(
                        "def group_words(words):\n"
                        "    out = {}\n"
                        "    for w in words:\n"
                        f"        out.setdefault({key}, []).append(w)\n"
                        "    return out\n"
                    ),
                    cases=cases,
                    example=True,
                    also=("strings",),
                )
            )
    return drafts


def _merge(rng: random.Random) -> list[Draft]:
    ops = [
        ("sum", "their sum", "a[k] + v"),
        ("max", "the larger value", "max(a[k], v)"),
        ("min", "the smaller value", "min(a[k], v)"),
        ("product", "their product", "a[k] * v"),
        ("difference", "the value from `a` minus the value from `b`", "a[k] - v"),
        ("keep-a", "the value from `a`", "a[k]"),
    ]
    drafts = []
    for slug, said, combine in ops:
        cases = [[_mapping(rng, 2, 5, -5, 9), _mapping(rng, 2, 5, -5, 9)] for _ in range(4)]
        cases += [[{}, {"a": 1}], [{"a": 2, "b": 3}, {"a": 4, "c": -1}], [{}, {}]]
        drafts.append(
            Draft(
                slug=f"merge-{slug}",
                fn="merge",
                signature="merge(a, b)",
                does="returns a new dict holding every key of the dicts `a` and `b`",
                notes=(
                    f"A key in only one of them keeps its value; a key in both maps to {said} of its two values.",
                    "Neither `a` nor `b` is modified.",
                ),
                reference=(
                    "def merge(a, b):\n"
                    "    out = dict(a)\n"
                    "    for k, v in b.items():\n"
                    f"        out[k] = {combine.replace('a[k]', 'out[k]')} if k in out else v\n"
                    "    return out\n"
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _top_k(rng: random.Random) -> list[Draft]:
    drafts = []
    for k in (2, 3, 4):
        for alphabetical in (True, False):
            key = "(-counts[w], w)" if alphabetical else "(-counts[w], first[w])"
            tie = "alphabetically" if alphabetical else "by which word appears first in the text"
            pool = words(rng, 4, 6)

            def text(pool: list[str] = pool) -> str:
                return " ".join(mixed_case(rng, rng.choice(pool)) for _ in range(rng.randint(6, 14)))

            cases = [[text()] for _ in range(4)] + [[""], ["solo"], [sentence(rng, 3, 3)]]
            drafts.append(
                Draft(
                    slug=f"top-{k}-{'alpha' if alphabetical else 'first-seen'}",
                    fn="top_words",
                    signature="top_words(text)",
                    does=f"returns the {k} most frequent words in `text` as a list of [word, count] pairs",
                    notes=(
                        "Words are separated by whitespace and compared in lowercase; return them in lowercase.",
                        f"Order by count, highest first, and break ties {tie}.",
                        f"If there are fewer than {k} distinct words, return all of them; an empty text returns [].",
                    ),
                    reference=(
                        "def top_words(text):\n"
                        "    counts = {}\n"
                        "    first = {}\n"
                        "    for i, w in enumerate(text.lower().split()):\n"
                        "        counts[w] = counts.get(w, 0) + 1\n"
                        "        first.setdefault(w, i)\n"
                        f"    order = sorted(counts, key=lambda w: {key})\n"
                        f"    return [[w, counts[w]] for w in order[:{k}]]\n"
                    ),
                    cases=cases,
                    example=True,
                    also=("strings",),
                )
            )
    return drafts


def _nested(rng: random.Random, depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _ in range(rng.randint(1, 3)):
        roll = rng.random()
        if depth > 0 and roll < 0.4:
            out[word(rng, 1, 3)] = _nested(rng, depth - 1)
        elif roll < 0.5:
            out[word(rng, 1, 3)] = {}
        else:
            out[word(rng, 1, 3)] = rng.randint(0, 9)
    return out


def _flatten(rng: random.Random) -> list[Draft]:
    drafts = []
    for sep in (".", "/", "_", ":"):
        for keep_empty in (False, True):
            empty = "                out[key] = None\n" if keep_empty else "                pass\n"
            rule = (
                "An empty nested dict becomes a key whose value is None."
                if keep_empty
                else "An empty nested dict contributes no keys."
            )
            cases = [[_nested(rng, 2)] for _ in range(4)] + [[{}], [{"a": {"b": {"c": 1}}, "d": 2}], [{"x": {}}]]
            drafts.append(
                Draft(
                    slug=f"flatten-{SEP_NAMES[sep]}-{'keep' if keep_empty else 'drop'}",
                    fn="flatten",
                    signature="flatten(d)",
                    does=f"flattens the nested dict `d` into a single-level dict whose keys are the paths of the leaves joined with {sep!r}",
                    notes=("A value that is not a dict is a leaf and keeps its value.", rule),
                    reference=(
                        "def flatten(d):\n"
                        "    out = {}\n"
                        "    def walk(prefix, node):\n"
                        "        for k, v in node.items():\n"
                        f"            key = prefix + {sep!r} + k if prefix else k\n"
                        "            if isinstance(v, dict) and v:\n"
                        "                walk(key, v)\n"
                        "            elif isinstance(v, dict):\n"
                        f"{empty}"
                        "            else:\n"
                        "                out[key] = v\n"
                        "    walk('', d)\n"
                        "    return out\n"
                    ),
                    cases=cases,
                    example=True,
                )
            )
    return drafts


TEMPLATES = [
    Template("invert", "small", _invert),
    Template("keys-above", "small", _above),
    Template("group-words", "medium", _group_by),
    Template("merge", "medium", _merge),
    Template("top-words", "large", _top_k),
    Template("flatten", "large", _flatten),
]

IMPOSSIBLE = [
    Impossible(
        slug="beats-every-key",
        fn="beats_every_key",
        signature="beats_every_key(d)",
        does="returns a key of the dict `d` whose value is strictly greater than the value of every key in `d`, itself included",
        cases=[[{"a": 1, "b": 2}], [{"x": 5}], [{"p": 0, "q": 0}]],
        attempt="def beats_every_key(d):\n    return max(d, key=d.get)\n",
    ),
    Impossible(
        slug="fewer-keys",
        fn="fewer_keys",
        signature="fewer_keys(d)",
        does="returns a dict with fewer keys than `d` that still contains every key of `d`",
        cases=[[{"a": 1, "b": 2}], [{"x": 5}], [{"p": 0, "q": 0, "r": 1}]],
        attempt="def fewer_keys(d):\n    return dict(list(d.items())[:-1])\n",
    ),
]
