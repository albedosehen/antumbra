"""List manipulation."""

from __future__ import annotations

import random

from spec import Draft, Impossible, Template, ints

SKILL = "lists"

ORDINAL = {2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


def _rotate(rng: random.Random) -> list[Draft]:
    combos = rng.sample([(d, k) for d in ("left", "right") for k in range(1, 7)], 8)
    drafts = []
    for direction, k in combos:
        body = "xs[k:] + xs[:k]" if direction == "left" else "xs[len(xs) - k:] + xs[:len(xs) - k]"
        cases = [[ints(rng, 4, 8)] for _ in range(3)] + [[[1, 2]], [[7]], [[]], [ints(rng, 9, 12)]]
        drafts.append(
            Draft(
                slug=f"rotate-{direction}-{k}",
                fn="rotate",
                signature="rotate(xs)",
                does=f"returns a new list: `xs` rotated {direction} by {k} positions",
                notes=("Rotating by more than the length wraps around; an empty list returns [].",),
                reference=f"def rotate(xs):\n    if not xs:\n        return []\n    k = {k} % len(xs)\n    return {body}\n",
                cases=cases,
            )
        )
    return drafts


def _every_nth(rng: random.Random) -> list[Draft]:
    combos = rng.sample([(n, s) for n in range(2, 6) for s in range(n)], 8)
    drafts = []
    for n, start in combos:
        cases = [[ints(rng, 0, 14)] for _ in range(5)] + [[list(range(10))], [[]]]
        drafts.append(
            Draft(
                slug=f"every-{n}-from-{start}",
                fn="every_nth",
                signature="every_nth(xs)",
                does=f"returns every {ORDINAL[n]} element of `xs`, starting with the element at index {start}",
                notes=("Return a list; it is empty when `xs` has no element at that index.",),
                reference=f"def every_nth(xs):\n    return xs[{start}::{n}]\n",
                cases=cases,
            )
        )
    return drafts


def _chunks(rng: random.Random) -> list[Draft]:
    combos = [(n, keep) for n in range(2, 6) for keep in (True, False)]
    drafts = []
    for n, keep in combos:
        stop = "len(xs)" if keep else f"len(xs) - {n} + 1"
        tail = (
            f"The last chunk may be shorter when the length is not a multiple of {n}."
            if keep
            else f"A final chunk with fewer than {n} elements is left out."
        )
        cases = [[ints(rng, 0, 13)] for _ in range(4)] + [[list(range(n * 2))], [list(range(n * 2 + 1))], [[]]]
        drafts.append(
            Draft(
                slug=f"chunks-{n}-{'keep' if keep else 'drop'}",
                fn="chunks",
                signature="chunks(xs)",
                does=f"splits `xs` into consecutive chunks of {n} elements, returned as a list of lists",
                notes=(tail, "An empty list returns []."),
                reference=f"def chunks(xs):\n    return [xs[i:i + {n}] for i in range(0, {stop}, {n})]\n",
                cases=cases,
                example=True,
            )
        )
    return drafts


def _dedupe(rng: random.Random) -> list[Draft]:
    variants = [
        (
            "first",
            "removes repeated values from `xs`, keeping the first occurrence of each value in its original position",
            "    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
        ),
        (
            "last",
            "removes repeated values from `xs`, keeping only the last occurrence of each value, in the order those last occurrences appear",
            "    seen = set()\n    out = []\n    for x in reversed(xs):\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out[::-1]\n",
        ),
        (
            "first-abs",
            "removes values from `xs` whose absolute value has already appeared, keeping the first occurrence in its original position",
            "    seen = set()\n    out = []\n    for x in xs:\n        if abs(x) not in seen:\n            seen.add(abs(x))\n            out.append(x)\n    return out\n",
        ),
        (
            "once",
            "returns the values of `xs` that occur exactly once in it, in their original order",
            "    return [x for x in xs if xs.count(x) == 1]\n",
        ),
    ]
    drafts = []
    for slug, does, body in variants:
        cases = [[ints(rng, 3, 10, -4, 4)] for _ in range(5)] + [[[]], [[3, -3, 3]]]
        drafts.append(
            Draft(
                slug=f"dedupe-{slug}",
                fn="dedupe",
                signature="dedupe(xs)",
                does=does,
                notes=("Return a new list; an empty list returns [].",),
                reference=f"def dedupe(xs):\n{body}",
                cases=cases,
                example=True,
            )
        )
    return drafts


def _longest_run(rng: random.Random) -> list[Draft]:
    relations = [(">", "strictly greater than"), (">=", "greater than or equal to"), ("<", "strictly less than")]
    drafts = []
    for op, said in relations:
        for latest in (False, True):
            tie = ">=" if latest else ">"
            reference = (
                "def longest_run(xs):\n"
                "    best = []\n"
                "    start = 0\n"
                "    for i in range(1, len(xs) + 1):\n"
                f"        if i == len(xs) or not (xs[i] {op} xs[i - 1]):\n"
                "            run = xs[start:i]\n"
                f"            if len(run) {tie} len(best):\n"
                "                best = run\n"
                "            start = i\n"
                "    return best\n"
            )
            cases = [[ints(rng, 5, 12, 0, 6)] for _ in range(4)] + [[[1, 2, 1, 2]], [[5]], [[]], [[3, 3, 3]]]
            drafts.append(
                Draft(
                    slug=f"run-{'inc' if op == '>' else 'nondec' if op == '>=' else 'dec'}-{'last' if latest else 'first'}",
                    fn="longest_run",
                    signature="longest_run(xs)",
                    does=f"returns the longest contiguous stretch of `xs` in which each element is {said} the one before it",
                    notes=(
                        f"If several stretches are equally long, return the {'last' if latest else 'first'} of them.",
                        "A single element is a stretch of length 1; an empty list returns [].",
                    ),
                    reference=reference,
                    cases=cases,
                    example=True,
                )
            )
    return drafts


def _pairs_to(rng: random.Random) -> list[Draft]:
    drafts = []
    for target in rng.sample(range(-4, 16), 8):
        cases = [[ints(rng, 3, 8, -3, 10)] for _ in range(5)] + [[[target, 0, 0]], [[]], [[target]]]
        drafts.append(
            Draft(
                slug=f"pairs-to-{target}",
                fn="pairs_to",
                signature="pairs_to(xs)",
                does=f"returns every pair of positions [i, j] with i < j such that xs[i] + xs[j] == {target}",
                notes=(
                    "Return the pairs as a list of two-element lists, ordered by i and then by j.",
                    f"Return [] when no two elements add up to {target}.",
                ),
                reference=(
                    "def pairs_to(xs):\n"
                    f"    return [[i, j] for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[i] + xs[j] == {target}]\n"
                ),
                cases=cases,
                example=True,
                also=("numbers",),
            )
        )
    return drafts


TEMPLATES = [
    Template("rotate", "small", _rotate),
    Template("every-nth", "small", _every_nth),
    Template("chunks", "medium", _chunks),
    Template("dedupe", "medium", _dedupe),
    Template("longest-run", "large", _longest_run),
    Template("pairs-to", "large", _pairs_to),
]

IMPOSSIBLE = [
    Impossible(
        slug="greater-than-all",
        fn="greater_than_all",
        signature="greater_than_all(xs)",
        does="returns an element of `xs` that is strictly greater than every element of `xs`, itself included",
        cases=[[[1, 5, 3]], [[7]], [[-2, -9]]],
        attempt="def greater_than_all(xs):\n    return max(xs)\n",
    ),
    Impossible(
        slug="shorter-superset",
        fn="shorter_superset",
        signature="shorter_superset(xs)",
        does="returns a list that contains every element of `xs`, each as many times as it occurs in `xs`, and has fewer elements than `xs`",
        cases=[[[1, 2, 2]], [[4, 4]], [[1, 2, 3, 4]]],
        attempt="def shorter_superset(xs):\n    return sorted(set(xs))\n",
    ),
]
