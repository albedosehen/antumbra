"""Number sequences."""

from __future__ import annotations

import random

from spec import Draft, Impossible, Template, ints

SKILL = "sequences"


def _nth_term(rng: random.Random) -> list[Draft]:
    combos = rng.sample([(a, d) for a in range(-5, 11) for d in range(-4, 8) if d != 0], 8)
    drafts = []
    for a, d in combos:
        change = f"increases by {d}" if d > 0 else f"decreases by {-d}"
        cases = [[1], [2], [rng.randint(3, 20)], [rng.randint(21, 400)], [rng.randint(3, 9)]]
        drafts.append(
            Draft(
                slug=f"nth-term-{a}-{d}".replace("--", "-minus"),
                fn="nth_term",
                signature="nth_term(n)",
                does=f"returns term number `n`, counting from 1, of the sequence that starts at {a} and {change} at every step",
                reference=f"def nth_term(n):\n    return {a} + (n - 1) * {d}\n",
                cases=cases,
            )
        )
    return drafts


def _running(rng: random.Random) -> list[Draft]:
    # (slug, what element i holds, its value at i == 0, its value after that)
    variants = [
        ("total", "the running total: element i is the sum of xs[0] through xs[i]", "x", "acc + x"),
        ("max", "the running maximum: element i is the largest of xs[0] through xs[i]", "x", "max(acc, x)"),
        ("min", "the running minimum: element i is the smallest of xs[0] through xs[i]", "x", "min(acc, x)"),
        ("product", "the running product: element i is the product of xs[0] through xs[i]", "x", "acc * x"),
        (
            "evens",
            "a running count: element i is how many of xs[0] through xs[i] are even",
            "int(x % 2 == 0)",
            "acc + int(x % 2 == 0)",
        ),
        (
            "abs-total",
            "a running total of sizes: element i is the sum of the absolute values of xs[0] through xs[i]",
            "abs(x)",
            "acc + abs(x)",
        ),
        ("from-first", "how far each element is from the first: element i is xs[i] - xs[0]", "0", "x - xs[0]"),
        (
            "alternating",
            "the running alternating sum: element i is xs[0] - xs[1] + xs[2] - ... with the sign of xs[i] set by whether i is even (+) or odd (-)",
            "x",
            "acc - x if i % 2 else acc + x",
        ),
    ]
    drafts = []
    for slug, said, first, step in variants:
        cases = [[ints(rng, 2, 8, -5, 5)] for _ in range(5)] + [[[]], [[7]]]
        drafts.append(
            Draft(
                slug=f"running-{slug}",
                fn="running",
                signature="running(xs)",
                does=f"returns a list of the same length as `xs` holding {said}",
                notes=("An empty list returns [].",),
                reference=(
                    "def running(xs):\n"
                    "    out = []\n"
                    "    for i, x in enumerate(xs):\n"
                    f"        acc = {first} if i == 0 else {step}\n"
                    "        out.append(acc)\n"
                    "    return out\n"
                ),
                cases=cases,
                also=("lists",),
            )
        )
    return drafts


def _fib_like(rng: random.Random) -> list[Draft]:
    combos = rng.sample([(a, b) for a in range(-3, 6) for b in range(-3, 6) if (a, b) != (0, 0)], 8)
    drafts = []
    for a, b in combos:
        cases = [[0], [1], [2], [rng.randint(3, 8)], [rng.randint(9, 20)]]
        drafts.append(
            Draft(
                slug=f"fib-like-{a}-{b}".replace("--", "-minus"),
                fn="fib_like",
                signature="fib_like(n)",
                does=f"returns the first `n` terms of the sequence whose first two terms are {a} and {b} and in which every later term is the sum of the two terms before it",
                notes=("`n` may be 0, which returns [], or 1.",),
                reference=(
                    "def fib_like(n):\n"
                    f"    out = [{a}, {b}][:n]\n"
                    "    while len(out) < n:\n"
                    "        out.append(out[-1] + out[-2])\n"
                    "    return out\n"
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _look_and_say(rng: random.Random) -> list[Draft]:
    drafts = []
    for count_first, length in ((True, False), (False, False), (True, True), (False, True)):
        piece = "str(j - i) + s[i]" if count_first else "s[i] + str(j - i)"
        said = (
            "the length of the run followed by its digit"
            if count_first
            else "the digit followed by the length of its run"
        )
        cases = [["1", 0], ["1", 1], ["1", 4], ["211", 2], ["".join(rng.choice("123") for _ in range(5)), 3], ["9", 3]]
        drafts.append(
            Draft(
                slug=f"look-and-say-{'count-first' if count_first else 'digit-first'}{'-length' if length else ''}",
                fn="look_and_say",
                signature="look_and_say(s, steps)",
                does="applies `steps` rounds of a look-and-say description to the digit string `s` and returns "
                + ("how many characters the result has" if length else "the result"),
                notes=(
                    f"One round replaces each run of a repeated digit with {said}, left to right.",
                    "`steps` may be 0, which returns `s` unchanged.",
                ),
                reference=(
                    "def look_and_say(s, steps):\n"
                    "    for _ in range(steps):\n"
                    "        out = []\n"
                    "        i = 0\n"
                    "        while i < len(s):\n"
                    "            j = i\n"
                    "            while j < len(s) and s[j] == s[i]:\n"
                    "                j += 1\n"
                    f"            out.append({piece})\n"
                    "            i = j\n"
                    "        s = ''.join(out)\n" + ("    return len(s)\n" if length else "    return s\n")
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _classify(rng: random.Random) -> list[Draft]:
    labels = [
        ("words", ("arithmetic", "geometric", "neither")),
        ("letters", ("A", "G", "N")),
        ("ops", ("add", "multiply", "none")),
        ("symbols", ("+", "*", "?")),
        ("codes", ("AR", "GE", "NO")),
    ]
    drafts = []
    for slug, (arith, geo, neither) in labels:

        def seq() -> list[int]:
            start, step, n = rng.randint(-5, 9), rng.choice([-3, -2, 2, 3]), rng.randint(3, 6)
            roll = rng.random()
            if roll < 0.35:
                return [start + i * step for i in range(n)]
            if roll < 0.7:
                return [(start or 1) * step**i for i in range(n)]
            return ints(rng, 3, 6)

        cases = [[seq()] for _ in range(6)] + [[[4, 4, 4]], [[3]], [[1, 2]], [[0, 0, 5]], [[2, -4, 8]]]
        drafts.append(
            Draft(
                slug=f"classify-{slug}",
                fn="classify",
                signature="classify(xs)",
                does=f"returns {arith!r} when the list of integers `xs` is an arithmetic sequence, {geo!r} when it is a geometric one, and {neither!r} otherwise",
                notes=(
                    "Arithmetic means the difference between consecutive terms is constant; geometric means every term after the first is the previous term times the same non-zero integer.",
                    f"A sequence that is both, such as a constant one, is {arith!r}; one with fewer than two terms is {neither!r}.",
                ),
                reference=(
                    "def classify(xs):\n"
                    "    if len(xs) < 2:\n"
                    f"        return {neither!r}\n"
                    "    if len({xs[i] - xs[i - 1] for i in range(1, len(xs))}) == 1:\n"
                    f"        return {arith!r}\n"
                    "    if xs[0] != 0 and xs[1] % xs[0] == 0:\n"
                    "        r = xs[1] // xs[0]\n"
                    "        if r != 0 and all(xs[i] == xs[i - 1] * r for i in range(1, len(xs))):\n"
                    f"            return {geo!r}\n"
                    f"    return {neither!r}\n"
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _collatz(rng: random.Random) -> list[Draft]:
    variants = [
        ("steps", "how many steps it takes to reach 1 (0 for n == 1)", "steps"),
        ("length", "how many numbers the sequence has, counting n itself and the final 1", "steps + 1"),
        ("peak", "the largest number the sequence reaches, n included", "peak"),
        ("odd", "how many odd numbers the sequence visits, counting n and the final 1", "odd"),
        ("halvings", "how many of its steps are halvings", "halvings"),
    ]
    drafts = []
    for slug, said, result in variants:
        cases = [[1], [6], [7], [27], [rng.randint(2, 200)], [rng.randint(200, 5000)], [0]]
        drafts.append(
            Draft(
                slug=f"collatz-{slug}",
                fn="collatz",
                signature="collatz(n)",
                does=f"follows the sequence that starts at `n` and repeatedly halves an even number and replaces an odd number m by 3*m + 1, stopping at 1, and returns {said}",
                notes=("`n` below 1 raises ValueError.",),
                reference=(
                    "def collatz(n):\n"
                    "    if n < 1:\n"
                    "        raise ValueError(n)\n"
                    "    steps, peak, odd, halvings = 0, n, n % 2, 0\n"
                    "    while n != 1:\n"
                    "        if n % 2 == 0:\n"
                    "            n //= 2\n"
                    "            halvings += 1\n"
                    "        else:\n"
                    "            n = 3 * n + 1\n"
                    "        odd += n % 2\n"
                    "        peak = max(peak, n)\n"
                    "        steps += 1\n"
                    f"    return {result}\n"
                ),
                cases=cases,
                example=True,
                raises=("ValueError",),
            )
        )
    return drafts


TEMPLATES = [
    Template("nth-term", "small", _nth_term),
    Template("running", "small", _running),
    Template("fib-like", "medium", _fib_like),
    Template("look-and-say", "medium", _look_and_say),
    Template("classify", "large", _classify),
    Template("collatz", "large", _collatz),
]

IMPOSSIBLE = [
    Impossible(
        slug="last-natural",
        fn="last_natural",
        signature="last_natural(n)",
        does="returns the last term of the infinite sequence n, n + 1, n + 2, ...",
        cases=[[1], [10], [-4]],
        attempt="def last_natural(n):\n    return float('inf')\n",
    ),
    Impossible(
        slug="descending-increasing",
        fn="descending_increasing",
        signature="descending_increasing(n)",
        does="returns a list of `n` integers, n >= 2, that is strictly increasing and also strictly decreasing",
        cases=[[2], [3], [5]],
        attempt="def descending_increasing(n):\n    return list(range(n))\n",
    ),
]
