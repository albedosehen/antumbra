"""Integer arithmetic."""

from __future__ import annotations

import random

from spec import Draft, Impossible, Template

SKILL = "numbers"

GLYPH_NAMES = {"..": "dots", "~": "tilde", ":": "colon"}
SEP_NAMES = {",": "comma", ";": "semicolon", " ": "space"}

PRIME_TEST = (
    "def _is_prime(n):\n"
    "    if n < 2:\n"
    "        return False\n"
    "    i = 2\n"
    "    while i * i <= n:\n"
    "        if n % i == 0:\n"
    "            return False\n"
    "        i += 1\n"
    "    return True\n"
)


def _count_multiples(rng: random.Random) -> list[Draft]:
    drafts = []
    for k in range(2, 10):
        cases = [[1, 10], [0, 0], [-10, 10], [5, 4], [k, k], [-20, -3]] + [
            sorted([rng.randint(-50, 50), rng.randint(-50, 50)]) for _ in range(2)
        ]
        drafts.append(
            Draft(
                slug=f"multiples-of-{k}",
                fn="count_multiples",
                signature="count_multiples(a, b)",
                does=f"returns how many integers from `a` to `b` inclusive are divisible by {k}",
                notes=("`a` and `b` may be negative or zero; if `a` is greater than `b`, return 0.",),
                reference=f"def count_multiples(a, b):\n    if a > b:\n        return 0\n    return b // {k} - (a - 1) // {k}\n",
                cases=cases,
            )
        )
    return drafts


def _next_multiple(rng: random.Random) -> list[Draft]:
    drafts = []
    for k in rng.sample(range(2, 13), 8):
        cases = [[0], [1], [k], [k + 1], [-1], [-k - 1]] + [[rng.randint(-100, 100)] for _ in range(2)]
        drafts.append(
            Draft(
                slug=f"next-multiple-{k}",
                fn="next_multiple",
                signature="next_multiple(n)",
                does=f"returns the smallest multiple of {k} that is greater than or equal to the integer `n`",
                notes=("`n` may be negative or zero.",),
                reference=f"def next_multiple(n):\n    return -(-n // {k}) * {k}\n",
                cases=cases,
            )
        )
    return drafts


def _to_base(rng: random.Random) -> list[Draft]:
    drafts = []
    for base in range(2, 10):
        cases = [[rng.randint(2, 999)], [0], [1], [base], [base * base - 1], [rng.randint(1000, 99999)]]
        drafts.append(
            Draft(
                slug=f"to-base-{base}",
                fn="to_base",
                signature="to_base(n)",
                does=f"returns the non-negative integer `n` written in base {base}, as a string of digits",
                notes=("0 is written as '0', and there are no leading zeros.",),
                reference=(
                    "def to_base(n):\n"
                    "    if n == 0:\n"
                    "        return '0'\n"
                    "    digits = []\n"
                    "    while n:\n"
                    f"        digits.append(str(n % {base}))\n"
                    f"        n //= {base}\n"
                    "    return ''.join(reversed(digits))\n"
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _primes_ending(rng: random.Random) -> list[Draft]:
    drafts = []
    for digit in (1, 3, 7, 9):
        for count in (False, True):
            body = (
                f"    return sum(1 for p in range(a, b + 1) if _is_prime(p) and p % 10 == {digit})\n"
                if count
                else f"    return [p for p in range(a, b + 1) if _is_prime(p) and p % 10 == {digit}]\n"
            )
            cases = [[1, 50], [-10, 2], [digit, digit], [90, 60]] + [
                [lo, lo + rng.randint(10, 120)] for lo in (rng.randint(0, 300) for _ in range(3))
            ]
            said = "how many primes" if count else "the primes"
            drafts.append(
                Draft(
                    slug=f"primes-ending-{digit}-{'count' if count else 'list'}",
                    fn="primes_ending",
                    signature="primes_ending(a, b)",
                    does=f"returns {said} p with a <= p <= b whose last decimal digit is {digit}"
                    + ("" if count else ", in increasing order"),
                    notes=(
                        f"`a` may be below 2 and may exceed `b`; return {'0' if count else '[]'} when there are none.",
                    ),
                    reference=f"{PRIME_TEST}\ndef primes_ending(a, b):\n{body}",
                    cases=cases,
                    example=True,
                )
            )
    return drafts


def _luhn_total(digits: list[int]) -> int:
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total


def _valid_number(rng: random.Random, modulus: int) -> list[int]:
    while True:
        body = [rng.randint(0, 9) for _ in range(rng.randint(6, 12))]
        for check in range(10):
            if _luhn_total([*body, check]) % modulus == 0:
                return [*body, check]


def _luhn(rng: random.Random) -> list[Draft]:
    drafts = []
    for modulus in (10, 7, 9, 11):
        for sep, said in ((" ", "spaces"), ("-", "dashes")):

            def written(digits: list[int], sep: str = sep) -> str:
                text = "".join(map(str, digits))
                return sep.join(text[i : i + 4] for i in range(0, len(text), 4))

            valid = [written(_valid_number(rng, modulus)) for _ in range(3)]
            broken = _valid_number(rng, modulus)
            broken[-1] = (broken[-1] + 1) % 10
            cases = [[v] for v in valid] + [
                [written(broken)],
                ["7"],
                [""],
                ["12a4"],
                [written(_valid_number(rng, modulus)).replace(sep, "")],
            ]
            drafts.append(
                Draft(
                    slug=f"luhn-{modulus}-{'space' if sep == ' ' else 'dash'}",
                    fn="luhn_ok",
                    signature="luhn_ok(s)",
                    does="returns True when the number written in `s` passes the checksum described below, and False otherwise",
                    notes=(
                        f"Ignore {said} in `s`; any other character that is not a digit 0-9 raises ValueError.",
                        "Going from the rightmost digit leftwards, double every second digit (the second, fourth, and so on); when a doubled digit is greater than 9, subtract 9 from it.",
                        f"The number passes when it has at least two digits and the sum of all the resulting digits is a multiple of {modulus}.",
                    ),
                    reference=(
                        "def luhn_ok(s):\n"
                        "    digits = []\n"
                        "    for c in s:\n"
                        f"        if c == {sep!r}:\n"
                        "            continue\n"
                        "        if c not in '0123456789':\n"
                        "            raise ValueError(c)\n"
                        "        digits.append(int(c))\n"
                        "    if len(digits) < 2:\n"
                        "        return False\n"
                        "    total = 0\n"
                        "    for i, d in enumerate(reversed(digits)):\n"
                        "        if i % 2 == 1:\n"
                        "            d *= 2\n"
                        "            if d > 9:\n"
                        "                d -= 9\n"
                        "        total += d\n"
                        f"    return total % {modulus} == 0\n"
                    ),
                    cases=cases,
                    example=True,
                    raises=("ValueError",),
                )
            )
    return drafts


def _ranges(rng: random.Random) -> list[Draft]:
    combos = rng.sample([(glyph, sep) for glyph in ("..", "~", ":") for sep in (",", ";", " ")], 8)
    drafts = []
    for glyph, sep in combos:

        def distinct() -> list[int]:
            start = rng.randint(-8, 8)
            values: list[int] = []
            for _ in range(rng.randint(3, 6)):
                values.extend(range(start, start + rng.randint(1, 5)))
                start = values[-1] + rng.randint(2, 4)
            return values

        cases = [[distinct()] for _ in range(4)] + [[[]], [[5]], [[1, 2]], [[-3, -2, -1, 0]]]
        drafts.append(
            Draft(
                slug=f"ranges-{GLYPH_NAMES[glyph]}-{SEP_NAMES[sep]}",
                fn="ranges",
                signature="ranges(xs)",
                does="compresses the sorted list of distinct integers `xs` into a string of ranges",
                notes=(
                    f"Each maximal run of three or more consecutive integers is written as first{glyph}last, for example 4{glyph}7; shorter runs are written as their individual numbers.",
                    f"The items are joined with {sep!r}; an empty list returns ''.",
                ),
                reference=(
                    "def ranges(xs):\n"
                    "    items = []\n"
                    "    i = 0\n"
                    "    while i < len(xs):\n"
                    "        j = i\n"
                    "        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:\n"
                    "            j += 1\n"
                    "        if j - i + 1 >= 3:\n"
                    f"            items.append(str(xs[i]) + {glyph!r} + str(xs[j]))\n"
                    "        else:\n"
                    "            items.extend(str(x) for x in xs[i:j + 1])\n"
                    "        i = j + 1\n"
                    f"    return {sep!r}.join(items)\n"
                ),
                cases=cases,
                example=True,
                also=("lists",),
            )
        )
    return drafts


TEMPLATES = [
    Template("count-multiples", "small", _count_multiples),
    Template("next-multiple", "small", _next_multiple),
    Template("to-base", "medium", _to_base),
    Template("primes-ending", "medium", _primes_ending),
    Template("luhn", "large", _luhn),
    Template("ranges", "large", _ranges),
]

IMPOSSIBLE = [
    Impossible(
        slug="largest-prime",
        fn="largest_prime",
        signature="largest_prime(n)",
        does="returns the largest prime number, which must also be greater than `n`",
        cases=[[1], [100], [7919]],
        attempt=f"{PRIME_TEST}\ndef largest_prime(n):\n    n += 1\n    while not _is_prime(n):\n        n += 1\n    return n\n",
    ),
    Impossible(
        slug="even-and-odd",
        fn="even_and_odd",
        signature="even_and_odd(n)",
        does="returns an integer greater than `n` that is both even and odd",
        cases=[[0], [3], [10]],
        attempt="def even_and_odd(n):\n    return n + 1\n",
    ),
]
