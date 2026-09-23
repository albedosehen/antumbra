"""Parsing small text formats."""

from __future__ import annotations

import random

from spec import Draft, Impossible, Template, word

SKILL = "parsing"

NAMES = {
    ";": "semicolon",
    ",": "comma",
    "&": "amp",
    "|": "pipe",
    "=": "eq",
    ":": "colon",
    ".": "dot",
    "h": "h",
    "-": "dash",
}


def _pairs(rng: random.Random) -> list[Draft]:
    drafts = []
    for sep in (";", ",", "&", "|"):
        for kv in ("=", ":"):

            def text(n: int, sep: str = sep, kv: str = kv) -> str:
                return sep.join(f"{word(rng, 1, 4)}{kv}{word(rng, 0, 5)}" for _ in range(n))

            cases = [[text(rng.randint(1, 5))] for _ in range(4)] + [[""], [f"a{kv}1{sep}a{kv}2"], [f"k{kv}"]]
            drafts.append(
                Draft(
                    slug=f"pairs-{NAMES[sep]}-{NAMES[kv]}",
                    fn="parse_pairs",
                    signature="parse_pairs(s)",
                    does=f"parses `s`, written as key{kv}value pairs separated by {sep!r}, into a dict of strings",
                    notes=(
                        f"Every pair contains exactly one {kv!r}; a value may be empty.",
                        "An empty string returns {}; if a key appears twice, the later value wins.",
                    ),
                    reference=(
                        "def parse_pairs(s):\n"
                        "    out = {}\n"
                        f"    for part in s.split({sep!r}):\n"
                        "        if part:\n"
                        f"            k, v = part.split({kv!r}, 1)\n"
                        "            out[k] = v\n"
                        "    return out\n"
                    ),
                    cases=cases,
                    also=("mappings",),
                )
            )
    return drafts


def _clock(rng: random.Random) -> list[Draft]:
    drafts = []
    for sep in (":", ".", "h", "-"):
        for seconds in (False, True):
            fields = f"HH{sep}MM{sep}SS" if seconds else f"HH{sep}MM"
            unit = "seconds" if seconds else "minutes"
            times = []
            for _ in range(6):
                parts = [rng.randint(0, 23), rng.randint(0, 59)] + ([rng.randint(0, 59)] if seconds else [])
                times.append([sep.join(f"{p:02d}" for p in parts)])
            times.append([sep.join(["00"] * (3 if seconds else 2))])
            body = "h * 3600 + m * 60 + s" if seconds else "h * 60 + m"
            unpack = "h, m, s" if seconds else "h, m"
            drafts.append(
                Draft(
                    slug=f"clock-{NAMES[sep]}-{unit}",
                    fn="since_midnight",
                    signature="since_midnight(t)",
                    does=f"returns how many {unit} after midnight the time `t` is, where `t` is written as {fields} on a 24-hour clock",
                    notes=("Every field is exactly two digits.",),
                    reference=(
                        "def since_midnight(t):\n"
                        f"    {unpack} = (int(x) for x in t.split({sep!r}))\n"
                        f"    return {body}\n"
                    ),
                    cases=times,
                )
            )
    return drafts


def _duration(rng: random.Random) -> list[Draft]:
    variants = [
        ("hms-seconds", "hms", {"h": 3600, "m": 60, "s": 1}, 1, "seconds"),
        ("dhm-minutes", "dhm", {"d": 1440, "h": 60, "m": 1}, 1, "minutes"),
        ("hms-minutes", "hms", {"h": 3600, "m": 60, "s": 1}, 60, "whole minutes, rounding down"),
        ("wd-hours", "wd", {"w": 168, "d": 24}, 1, "hours"),
        ("hm-minutes", "hm", {"h": 60, "m": 1}, 1, "minutes"),
        ("dh-hours", "dh", {"d": 24, "h": 1}, 1, "hours"),
        ("wdh-days", "wdh", {"w": 168, "d": 24, "h": 1}, 24, "whole days, rounding down"),
        ("dhms-hours", "dhms", {"d": 86400, "h": 3600, "m": 60, "s": 1}, 3600, "whole hours, rounding down"),
    ]
    drafts = []
    for slug, units, weights, divisor, result in variants:

        def written(units: str = units) -> str:
            chosen = [u for u in units if rng.random() < 0.7] or [units[-1]]
            return "".join(f"{rng.randint(1, 40)}{u}" for u in chosen)

        cases = [[written()] for _ in range(5)] + [[""], [f"1{units[0]}"], [f"7{units[-1]}"]]
        listed = ", ".join(f"'{u}'" for u in units)
        drafts.append(
            Draft(
                slug=f"duration-{slug}",
                fn="duration",
                signature="duration(s)",
                does=f"returns the length of the duration `s` in {result}",
                notes=(
                    f"`s` is a sequence of whole numbers each followed by one of the units {listed}, in that order, for example '{written()}'.",
                    "Each unit appears at most once and any of them may be missing; an empty string is a duration of 0.",
                ),
                reference=(
                    "import re\n"
                    "def duration(s):\n"
                    f"    weights = {weights!r}\n"
                    "    total = 0\n"
                    f"    for amount, unit in re.findall(r'(\\d+)([{units}])', s):\n"
                    "        total += int(amount) * weights[unit]\n"
                    + (f"    return total // {divisor}\n" if divisor != 1 else "    return total\n")
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _semver(rng: random.Random) -> list[Draft]:
    drafts = []
    for sep in (".", "-"):
        for words_out in (False, True):
            for pad in (True, False):
                lt, eq, gt = ("'lt'", "'eq'", "'gt'") if words_out else ("-1", "0", "1")
                answer = "'lt', 'eq' or 'gt'" if words_out else "-1, 0 or 1"
                joiner = "dot" if sep == "." else "dash"

                def version(pad: bool = pad, sep: str = sep) -> str:
                    parts = [rng.randint(0, 12) for _ in range(3 if not pad else rng.randint(1, 3))]
                    return sep.join(map(str, parts))

                def v(text: str, sep: str = sep) -> str:
                    return text.replace(".", sep)

                cases = [[version(), version()] for _ in range(5)] + [
                    [v("1.10.0"), v("1.9.0")],
                    [v("2.0.0"), v("2.0.0")],
                    [v("0.1.2"), v("0.1.10")],
                ]
                if pad:
                    cases.append([v("1.2"), v("1.2.0")])
                missing = (
                    f"A version may have fewer than three parts; a missing part counts as 0, so {v('1.2')!r} equals {v('1.2.0')!r}."
                    if pad
                    else "Every version has exactly three parts."
                )
                drafts.append(
                    Draft(
                        slug=f"semver-{joiner}-{'words' if words_out else 'sign'}-{'pad' if pad else 'exact'}",
                        fn="compare_versions",
                        signature="compare_versions(a, b)",
                        does=f"compares the version strings `a` and `b`, written as whole numbers separated by {sep!r}, returning {answer} when `a` is lower than, equal to, or higher than `b`",
                        notes=(
                            f"Parts are compared as numbers from left to right, so {v('1.10')!r} is higher than {v('1.9')!r}.",
                            missing,
                        ),
                        reference=(
                            "def compare_versions(a, b):\n"
                            f"    x = [int(p) for p in a.split({sep!r})]\n"
                            f"    y = [int(p) for p in b.split({sep!r})]\n"
                            "    n = max(len(x), len(y))\n"
                            "    x += [0] * (n - len(x))\n"
                            "    y += [0] * (n - len(y))\n"
                            f"    return {lt} if x < y else {gt} if x > y else {eq}\n"
                        ),
                        cases=cases,
                        example=True,
                    )
                )
    return drafts


def _roman(rng: random.Random) -> list[Draft]:
    samples = ["XIV", "MCMXCIV", "iv", "MMXXVI", "LVIII", "", "IIII", "XIZ", "cdxliv", "IX"]
    drafts = []
    for lenient in (False, True):
        for any_case in (True, False):
            for empty_zero in (True, False):
                rng.shuffle(samples)
                refuse = "returns None" if lenient else "raises ValueError"
                refusal = "        return None\n" if lenient else "        raise ValueError(s)\n"
                case_note = (
                    "Lowercase letters are accepted and read like uppercase ones."
                    if any_case
                    else f"Only uppercase letters are accepted; a lowercase letter {refuse}."
                )
                empty_note = "An empty string returns 0." if empty_zero else f"An empty string {refuse}."
                drafts.append(
                    Draft(
                        slug=f"roman-{'anycase' if any_case else 'upper'}-{'zero' if empty_zero else 'strict'}-{'none' if lenient else 'raise'}",
                        fn="roman",
                        signature="roman(s)",
                        does="returns the integer value of the Roman numeral `s`",
                        notes=(
                            "Use the subtractive rule: a symbol whose value is smaller than the symbol after it is subtracted, otherwise it is added. Do not check that the numeral is in canonical form, so 'IIII' is 4.",
                            f"{case_note} Any character other than I, V, X, L, C, D and M {refuse}.",
                            empty_note,
                        ),
                        reference=(
                            "def roman(s):\n"
                            "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
                            + ("    s = s.upper()\n" if any_case else "")
                            + "    if not s:\n"
                            + ("        return 0\n" if empty_zero else refusal)
                            + "    if any(c not in values for c in s):\n"
                            + refusal
                            + "    total = 0\n"
                            "    for i, c in enumerate(s):\n"
                            "        v = values[c]\n"
                            "        if i + 1 < len(s) and values[s[i + 1]] > v:\n"
                            "            total -= v\n"
                            "        else:\n"
                            "            total += v\n"
                            "    return total\n"
                        ),
                        cases=[[s] for s in samples],
                        example=True,
                        raises=() if lenient else ("ValueError",),
                    )
                )
    return drafts


def _query(rng: random.Random) -> list[Draft]:
    drafts = []
    for sep in ("&", ";"):
        for plus in (False, True):
            for lower in (False, True):

                def text(sep: str = sep) -> str:
                    parts = []
                    for _ in range(rng.randint(1, 6)):
                        key = rng.choice(["a", "B", "tag", "Q", "q"])
                        roll = rng.random()
                        parts.append(
                            key
                            if roll < 0.15
                            else f"{key}={word(rng, 0, 4)}"
                            if roll < 0.8
                            else f"{key}={word(rng, 1, 3)}+{word(rng, 1, 3)}"
                        )
                    return sep.join(parts)

                decode = "Every '+' in a key or value stands for a space." if plus else "'+' has no special meaning."
                keys = (
                    "Keys are case-insensitive and are returned in lowercase."
                    if lower
                    else "Keys are case-sensitive and returned as written."
                )
                fix = ".replace('+', ' ')" if plus else ""
                key_fix = f"{fix}.lower()" if lower else fix
                cases = [[text()] for _ in range(5)] + [[""], [f"a=1{sep}{sep}A=2{sep}b"], ["k=v=w"]]
                drafts.append(
                    Draft(
                        slug=f"query-{NAMES[sep]}-{'plus' if plus else 'raw'}-{'lower' if lower else 'exact'}",
                        fn="parse_query",
                        signature="parse_query(s)",
                        does="parses the query string `s` into a dict mapping each key to the list of its values, in the order they appear",
                        notes=(
                            f"Pairs are separated by {sep!r}, and empty pairs are ignored.",
                            "Only the first '=' in a pair separates the key from the value; a pair without '=' has the value ''.",
                            f"{decode} {keys}",
                        ),
                        reference=(
                            "def parse_query(s):\n"
                            "    out = {}\n"
                            f"    for part in s.split({sep!r}):\n"
                            "        if not part:\n"
                            "            continue\n"
                            "        key, _, value = part.partition('=')\n"
                            f"        out.setdefault(key{key_fix}, []).append(value{fix})\n"
                            "    return out\n"
                        ),
                        cases=cases,
                        example=True,
                        also=("mappings",),
                    )
                )
    return drafts


TEMPLATES = [
    Template("pairs", "small", _pairs),
    Template("clock", "small", _clock),
    Template("duration", "medium", _duration),
    Template("semver", "medium", _semver),
    Template("roman", "large", _roman),
    Template("query", "large", _query),
]

IMPOSSIBLE = [
    Impossible(
        slug="thirteenth-month",
        fn="thirteenth_month",
        signature="thirteenth_month(s)",
        does="parses the date `s`, written as YYYY-MM-DD, and returns the name of the thirteenth month of that year in the Gregorian calendar",
        cases=[["2024-01-15"], ["1999-12-31"], ["2030-06-01"]],
        attempt="def thirteenth_month(s):\n    return 'December'\n",
    ),
    Impossible(
        slug="both-signs",
        fn="both_signs",
        signature="both_signs(s)",
        does="parses the integer written in `s` and returns a number that is both strictly positive and strictly negative with the same magnitude",
        cases=[["5"], ["-3"], ["12"]],
        attempt="def both_signs(s):\n    return abs(int(s))\n",
    ),
]
