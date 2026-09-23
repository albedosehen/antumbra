"""Calendar arithmetic with the standard library's datetime."""

from __future__ import annotations

import datetime
import random
from typing import Any

from spec import Draft, Impossible, Template

SKILL = "dates"

DAY_NAMES = "['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']"


def day(rng: random.Random, lo: int = 1950, hi: int = 2060) -> datetime.date:
    start = datetime.date(lo, 1, 1)
    return start + datetime.timedelta(days=rng.randrange((datetime.date(hi, 12, 31) - start).days))


def _weekday(rng: random.Random) -> list[Draft]:
    outputs = [
        ("name", "the English name of its day of the week, such as 'Monday'", f"{DAY_NAMES}[d.weekday()]"),
        ("short", "the first three letters of its day of the week, such as 'Mon'", f"{DAY_NAMES}[d.weekday()][:3]"),
        ("monday-zero", "its day of the week as a number, Monday being 0 and Sunday 6", "d.weekday()"),
        ("iso", "its ISO day of the week, Monday being 1 and Sunday 7", "d.isoweekday()"),
    ]
    drafts = []
    for iso_input in (False, True):
        for slug, said, expr in outputs:
            dates = [day(rng) for _ in range(6)] + [datetime.date(2000, 2, 29)]
            cases: list[list[Any]]
            if iso_input:
                signature, parse = "weekday(s)", "datetime.date.fromisoformat(s)"
                cases = [[d.isoformat()] for d in dates]
                given = "the date `s`, written as YYYY-MM-DD"
            else:
                signature, parse = "weekday(y, m, d)", "datetime.date(y, m, d)"
                cases = [[d.year, d.month, d.day] for d in dates]
                given = "the date with year `y`, month `m` and day `d`"
            args = "s" if iso_input else "y, m, d"
            drafts.append(
                Draft(
                    slug=f"weekday-{'iso' if iso_input else 'ymd'}-{slug}",
                    fn="weekday",
                    signature=signature,
                    does=f"returns, for {given}, {said}",
                    reference=f"import datetime\ndef weekday({args}):\n    d = {parse}\n    return {expr}\n",
                    cases=cases,
                )
            )
    return drafts


def _days_between(rng: random.Random) -> list[Draft]:
    formats = [
        ("iso", "YYYY-MM-DD", "%Y-%m-%d"),
        ("dmy", "DD/MM/YYYY", "%d/%m/%Y"),
        ("compact", "YYYYMMDD", "%Y%m%d"),
    ]
    drafts = []
    for slug, shown, fmt in formats:
        for signed in (False, True):
            pairs = [(day(rng), day(rng)) for _ in range(5)] + [(datetime.date(2024, 2, 28), datetime.date(2024, 3, 1))]
            cases = [[a.strftime(fmt), b.strftime(fmt)] for a, b in pairs] + [[pairs[0][0].strftime(fmt)] * 2]
            result = "(b - a).days" if signed else "abs((b - a).days)"
            said = (
                "the number of days from date `a` to date `b`, negative when `b` is before `a`"
                if signed
                else "how many days apart the dates `a` and `b` are, as a non-negative number"
            )
            drafts.append(
                Draft(
                    slug=f"weeks-between-{slug}-{'signed' if signed else 'abs'}",
                    fn="weeks_between",
                    signature="weeks_between(a, b)",
                    does="returns how many whole weeks lie between the dates `a` and `b`: the number of days between them, ignoring which comes first, divided by 7 and rounded down"
                    if not signed
                    else "returns the number of whole weeks from date `a` to date `b`: the signed number of days from `a` to `b` divided by 7, rounded towards zero",
                    notes=(f"Both dates are written as {shown}.",),
                    reference=(
                        "import datetime\n"
                        "def weeks_between(a, b):\n"
                        f"    a = datetime.datetime.strptime(a, {fmt!r}).date()\n"
                        f"    b = datetime.datetime.strptime(b, {fmt!r}).date()\n"
                        "    days = (b - a).days\n"
                        + ("    return int(days / 7)\n" if signed else "    return abs(days) // 7\n")
                    ),
                    cases=cases,
                )
            )
            drafts.append(
                Draft(
                    slug=f"days-between-{slug}-{'signed' if signed else 'abs'}",
                    fn="days_between",
                    signature="days_between(a, b)",
                    does=f"returns {said}",
                    notes=(f"Both dates are written as {shown}.",),
                    reference=(
                        "import datetime\n"
                        "def days_between(a, b):\n"
                        f"    a = datetime.datetime.strptime(a, {fmt!r}).date()\n"
                        f"    b = datetime.datetime.strptime(b, {fmt!r}).date()\n"
                        f"    return {result}\n"
                    ),
                    cases=cases,
                )
            )
    return drafts


def _add_days(rng: random.Random) -> list[Draft]:
    outputs = [
        ("iso", "YYYY-MM-DD", "%Y-%m-%d"),
        ("dotted", "DD.MM.YYYY", "%d.%m.%Y"),
        ("us", "MM/DD/YYYY", "%m/%d/%Y"),
    ]
    drafts = []
    for (slug, shown, fmt), unit in [(o, u) for o in outputs for u in ("days", "weeks")]:
        per = 1 if unit == "days" else 7
        cases = [[day(rng).isoformat(), rng.randint(-60, 60) * (1 if unit == "weeks" else 7)] for _ in range(5)] + [
            ["2023-12-31", 1],
            ["2024-02-28", 1],
            ["2024-03-01", 0],
        ]
        drafts.append(
            Draft(
                slug=f"add-{unit}-{slug}",
                fn=f"add_{unit}",
                signature=f"add_{unit}(s, n)",
                does=f"returns the date `n` {unit} after the date `s`, written as {shown}",
                notes=("`s` is written as YYYY-MM-DD; `n` may be negative or zero.",),
                reference=(
                    "import datetime\n"
                    f"def add_{unit}(s, n):\n"
                    f"    d = datetime.date.fromisoformat(s) + datetime.timedelta(days=n * {per})\n"
                    f"    return d.strftime({fmt!r})\n"
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _month_end(rng: random.Random) -> list[Draft]:
    outputs = [
        ("date", "the date of the last day of that month, written as YYYY-MM-DD", "last.isoformat()"),
        ("length", "how many days that month has", "last.day"),
        ("weekday", "the English name of the day of the week its last day falls on", f"{DAY_NAMES}[last.weekday()]"),
        (
            "first-weekday",
            "the English name of the day of the week its first day falls on",
            f"{DAY_NAMES}[datetime.date(y, m, 1).weekday()]",
        ),
        (
            "weekend-days",
            "how many of its days are Saturdays or Sundays",
            "sum(1 for k in range(1, last.day + 1) if datetime.date(y, m, k).weekday() >= 5)",
        ),
    ]
    drafts = []
    for slug, said, expr in outputs:
        cases = [[rng.randint(1950, 2060), rng.randint(1, 12)] for _ in range(5)] + [
            [2024, 2],
            [2023, 2],
            [1900, 2],
            [2000, 12],
        ]
        drafts.append(
            Draft(
                slug=f"month-end-{slug}",
                fn="month_end",
                signature="month_end(y, m)",
                does=f"returns, for month `m` (1 to 12) of year `y`, {said}",
                notes=("Leap years follow the Gregorian calendar.",),
                reference=(
                    "import datetime\n"
                    "def month_end(y, m):\n"
                    "    first_of_next = datetime.date(y + m // 12, m % 12 + 1, 1)\n"
                    "    last = first_of_next - datetime.timedelta(days=1)\n"
                    f"    return {expr}\n"
                ),
                cases=cases,
                example=True,
            )
        )
    return drafts


def _business_days(rng: random.Random) -> list[Draft]:
    weekends = [
        ("sat-sun", "Saturday and Sunday", "(5, 6)"),
        ("fri-sat", "Friday and Saturday", "(4, 5)"),
        ("sun", "Sunday", "(6,)"),
    ]
    drafts = []
    for slug, said, days in weekends:
        cases = [[day(rng, 2000, 2040).isoformat(), rng.randint(1, 30)] for _ in range(5)] + [
            ["2026-09-26", 0],
            ["2026-09-25", 1],
            ["2026-09-23", -1],
        ]
        drafts.append(
            Draft(
                slug=f"business-days-{slug}",
                fn="add_business_days",
                signature="add_business_days(s, n)",
                does=f"returns the date reached by moving forward `n` business days from the date `s`, where the weekend is {said}",
                notes=(
                    "Dates are written as YYYY-MM-DD. Step forward one day at a time and count only the days that are not weekend days, stopping when `n` have been counted.",
                    "`n` == 0 returns `s` itself even when it falls on a weekend; a negative `n` raises ValueError.",
                ),
                reference=(
                    "import datetime\n"
                    "def add_business_days(s, n):\n"
                    "    if n < 0:\n"
                    "        raise ValueError(n)\n"
                    "    d = datetime.date.fromisoformat(s)\n"
                    "    while n > 0:\n"
                    "        d += datetime.timedelta(days=1)\n"
                    f"        if d.weekday() not in {days}:\n"
                    "            n -= 1\n"
                    "    return d.isoformat()\n"
                ),
                cases=cases,
                example=True,
                raises=("ValueError",),
            )
        )
    return drafts


def _age_on(rng: random.Random) -> list[Draft]:
    drafts = []
    for march in (True, False):
        leap_day = "on 1 March" if march else "on 28 February"
        fallback = "datetime.date(d.year, 3, 1)" if march else "datetime.date(d.year, 2, 28)"
        pairs = [(day(rng, 1940, 2000), day(rng, 2001, 2050)) for _ in range(4)]
        cases = [[b.isoformat(), d.isoformat()] for b, d in pairs] + [
            ["2000-02-29", "2023-02-28"],
            ["2000-02-29", "2023-03-01"],
            ["1990-06-15", "1990-06-15"],
            ["2010-01-01", "2009-12-31"],
        ]
        drafts.append(
            Draft(
                slug=f"age-on-{'march' if march else 'february'}",
                fn="age_on",
                signature="age_on(birth, on)",
                does="returns the age in whole years, on the date `on`, of someone born on the date `birth`",
                notes=(
                    "Both dates are written as YYYY-MM-DD. The age goes up by one on each birthday.",
                    f"Someone born on 29 February has their birthday {leap_day} in years that are not leap years.",
                    "`on` earlier than `birth` raises ValueError.",
                ),
                reference=(
                    "import datetime\n"
                    "def age_on(birth, on):\n"
                    "    b = datetime.date.fromisoformat(birth)\n"
                    "    d = datetime.date.fromisoformat(on)\n"
                    "    if d < b:\n"
                    "        raise ValueError('before birth')\n"
                    "    years = d.year - b.year\n"
                    "    try:\n"
                    "        birthday = b.replace(year=d.year)\n"
                    "    except ValueError:\n"
                    f"        birthday = {fallback}\n"
                    "    if d < birthday:\n"
                    "        years -= 1\n"
                    "    return years\n"
                ),
                cases=cases,
                example=True,
                raises=("ValueError",),
            )
        )
    return drafts


def _calendar_position(rng: random.Random) -> list[Draft]:
    outputs = [
        ("iso-week", "its ISO 8601 week number, from 1 to 53", "d.isocalendar()[1]"),
        (
            "iso-week-string",
            "its ISO 8601 week written as YYYY-Www, such as '2024-W05', using the ISO week-numbering year",
            "f'{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}'",
        ),
        ("day-of-year", "its day of the year, 1 for 1 January", "d.timetuple().tm_yday"),
        (
            "quarter",
            "its quarter of the year, 1 for January to March through 4 for October to December",
            "(d.month - 1) // 3 + 1",
        ),
    ]
    drafts = []
    for slug, said, expr in outputs:
        cases = [[day(rng).isoformat()] for _ in range(5)] + [["2021-01-01"], ["2024-12-30"], ["2024-12-31"]]
        drafts.append(
            Draft(
                slug=slug,
                fn="calendar_position",
                signature="calendar_position(s)",
                does=f"returns, for the date `s` written as YYYY-MM-DD, {said}",
                reference=f"import datetime\ndef calendar_position(s):\n    d = datetime.date.fromisoformat(s)\n    return {expr}\n",
                cases=cases,
            )
        )
    return drafts


TEMPLATES = [
    Template("weekday", "small", _weekday),
    Template("calendar-position", "small", _calendar_position),
    Template("days-between", "small", _days_between),
    Template("add-days", "medium", _add_days),
    Template("month-end", "medium", _month_end),
    Template("business-days", "large", _business_days),
    Template("age-on", "large", _age_on),
]

IMPOSSIBLE = [
    Impossible(
        slug="february-thirtieth",
        fn="february_thirtieth",
        signature="february_thirtieth(y)",
        does="returns the date of 30 February in the Gregorian year `y`, written as YYYY-MM-DD",
        cases=[[2024], [2023], [1900]],
        attempt="def february_thirtieth(y):\n    return f'{y}-03-01'\n",
    ),
    Impossible(
        slug="before-and-after",
        fn="before_and_after",
        signature="before_and_after(s)",
        does="returns a date, written as YYYY-MM-DD, that is both strictly before and strictly after the date `s`",
        cases=[["2024-01-15"], ["1999-12-31"], ["2030-06-01"]],
        attempt="def before_and_after(s):\n    return s\n",
    ),
]
