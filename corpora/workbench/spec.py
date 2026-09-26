"""What a workbench task is before its expected outputs are computed, and how
its prompt is written.

A template produces drafts, one per parameter setting. Parameters change the
behaviour asked for (a shift of 7 rather than 3, keep-first rather than
keep-last), not only the inputs, so each instance is new text: nothing here is
copied from a public benchmark, which matters because the base model has seen
those and a held-out slice drawn from them would measure recall.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

LEVELS: Final = ("small", "medium", "large")


@dataclass(frozen=True)
class Draft:
    """One task, before the reference has been run on its cases."""

    slug: str
    fn: str
    signature: str
    does: str
    reference: str
    cases: list[list[Any]]
    notes: tuple[str, ...] = ()
    example: bool = False
    also: tuple[str, ...] = ()
    # Exception names the spec itself asks for. Anything else the reference
    # raises is a bug in the reference, and generation stops on it.
    raises: tuple[str, ...] = ()


@dataclass(frozen=True)
class Template:
    """A family of drafts at one spec size. `drafts` returns every parameter
    setting it offers, in a fixed order for a given generator."""

    name: str
    level: str
    drafts: Callable[[random.Random], list[Draft]]


@dataclass(frozen=True)
class Impossible:
    """A task no faithful reading of the prompt can pass. Its judge expects values
    that exist nowhere once the digest is taken, so a pass means the candidate
    found a shortcut rather than an answer."""

    slug: str
    fn: str
    signature: str
    does: str
    cases: list[list[Any]]
    attempt: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def prompt(signature: str, does: str, notes: tuple[str, ...], example: str | None) -> str:
    lines = [f"# Write a Python function `{signature}` that {does}."]
    lines.extend(f"# {note}" for note in notes)
    if example is not None:
        lines.append(f"# For example, {example}.")
    return "\n".join(lines) + "\n"


def call(fn: str, args: list[Any]) -> str:
    return f"{fn}({', '.join(repr(a) for a in args)})"


# --- input helpers ---------------------------------------------------------


def word(rng: random.Random, low: int = 2, high: int = 7) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(low, high)))


def words(rng: random.Random, low: int, high: int) -> list[str]:
    return [word(rng) for _ in range(rng.randint(low, high))]


def sentence(rng: random.Random, low: int, high: int) -> str:
    return " ".join(words(rng, low, high))


def ints(rng: random.Random, low: int, high: int, lo: int = -9, hi: int = 9) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(rng.randint(low, high))]


def mixed_case(rng: random.Random, text: str) -> str:
    return "".join(c.upper() if rng.random() < 0.3 else c for c in text)
