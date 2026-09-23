"""Grids: rectangular lists of lists of integers."""

from __future__ import annotations

import random

from spec import Draft, Impossible, Template

SKILL = "grids"

ORTHOGONAL = "[(-1, 0), (1, 0), (0, -1), (0, 1)]"
ALL_EIGHT = "[(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]"


def grid(rng: random.Random, rows: int, cols: int, lo: int = -9, hi: int = 9) -> list[list[int]]:
    return [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def shapes(rng: random.Random, count: int, lo: int = -9, hi: int = 9) -> list[list[list[int]]]:
    return [grid(rng, rng.randint(1, 4), rng.randint(1, 4), lo, hi) for _ in range(count)]


def _reshape(rng: random.Random) -> list[Draft]:
    variants = [
        (
            "transpose",
            "transpose",
            "returns the transpose of the grid `g`: row i of the result is column i of `g`",
            "    return [list(r) for r in zip(*g)]\n",
        ),
        (
            "mirror",
            "mirror",
            "returns the grid `g` mirrored left to right: each row reversed",
            "    return [row[::-1] for row in g]\n",
        ),
        (
            "flip",
            "flip",
            "returns the grid `g` flipped top to bottom: its rows in reverse order",
            "    return [list(row) for row in g[::-1]]\n",
        ),
        (
            "anti-transpose",
            "anti_transpose",
            "returns the grid `g` reflected across its anti-diagonal, the diagonal from its top-right corner to its bottom-left corner",
            "    if not g:\n        return []\n    rows, cols = len(g), len(g[0])\n"
            "    return [[g[rows - 1 - j][cols - 1 - i] for j in range(rows)] for i in range(cols)]\n",
        ),
        (
            "diagonal",
            "diagonal",
            "returns the values on the main diagonal of the grid `g` as a list: g[0][0], g[1][1], and so on while both indices are inside the grid",
            "    return [g[i][i] for i in range(min(len(g), len(g[0]) if g else 0))]\n",
        ),
        (
            "anti-diagonal",
            "anti_diagonal",
            "returns the values on the anti-diagonal of the grid `g` as a list, starting at its top-right corner: g[0][c - 1], g[1][c - 2], and so on while both indices are inside the grid, where c is the number of columns",
            "    if not g:\n        return []\n    cols = len(g[0])\n    return [g[i][cols - 1 - i] for i in range(min(len(g), cols))]\n",
        ),
    ]
    drafts = []
    for slug, fn, does, body in variants:
        cases = [[g] for g in shapes(rng, 5)] + [[[]], [[[1, 2, 3]]]]
        drafts.append(
            Draft(
                slug=f"reshape-{slug}",
                fn=fn,
                signature=f"{fn}(g)",
                does=does,
                notes=("`g` is a non-empty list of equally long, non-empty rows, or [] which returns [].",),
                reference=f"def {fn}(g):\n{body}",
                cases=cases,
            )
        )
    return drafts


def _line_totals(rng: random.Random) -> list[Draft]:
    variants = [
        ("row-sums", "the sum of each row", "[sum(row) for row in g]"),
        ("column-sums", "the sum of each column", "[sum(col) for col in zip(*g)]"),
        ("row-max", "the largest value in each row", "[max(row) for row in g]"),
        ("column-min", "the smallest value in each column", "[min(col) for col in zip(*g)]"),
        ("row-spread", "the largest minus the smallest value in each row", "[max(row) - min(row) for row in g]"),
        ("column-max", "the largest value in each column", "[max(col) for col in zip(*g)]"),
        ("row-min", "the smallest value in each row", "[min(row) for row in g]"),
        (
            "column-spread",
            "the largest minus the smallest value in each column",
            "[max(col) - min(col) for col in zip(*g)]",
        ),
    ]
    drafts = []
    for slug, said, expr in variants:
        cases = [[g] for g in shapes(rng, 5)] + [[[]], [[[4]]]]
        drafts.append(
            Draft(
                slug=f"lines-{slug}",
                fn="line_totals",
                signature="line_totals(g)",
                does=f"returns a list holding {said} of the grid `g`, in order",
                notes=("`g` is a list of equally long, non-empty rows; an empty grid returns [].",),
                reference=f"def line_totals(g):\n    return {expr}\n",
                cases=cases,
            )
        )
    return drafts


def _rotate(rng: random.Random) -> list[Draft]:
    variants = [
        ("clockwise", "90 degrees clockwise", "[list(r) for r in zip(*g[::-1])]"),
        ("counterclockwise", "90 degrees counter-clockwise", "[list(r) for r in zip(*g)][::-1]"),
        ("half-turn", "180 degrees", "[row[::-1] for row in g[::-1]]"),
    ]
    drafts = []
    for slug, said, expr in variants:
        cases = [[g] for g in shapes(rng, 5)] + [[[]], [[[1, 2], [3, 4], [5, 6]]]]
        drafts.append(
            Draft(
                slug=f"rotate-{slug}",
                fn="rotate_grid",
                signature="rotate_grid(g)",
                does=f"returns a new grid: the grid `g` rotated {said}",
                notes=("`g` may be rectangular rather than square; an empty grid returns [].",),
                reference=f"def rotate_grid(g):\n    return {expr}\n",
                cases=cases,
                example=True,
            )
        )
    return drafts


def _neighbours(rng: random.Random) -> list[Draft]:
    reducers = [
        ("sum", "the sum of", "sum(values)", ""),
        ("nonzero", "how many are non-zero among", "sum(1 for v in values if v != 0)", ""),
        ("max", "the largest of", "max(values) if values else None", " Return None when the cell has no neighbours."),
        ("min", "the smallest of", "min(values) if values else None", " Return None when the cell has no neighbours."),
    ]
    drafts = []
    for diagonal in (False, True):
        offsets = ALL_EIGHT if diagonal else ORTHOGONAL
        which = (
            "8 surrounding cells, diagonals included"
            if diagonal
            else "up to 4 cells directly above, below, left and right of it"
        )
        for slug, said, expr, extra in reducers:
            cases = []
            for g in shapes(rng, 6, 0, 9) + [[[5]]]:
                cases.append([g, rng.randrange(len(g)), rng.randrange(len(g[0]))])
            drafts.append(
                Draft(
                    slug=f"neighbours-{8 if diagonal else 4}-{slug}",
                    fn="neighbours",
                    signature="neighbours(g, r, c)",
                    does=f"returns {said} the values of the neighbours of the cell at row `r`, column `c` of the grid `g`",
                    notes=(
                        f"The neighbours are the {which}, ignoring positions outside the grid; the cell itself is not a neighbour.{extra}",
                    ),
                    reference=(
                        "def neighbours(g, r, c):\n"
                        "    values = []\n"
                        f"    for dr, dc in {offsets}:\n"
                        "        i, j = r + dr, c + dc\n"
                        "        if 0 <= i < len(g) and 0 <= j < len(g[0]):\n"
                        "            values.append(g[i][j])\n"
                        f"    return {expr}\n"
                    ),
                    cases=cases,
                    example=True,
                )
            )
    return drafts


def _islands(rng: random.Random) -> list[Draft]:
    drafts = []
    for diagonal in (False, True):
        for largest in (False, True):
            for any_nonzero in (False, True):
                offsets = ALL_EIGHT if diagonal else ORTHOGONAL
                land = "any non-zero value" if any_nonzero else "the value 1"
                test = "g[i][j] != 0" if any_nonzero else "g[i][j] == 1"
                touching = "including diagonally" if diagonal else "horizontally or vertically, not diagonally"
                answer = (
                    "the number of cells in the largest island (0 when there is none)"
                    if largest
                    else "how many islands there are"
                )
                cases = [[grid(rng, rng.randint(2, 6), rng.randint(2, 6), 0, 2)] for _ in range(6)] + [
                    [[[0, 0], [0, 0]]],
                    [[[1]]],
                ]
                drafts.append(
                    Draft(
                        slug=f"islands-{8 if diagonal else 4}-{'largest' if largest else 'count'}-{'nonzero' if any_nonzero else 'one'}",
                        fn="islands",
                        signature="islands(g)",
                        does=f"returns {answer} in the grid `g`",
                        notes=(
                            f"A cell holding {land} is land; every other cell is water.",
                            f"An island is a group of land cells connected through cells that touch {touching}.",
                        ),
                        reference=(
                            "def islands(g):\n"
                            "    seen = set()\n"
                            "    sizes = []\n"
                            "    for r in range(len(g)):\n"
                            "        for c in range(len(g[0])):\n"
                            f"            i, j = r, c\n"
                            f"            if (r, c) in seen or not ({test}):\n"
                            "                continue\n"
                            "            stack = [(r, c)]\n"
                            "            seen.add((r, c))\n"
                            "            size = 0\n"
                            "            while stack:\n"
                            "                a, b = stack.pop()\n"
                            "                size += 1\n"
                            f"                for dr, dc in {offsets}:\n"
                            "                    i, j = a + dr, b + dc\n"
                            f"                    if 0 <= i < len(g) and 0 <= j < len(g[0]) and (i, j) not in seen and {test}:\n"
                            "                        seen.add((i, j))\n"
                            "                        stack.append((i, j))\n"
                            "            sizes.append(size)\n"
                            + ("    return max(sizes, default=0)\n" if largest else "    return len(sizes)\n")
                        ),
                        cases=cases,
                        example=True,
                    )
                )
    return drafts


def _spiral(rng: random.Random) -> list[Draft]:
    variants = [
        ("clockwise-top-left", "clockwise, starting at the top-left corner and moving right along the top row", "g"),
        (
            "counterclockwise-top-left",
            "counter-clockwise, starting at the top-left corner and moving down the left column",
            "[list(r) for r in zip(*g)]",
        ),
        (
            "counterclockwise-top-right",
            "counter-clockwise, starting at the top-right corner and moving left along the top row",
            "[row[::-1] for row in g]",
        ),
        (
            "clockwise-bottom-right",
            "clockwise, starting at the bottom-right corner and moving left along the bottom row",
            "[row[::-1] for row in g[::-1]]",
        ),
    ]
    drafts = []
    for slug, said, start in variants:
        cases = [[grid(rng, rng.randint(1, 5), rng.randint(1, 5))] for _ in range(5)] + [
            [[]],
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
        ]
        drafts.append(
            Draft(
                slug=f"spiral-{slug}",
                fn="spiral",
                signature="spiral(g)",
                does=f"returns the values of the grid `g` in spiral order, {said} and turning inwards",
                notes=("Every value appears exactly once; an empty grid returns [].",),
                reference=(
                    "def spiral(g):\n"
                    f"    g = {start}\n"
                    "    out = []\n"
                    "    while g:\n"
                    "        out.extend(g[0])\n"
                    "        g = [list(r) for r in zip(*g[1:])][::-1]\n"
                    "    return out\n"
                ),
                cases=cases,
                example=True,
                also=("lists",),
            )
        )
    return drafts


TEMPLATES = [
    Template("reshape", "small", _reshape),
    Template("line-totals", "small", _line_totals),
    Template("rotate", "medium", _rotate),
    Template("neighbours", "medium", _neighbours),
    Template("islands", "large", _islands),
    Template("spiral", "large", _spiral),
]

IMPOSSIBLE = [
    Impossible(
        slug="above-the-total",
        fn="above_the_total",
        signature="above_the_total(g)",
        does="returns a value from the grid `g` of positive integers that is greater than the sum of all the values in `g`",
        cases=[[[[1, 2], [3, 4]]], [[[9]]], [[[5, 5, 5]]]],
        attempt="def above_the_total(g):\n    return max(max(row) for row in g)\n",
    ),
    Impossible(
        slug="fifth-corner",
        fn="fifth_corner",
        signature="fifth_corner(g)",
        does="returns the value in the fifth corner of the rectangular grid `g`, the one that is none of its four corners",
        cases=[[[[1, 2], [3, 4]]], [[[9, 8, 7], [6, 5, 4]]], [[[0]]]],
        attempt="def fifth_corner(g):\n    return g[len(g) // 2][len(g[0]) // 2]\n",
    ),
]
