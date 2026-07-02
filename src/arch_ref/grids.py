"""Structural grid extraction from drawing sheets (ARCH or STR).

Grid bubbles are small circles at the sheet edges with a short label inside
(A..Z rows, 1..n columns).  Revit exports draw them either as bezier curves
(ARCH sets) or as many-segment polylines (STR sets), so detection keys on the
bounding box being square and the item count, not the primitive type.

Circles are also used for element tags (PT, RC ...) inside the plan; real
grid bubbles are separated from tags by the alignment rule: a family needs
at least 3 bubbles sharing the same x (row axes) or the same y (column axes).

Coordinates returned are bubble centres in page pt — evidence for aligning
sheets to a common grid, never exported geometry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz

_LABEL_RE = re.compile(r"^([A-Z]{1,2}|\d{1,2})$")
_MIN_DIAM_PT = 10.0
_MAX_DIAM_PT = 60.0
_SQUARENESS_PT = 4.0
_MIN_ITEMS = 4          # a circle needs several curve/line items
_ALIGN_TOL_PT = 3.0     # bubbles of one family share x (rows) or y (cols)
_MIN_FAMILY = 3


@dataclass
class GridSet:
    """Grid axes of one sheet: label -> bubble-centre coordinate (pt)."""
    cols: dict[str, float] = field(default_factory=dict)  # label -> x
    rows: dict[str, float] = field(default_factory=dict)  # label -> y


def _bubbles(page: fitz.Page) -> list[tuple[str, float, float]]:
    words = page.get_text("words")
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if not (_MIN_DIAM_PT < r.width < _MAX_DIAM_PT
                and abs(r.width - r.height) < _SQUARENESS_PT):
            continue
        if len(d["items"]) < _MIN_ITEMS:
            continue
        inside = [w[4] for w in words
                  if r.x0 - 1 < w[0] and w[2] < r.x1 + 1
                  and r.y0 - 1 < w[1] and w[3] < r.y1 + 1]
        if len(inside) == 1 and _LABEL_RE.match(inside[0]):
            out.append((inside[0], (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0))
    return out


def _largest_aligned_family(
    bubbles: list[tuple[str, float, float]], coord_idx: int,
) -> list[tuple[str, float, float]]:
    """Largest group of bubbles sharing the same coordinate (x or y)."""
    best: list[tuple[str, float, float]] = []
    for _, *anchor in bubbles:
        group = [b for b in bubbles
                 if abs((b[1], b[2])[coord_idx] - anchor[coord_idx]) <= _ALIGN_TOL_PT]
        # one label per axis within the family
        seen: dict[str, tuple[str, float, float]] = {}
        for b in group:
            seen.setdefault(b[0], b)
        group = list(seen.values())
        if len(group) > len(best):
            best = group
    return best if len(best) >= _MIN_FAMILY else []


def extract_grid(page: fitz.Page) -> GridSet:
    bubbles = _bubbles(page)
    numeric = [b for b in bubbles if b[0].isdigit()]
    alpha = [b for b in bubbles if not b[0].isdigit()]

    grid = GridSet()
    # column axes (numbered): aligned on a shared y, vary in x
    for label, x, _ in _largest_aligned_family(numeric, 1):
        grid.cols[label] = x
    # row axes (lettered): aligned on a shared x, vary in y
    for label, _, y in _largest_aligned_family(alpha, 0):
        grid.rows[label] = y
    return grid
