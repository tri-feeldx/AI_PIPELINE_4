"""Deterministic parsing of on-page COLUMN/WALL schedules (Phase 2.1).

GA sheets print the element schedules as vector text.  The parser anchors
on the schedule title ("... COLUMN SCHEDULE", "WALL SCHEDULE"), takes the
MARK header below it, groups the words underneath into rows by baseline,
and reads each row positionally.  No AI involved — Gemini census becomes a
cross-check against this, never the source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz

_COL_MARK_RE = re.compile(r"^C-[A-Z]\d?(?:\(u\))?$")
_WALL_MARK_RE = re.compile(r"^(?:[A-Z]{2}\d{1,2}|NLB\d?)$")
_SIZE_RE = re.compile(r"^(\d{3})x(\d{3,4})$")
_INT_RE = re.compile(r"^\d{2,4}$")

_ROW_TOL_PT = 6.0       # words within this y-distance belong to one row
_TABLE_MAX_ROWS = 40
_TABLE_WIDTH_PT = 420.0  # generous width right of the MARK column


@dataclass
class ColumnType:
    mark: str
    size_mm: tuple | None = None            # (w, h)
    concrete_grade: int | None = None
    reinforcement_rate_kg_m3: int | None = None


@dataclass
class WallType:
    mark: str
    thickness_mm: int | None = None
    description: str = ""


@dataclass
class PageSchedules:
    columns: dict = field(default_factory=dict)   # mark -> ColumnType
    walls: dict = field(default_factory=dict)     # mark -> WallType


def _rows_below(words, x0, y0, x1, y_limit):
    """Words in the column strip below y0, grouped into baseline rows."""
    inside = [w for w in words
              if x0 <= w[0] <= x1 and y0 < w[1] <= y_limit]
    inside.sort(key=lambda w: (w[1], w[0]))
    rows, cur, cur_y = [], [], None
    for w in inside:
        if cur_y is None or abs(w[1] - cur_y) <= _ROW_TOL_PT:
            cur.append(w)
            cur_y = w[1] if cur_y is None else cur_y
        else:
            rows.append(cur)
            cur, cur_y = [w], w[1]
    if cur:
        rows.append(cur)
    return rows[:_TABLE_MAX_ROWS]


def _find_mark_header(words, title_word):
    """The MARK header cell nearest below the schedule title."""
    tx, ty = title_word[0], title_word[3]
    cands = [w for w in words if w[4] == "MARK"
             and w[1] > ty - 5 and abs(w[0] - tx) < 400]
    return min(cands, key=lambda w: w[1] - ty, default=None)


def parse_schedules(page: fitz.Page) -> PageSchedules:
    words = page.get_text("words")
    out = PageSchedules()

    titles = [w for w in words if w[4] == "SCHEDULE"]
    for t in titles:
        # context word before the title distinguishes the tables
        before = [w for w in words
                  if abs(w[1] - t[1]) < 4 and 0 < t[0] - w[2] < 120]
        ctx = " ".join(w[4] for w in sorted(before, key=lambda w: w[0]))
        kind = ("column" if "COLUMN" in ctx.upper()
                else "wall" if "WALL" in ctx.upper() else None)
        if kind is None:
            continue
        mark_hdr = _find_mark_header(words, t)
        if mark_hdr is None:
            continue
        x0 = mark_hdr[0] - 10
        rows = _rows_below(words, x0, mark_hdr[3],
                           mark_hdr[0] + _TABLE_WIDTH_PT,
                           mark_hdr[3] + 600)
        for row in rows:
            row.sort(key=lambda w: w[0])
            texts = [w[4] for w in row]
            if not texts:
                continue
            mark = texts[0]
            if kind == "column" and _COL_MARK_RE.match(mark):
                col = ColumnType(mark=mark)
                ints = []
                for tk in texts[1:]:
                    m = _SIZE_RE.match(tk)
                    if m:
                        col.size_mm = (int(m.group(1)), int(m.group(2)))
                    elif _INT_RE.match(tk):
                        ints.append(int(tk))
                if ints:
                    col.concrete_grade = ints[0]
                if len(ints) >= 2:
                    col.reinforcement_rate_kg_m3 = ints[-1]
                out.columns.setdefault(mark, col)
            elif kind == "wall" and _WALL_MARK_RE.match(mark):
                wall = WallType(mark=mark)
                rest = []
                for tk in texts[1:]:
                    if _INT_RE.match(tk) and wall.thickness_mm is None:
                        wall.thickness_mm = int(tk)
                    else:
                        rest.append(tk)
                wall.description = " ".join(rest)
                out.walls.setdefault(mark, wall)
    return out
