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

# marks across offices: "C-A1"/"C-B" (2381), "C2"/"CC1"/"SC1"/"M2" (SMPS)
_COL_MARK_RE = re.compile(
    r"^(?:C-[A-Z]\d?(?:\(u\))?|[A-Z]{1,3}\d{1,3}[A-Z]?)$")
_WALL_MARK_RE = re.compile(r"^(?:[A-Z]{2}\d{1,2}|NLB\d?)$")
_SIZE_RE = re.compile(r"^(\d{3})x(\d{3,4})$")
_INT_RE = re.compile(r"^\d{2,4}$")
# rolled steel sections are unambiguous: 250UC90, 360UB51
_STEEL_ROLLED_RE = re.compile(
    r"^\d{2,4}(?:\.\d)?(?:UB|UC|PFC|WB|WC)\d{1,4}(?:\.\d)?$")
# hollow sections are "125x6.0" + a shape token; without the token an
# NxM string is an RC size (450x800)
_STEEL_HOLLOW_RE = re.compile(r"^\d{2,4}(?:\.\d)?x\d{1,3}(?:\.\d)?$")
_STEEL_SHAPE_RE = re.compile(r"^(?:SHS|CHS|RHS|EA|UA)$")
_MAIN_BARS_RE = re.compile(r"^\d{1,2}[NY]\d{2}$")
_LIGATURE_RE = re.compile(r"^[RNL]\d{1,2}-\d{2,4}$")

_ROW_TOL_PT = 6.0       # words within this y-distance belong to one row
_TABLE_MAX_ROWS = 40
_TABLE_WIDTH_PT = 420.0  # generous width right of the MARK column


@dataclass
class ColumnType:
    mark: str
    size_mm: tuple | None = None            # (w, h) rectangular RC
    diameter_mm: int | None = None          # circular RC
    concrete_grade: int | None = None
    reinforcement_rate_kg_m3: int | None = None
    material: str = "RC"                    # RC | STEEL
    section: str = ""                       # steel section string (250UC90)
    main_bars: str = ""                     # e.g. 8N20
    ligatures: str = ""                     # e.g. R10-300

    def rebar_mass_kg(self, height_mm: float) -> float | None:
        """Reinforcement mass of one column of this type at the given
        storey height, from the schedule's rate (kg per m3 of concrete)."""
        if self.size_mm is None or self.reinforcement_rate_kg_m3 is None:
            return None
        w, h = self.size_mm
        vol_m3 = (w / 1000.0) * (h / 1000.0) * (height_mm / 1000.0)
        return round(vol_m3 * self.reinforcement_rate_kg_m3, 1)


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
    cands = [w for w in words if w[4].rstrip(":") == "MARK"
             and w[1] > ty - 5 and abs(w[0] - tx) < 400]
    return min(cands, key=lambda w: w[1] - ty, default=None)


def _display_words(page: fitz.Page) -> list:
    """Words with bboxes in DISPLAY coordinates (rotation applied).

    Rotated sheets (SMPS exports carry /Rotate 90) return raw-space word
    boxes from get_text; row grouping by baseline only works in display
    space. Unrotated pages pass through unchanged.
    """
    words = page.get_text("words")
    if not page.rotation:
        return words
    m = page.rotation_matrix
    out = []
    for w in words:
        r = fitz.Rect(w[:4]) * m
        r.normalize()
        out.append((r.x0, r.y0, r.x1, r.y1) + tuple(w[4:]))
    return out


def parse_schedules(page: fitz.Page) -> PageSchedules:
    words = _display_words(page)
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
        material = "STEEL" if "STEEL" in ctx.upper() else "RC"
        mark_hdr = _find_mark_header(words, t)
        if mark_hdr is None:
            continue
        x0 = mark_hdr[0] - 10
        # the table ends where the next schedule title starts (stacked
        # tables must not bleed into each other)
        next_titles = [o[1] for o in titles
                       if o[1] > mark_hdr[3] + 5 and abs(o[0] - t[0]) < 400]
        y_limit = min([mark_hdr[3] + 600] + [y - 5 for y in next_titles])
        rows = _rows_below(words, x0, mark_hdr[3],
                           mark_hdr[0] + _TABLE_WIDTH_PT, y_limit)
        for row in rows:
            row.sort(key=lambda w: w[0])
            texts = [w[4] for w in row]
            if not texts:
                continue
            mark = texts[0]
            if kind == "column" and _COL_MARK_RE.match(mark):
                col = ColumnType(mark=mark, material=material)
                ints = []
                for i, tk in enumerate(texts[1:], start=1):
                    m = _SIZE_RE.match(tk)
                    nxt = texts[i + 1] if i + 1 < len(texts) else ""
                    if _STEEL_ROLLED_RE.match(tk):
                        col.material = "STEEL"
                        col.section = tk
                    elif _STEEL_HOLLOW_RE.match(tk) and _STEEL_SHAPE_RE.match(nxt):
                        # "450x800" alone is an RC size; "125x6.0 SHS" is
                        # a hollow steel section — the shape token decides
                        col.material = "STEEL"
                        col.section = f"{tk} {nxt}"
                    elif m:
                        col.size_mm = (int(m.group(1)), int(m.group(2)))
                    elif _MAIN_BARS_RE.match(tk):
                        col.main_bars = tk
                    elif _LIGATURE_RE.match(tk):
                        col.ligatures = tk
                    elif _INT_RE.match(tk):
                        ints.append(int(tk))
                if col.material == "RC" and col.size_mm is None \
                        and ints and 150 <= ints[0] <= 2000:
                    col.diameter_mm = ints.pop(0)   # circular column
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
