"""Extract level elevations from ARCH section/elevation sheets.

Revit section sheets carry level datums as real vector text: a name span
("LEVEL 01", "TOP OF ROOF") with the elevation span ("7.880m") drawn right
next to it, repeated once per section column on the sheet.  Both spans share
the same text direction (often rotated 90 deg), so pairing is done by bbox
proximity, not reading order.

All values are evidence only — they feed floor heights / column lengths into
the export, never geometry coordinates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz

# "7.880m", "46.000 m"
_ELEV_RE = re.compile(r"^(\d{1,3}\.\d{3})\s*m$")
# datum names worth keeping (project-agnostic core set)
_NAME_RE = re.compile(
    r"^(LEVEL\s+\S+(?:\s\S+)?|GROUND\s+LEVEL|TOP\s+OF\s+ROOF|ROOF(?:\s+LEVEL)?|"
    r"NATURAL\s+GROUND|BASEMENT\s*\d*|PODIUM|MEZZANINE)$",
    re.IGNORECASE,
)
# max centre-to-centre distance (pt) between a name span and its elevation
_PAIR_DIST_PT = 60.0
# elevations from different section columns must agree within this (m)
_CONSENSUS_TOL_M = 0.001


@dataclass
class LevelTable:
    """Consensus level elevations for one sheet."""
    elevations_m: dict[str, float] = field(default_factory=dict)
    floor_to_floor_m: dict[str, float] = field(default_factory=dict)
    conflicts: dict[str, list[float]] = field(default_factory=dict)
    n_datums: int = 0


def _spans(page: fitz.Page) -> list[tuple[str, tuple[float, float]]]:
    out = []
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            for sp in ln.get("spans", []):
                t = sp["text"].strip()
                if not t:
                    continue
                x0, y0, x1, y1 = sp["bbox"]
                out.append((t, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))
    return out


def extract_levels(page: fitz.Page) -> LevelTable:
    spans = _spans(page)
    names = [(t.upper(), c) for t, c in spans if _NAME_RE.match(t)]
    elevs = []
    for t, c in spans:
        m = _ELEV_RE.match(t)
        if m:
            elevs.append((float(m.group(1)), c))

    samples: dict[str, list[float]] = {}
    for val, (ex, ey) in elevs:
        best_name, best_d = None, _PAIR_DIST_PT
        for name, (nx, ny) in names:
            d = ((ex - nx) ** 2 + (ey - ny) ** 2) ** 0.5
            if d < best_d:
                best_name, best_d = name, d
        if best_name is not None:
            samples.setdefault(best_name, []).append(val)

    table = LevelTable(n_datums=sum(len(v) for v in samples.values()))
    for name, vals in samples.items():
        lo, hi = min(vals), max(vals)
        if hi - lo > _CONSENSUS_TOL_M:
            table.conflicts[name] = sorted(set(vals))
            continue
        table.elevations_m[name] = vals[0]

    # floor-to-floor: gap to the next level above (by elevation order)
    ordered = sorted(table.elevations_m.items(), key=lambda kv: kv[1])
    for (name, z), (_, z_above) in zip(ordered, ordered[1:]):
        table.floor_to_floor_m[name] = round(z_above - z, 4)
    return table
