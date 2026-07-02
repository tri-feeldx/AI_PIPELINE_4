"""ARCH -> STR height enrichment (Phase 2.5.4).

build_level_table scans an ARCH set once: section sheets give the level
elevations (levels.py, multi-sheet consensus), plan sheets give split-deck
RL zones (zone_levels.py) which are attached to the level whose elevation
matches their LOW deck.  map_str_pages then matches STR sheet titles
("GENERAL ARRANGEMENT PLAN - LEVEL 01") to level names so the building
export can stack storeys at real FFLs with real storey heights.

Confidence is fail-closed: a level is VERIFIED only when the section
consensus had no conflicts; anything else keeps confidence NONE and the
export falls back to defaults WITH a warning, never silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz

from src.arch_ref.levels import extract_levels
from src.arch_ref.zone_levels import extract_zone_levels

_MIN_SECTION_LEVELS = 5     # a sheet counts as a section sheet at >= this
_ZONE_MATCH_TOL_M = 0.005   # RL low deck must equal a level elevation


@dataclass
class LevelInfo:
    name: str
    elevation_m: float
    floor_to_floor_m: float | None = None
    zones: list = field(default_factory=list)   # [{"rl_m", "positions", "page_no"}]
    confidence: str = "NONE"                    # VERIFIED | NONE


@dataclass
class ArchLevelTable:
    levels: dict = field(default_factory=dict)  # name -> LevelInfo
    warnings: list = field(default_factory=list)


def build_level_table(arch_pdf: str) -> ArchLevelTable:
    doc = fitz.open(arch_pdf)
    out = ArchLevelTable()

    # 1) sections: merge level tables from every sheet that carries datums
    merged: dict[str, list[float]] = {}
    f2f: dict[str, float] = {}
    for pno, page in enumerate(doc, start=1):
        t = extract_levels(page)
        if len(t.elevations_m) < _MIN_SECTION_LEVELS:
            continue
        if t.conflicts:
            # ambiguous pairing on ONE sheet (working sections carry extra
            # construction elevations) — that sheet simply contributes
            # nothing for those names; cross-sheet consensus decides
            out.warnings.append(
                f"ARCH p{pno}: ambiguous datum pairing for "
                f"{sorted(t.conflicts)} — sheet skipped for those levels")
        for name, z in t.elevations_m.items():
            merged.setdefault(name, []).append(z)
        f2f.update(t.floor_to_floor_m)
    for name, vals in merged.items():
        if max(vals) - min(vals) > _ZONE_MATCH_TOL_M:
            out.warnings.append(
                f"{name}: section sheets disagree ({sorted(set(vals))})")
            out.levels[name] = LevelInfo(
                name=name, elevation_m=min(vals), confidence="NONE")
            continue
        out.levels[name] = LevelInfo(
            name=name, elevation_m=vals[0],
            floor_to_floor_m=f2f.get(name),
            confidence="VERIFIED" if len(vals) >= 2 else "NONE")

    if not out.levels:
        return out

    # 2) plans: attach split-deck RL zones to the level of their LOW deck
    by_elev = sorted(out.levels.values(), key=lambda l: l.elevation_m)
    for pno, page in enumerate(doc, start=1):
        zl = extract_zone_levels(page)
        if not zl.zones:
            continue
        low = min(zl.zones)
        match = next((l for l in by_elev
                      if abs(l.elevation_m - low) <= _ZONE_MATCH_TOL_M), None)
        if match is None:
            out.warnings.append(
                f"ARCH p{pno}: RL {low} matches no section level — flagged")
            continue
        for rl, pts in zl.zones.items():
            if len(pts) < 2:
                out.warnings.append(
                    f"ARCH p{pno}: zone RL {rl} confirmed by only "
                    f"{len(pts)} label(s) — ignored")
                continue
            match.zones.append(
                {"rl_m": rl, "positions": pts, "page_no": pno})
    return out


# ── STR page mapping ────────────────────────────────────────────────────────

_LEVEL_IN_TITLE_RE = re.compile(
    r"(?:LEVEL\s*0?(\d{1,2})|GROUND\s+FLOOR|\bROOF\b)", re.IGNORECASE)


def _level_name_from_title(title: str) -> str | None:
    m = _LEVEL_IN_TITLE_RE.search(title or "")
    if not m:
        return None
    if m.group(1) is not None:
        return f"LEVEL {int(m.group(1)):02d}"
    if "GROUND" in m.group(0).upper():
        return "LEVEL GROUND"
    return "TOP OF ROOF"        # roof slab sits at the roof datum


def map_str_pages(str_pdf: str, table: ArchLevelTable) -> list[dict]:
    """One entry per STR page whose title names a level in the table."""
    from src.slab_v2.pipeline import _page_text_audits

    doc = fitz.open(str_pdf)
    out = []
    for pi in range(len(doc)):
        _, _, role = _page_text_audits(doc, pi)
        name = _level_name_from_title(role.get("title", ""))
        if name is None or name not in table.levels:
            continue
        lv = table.levels[name]
        height_mm = (lv.floor_to_floor_m * 1000.0
                     if lv.floor_to_floor_m else None)
        out.append({
            "page_no": pi + 1,
            "title": role.get("title", ""),
            "role": role.get("role"),
            "level_name": name,
            "ffl_mm": lv.elevation_m * 1000.0,
            "height_mm": height_mm,
            "zones": lv.zones,
            "confidence": lv.confidence,
        })
    return out
