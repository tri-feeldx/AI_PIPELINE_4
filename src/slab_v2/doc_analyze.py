"""
Whole-document analysis — delegates to v1's proven Gemini prompts.

Uses two v1 modules that have been validated on production PDFs:
  - ai_floor_analyzer.py: floor detection + post-processing
    (filters STEEL MARKING, validates page titles, page_level_map)
  - column_analyzer.py: column/foundation census with dedicated prompt

Results are converted to the slab_v2 DocAnalysis model so the rest of
the v2 pipeline (height_reconcile, export_ruby, app_v2) works unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import (BuildingInfo, ColumnType, DocAnalysis,
                                FloorInfo)


def _pages_0based(raw, n_pages: int, what: str,
                  warnings: list[str]) -> list[int]:
    out = []
    for p in raw or []:
        if isinstance(p, int) and 1 <= p <= n_pages:
            out.append(p - 1)
        else:
            warnings.append(f"{what}: page {p!r} out of range — dropped")
    return sorted(set(out))


def analyze_document(pdf_path: str,
                     cfg: SlabV2Config | None = None) -> DocAnalysis:
    """Analyze PDF using v1's proven prompts, return v2 DocAnalysis model."""
    from src.ai_floor_analyzer import analyze_floor_structure
    from src.column_analyzer import analyze_columns_and_foundations
    from src.slab_v2.pipeline import run_dir

    cfg = cfg or SlabV2Config()
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()

    all_pages = list(range(n_pages))
    out_root = run_dir(cfg, pdf_path)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Call 1: v1 floor analyzer (includes post-processing) ────────
    floor_result, _ = analyze_floor_structure(
        pdf_path, all_pages, save_dir=str(out_root))

    # ── Call 2: v1 column/foundation census ─────────────────────────
    col_result = analyze_columns_and_foundations(
        pdf_path, all_pages, floor_result)

    # ── Merge into DocAnalysis ──────────────────────────────────────
    res = DocAnalysis()
    res.raw = {"floor_result": floor_result, "col_result": col_result}
    res.confidence = floor_result.get("detection_confidence", "")
    res.notes = floor_result.get("notes", "")

    # Build column lookup from col_result per building/floor
    col_buildings: dict[str, dict[str, dict]] = {}
    for cb in col_result.get("buildings", []):
        bname = cb.get("name", "")
        for cf in cb.get("floors", []):
            level_name = cf.get("level_name", "")
            cols = cf.get("columns", {})
            total = cf.get("total_columns", sum(cols.values()))
            col_buildings.setdefault(bname, {})[level_name] = {
                "columns": cols, "total_columns": total}

    # Build buildings/floors from floor_result
    seen_pages: dict[int, str] = {}
    for b in floor_result.get("buildings", []):
        bname = b.get("name") or "(unnamed)"
        bi = BuildingInfo(name=bname)

        for f in b.get("floors", []):
            level_id = f.get("level_id", "")
            level_name = f.get("level_name", "")
            pages = _pages_0based(
                f.get("slab_plan_pages"), n_pages,
                f"{bname}/{level_id}", res.warnings)

            # Match column data from col_result
            col_data = _match_col_data(col_buildings, bname, level_name,
                                       level_id)
            cols_dict = col_data.get("columns", {})
            total_cols = col_data.get("total_columns", 0)

            fi = FloorInfo(
                level_name=level_name,
                level_id=level_id,
                ffl_m=f.get("ffl_m"),
                pages=pages,
                titles=f.get("page_titles") or [],
                storey_height_mm=0.0,
                columns=cols_dict,
                total_columns=total_cols)

            for p in fi.pages:
                key = f"{bname}/{level_id}"
                if p in seen_pages and seen_pages[p] != key:
                    res.warnings.append(
                        f"page {p + 1} assigned to both {seen_pages[p]} "
                        f"and {key}")
                seen_pages[p] = key
            bi.floors.append(fi)

        ffls = [f.ffl_m for f in bi.floors if f.ffl_m is not None]
        if any(b2 < a2 for a2, b2 in zip(ffls, ffls[1:])) and \
                sorted(ffls) != ffls:
            res.warnings.append(
                f"{bname}: floor FFLs not monotonically increasing "
                f"({ffls}) — check doc_analysis.json")
        res.buildings.append(bi)

    # Column types from col_result
    for sym, info in col_result.get("column_types", {}).items():
        sym = str(sym).strip()
        if not sym:
            continue
        if isinstance(info, dict):
            res.column_types[sym] = ColumnType(
                symbol=sym,
                width_mm=float(info.get("width_mm") or 0),
                depth_mm=float(info.get("depth_mm") or 0),
                count_total=int(info.get("count_total") or 0))
        else:
            res.column_types[sym] = ColumnType(symbol=sym)

    # Column schedule pages
    res.column_schedule_pages = _pages_0based(
        col_result.get("column_schedule_pages"), n_pages,
        "column_schedule_pages", res.warnings)

    # Detail pages
    for fld in ("stair_detail_pages", "lift_detail_pages",
                "foundation_detail_pages"):
        raw_pages = col_result.get(fld) or floor_result.get(fld) or []
        setattr(res, fld, _pages_0based(raw_pages, n_pages, fld,
                                        res.warnings))
    res.detail_pages = _pages_0based(
        col_result.get("detail_pages"), n_pages, "detail_pages",
        res.warnings)

    # columns_per_floor
    for b in res.buildings:
        for f in b.floors:
            if f.columns:
                res.columns_per_floor.append({
                    "building": b.name,
                    "level_id": f.level_id,
                    "counts": dict(f.columns),
                })

    # Foundation types and footing pages
    raw_fdn = col_result.get("foundation_types") or {}
    if isinstance(raw_fdn, dict):
        res.foundation_types = raw_fdn
    res.footing_plan_pages = _pages_0based(
        col_result.get("footing_plan_pages"), n_pages,
        "footing_plan_pages", res.warnings)

    # Orphan columns
    raw_orphan = col_result.get("orphan_columns") or {}
    if isinstance(raw_orphan, dict):
        res.orphan_columns = {str(k): int(v) for k, v in raw_orphan.items()
                              if isinstance(v, (int, float))}

    # Save merged JSON
    with open(out_root / "doc_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({
            "buildings": [
                {"name": b.name,
                 "floors": [{"level_name": f.level_name,
                             "level_id": f.level_id,
                             "ffl_m": f.ffl_m,
                             "storey_height_mm": f.storey_height_mm,
                             "pages_1based": [p + 1 for p in f.pages],
                             "titles": f.titles,
                             "columns": f.columns,
                             "total_columns": f.total_columns}
                            for f in b.floors]}
                for b in res.buildings],
            "column_types": {s: vars(t) for s, t in res.column_types.items()},
            "column_schedule_pages_1based":
                [p + 1 for p in res.column_schedule_pages],
            "columns_per_floor": res.columns_per_floor,
            "stair_detail_pages_1based":
                [p + 1 for p in res.stair_detail_pages],
            "lift_detail_pages_1based":
                [p + 1 for p in res.lift_detail_pages],
            "foundation_detail_pages_1based":
                [p + 1 for p in res.foundation_detail_pages],
            "foundation_types": res.foundation_types,
            "footing_plan_pages_1based":
                [p + 1 for p in res.footing_plan_pages],
            "orphan_columns": res.orphan_columns,
            "confidence": res.confidence,
            "notes": res.notes,
            "warnings": res.warnings,
        }, fh, indent=2, ensure_ascii=False)
    return res


def _match_col_data(col_buildings: dict, bname: str,
                    level_name: str, level_id: str) -> dict:
    """Find column data matching a floor, with fuzzy building name matching."""
    # Exact building match
    if bname in col_buildings:
        bdata = col_buildings[bname]
        if level_name in bdata:
            return bdata[level_name]
        for k, v in bdata.items():
            if _level_match(k, level_id):
                return v

    # Fuzzy building match (v1 column_analyzer may use different name)
    for cb_name, bdata in col_buildings.items():
        if _fuzzy_bname(bname, cb_name):
            if level_name in bdata:
                return bdata[level_name]
            for k, v in bdata.items():
                if _level_match(k, level_id):
                    return v

    # Single building fallback
    if len(col_buildings) == 1:
        bdata = next(iter(col_buildings.values()))
        if level_name in bdata:
            return bdata[level_name]
        for k, v in bdata.items():
            if _level_match(k, level_id):
                return v

    return {}


def _level_match(level_name: str, level_id: str) -> bool:
    """Check if a level_name like 'Level 04' matches level_id 'level_04'."""
    m = re.search(r"(\d+)", level_name)
    if m:
        n = re.search(r"(\d+)", level_id)
        if n and m.group(1).lstrip("0") == n.group(1).lstrip("0"):
            return True
    return level_name.lower().replace(" ", "_") == level_id


def _fuzzy_bname(a: str, b: str) -> bool:
    """Fuzzy match two building names (case-insensitive, ignore punctuation)."""
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    na, nb = norm(a), norm(b)
    return na in nb or nb in na
