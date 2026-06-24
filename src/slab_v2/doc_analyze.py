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
                                FloorInfo, WallType)


_WALL_SCOPE_SUFFIXES = {
    "(U)": "under_only",
    "U": "under_only",
    "(O)": "over_only",
    "O": "over_only",
}


def collect_page_wall_scope_evidence(
    words: list,
    content_rect: fitz.Rect,
    symbols: set[str],
) -> dict[str, dict[str, int]]:
    """Count current/under/over wall labels inside the drawing area."""
    normalized = {str(s).strip().upper(): str(s).strip() for s in symbols}
    by_line: dict[tuple[int, int], list] = {}
    for word in words:
        if len(word) < 7:
            continue
        cx = (float(word[0]) + float(word[2])) / 2.0
        cy = (float(word[1]) + float(word[3])) / 2.0
        if not content_rect.contains(fitz.Point(cx, cy)):
            continue
        by_line.setdefault((int(word[5]), int(word[6])), []).append(word)

    result: dict[str, dict[str, int]] = {}
    for line_words in by_line.values():
        line_words.sort(key=lambda w: (float(w[0]), float(w[1])))
        for idx, word in enumerate(line_words):
            token = str(word[4]).strip().upper()
            if token not in normalized:
                continue
            scope = "current"
            neighbours = (line_words[max(0, idx - 2):idx]
                          + line_words[idx + 1:idx + 3])
            for neighbour in neighbours:
                suffix = str(neighbour[4]).strip().upper()
                if suffix in _WALL_SCOPE_SUFFIXES:
                    scope = _WALL_SCOPE_SUFFIXES[suffix]
                    break
            symbol = normalized[token]
            counts = result.setdefault(symbol, {
                "current": 0, "under_only": 0, "over_only": 0})
            counts[scope] += 1
    return result


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
    """Analyze PDF using v1 floor analyzer + v2 column census prompt."""
    from src.ai_floor_analyzer import analyze_floor_structure
    from src.slab_v2.column_census import analyze_column_census
    from src.slab_v2.wall_census import analyze_wall_census
    from src.slab_v2.pipeline import run_dir

    cfg = cfg or SlabV2Config()
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    page_text_upper = [page.get_text("text").upper() for page in doc]
    page_words = [page.get_text("words") for page in doc]
    from src.slab_v2.pipeline import _content_rect
    page_content_rects = [_content_rect(page) for page in doc]
    doc.close()

    all_pages = list(range(n_pages))
    out_root = run_dir(cfg, pdf_path)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Parallel: floor + column census + wall census (independent Gemini calls)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_floor = ex.submit(
            analyze_floor_structure, pdf_path, all_pages,
            save_dir=str(out_root))
        fut_col = ex.submit(
            analyze_column_census, pdf_path, all_pages, None, str(out_root))
        fut_wall = ex.submit(
            analyze_wall_census, pdf_path, all_pages, None, str(out_root))
        floor_result, _ = fut_floor.result()
        col_result = fut_col.result()
        wall_result = fut_wall.result()

    # ── Fallback: document_intelligence if column census found nothing ──
    doc_intel_used = False
    if not col_result.get("column_types"):
        try:
            from src.document_intelligence import analyze_document_intelligence
            from src.column_detector import (
                build_column_types_from_intelligence,
                build_foundation_types_from_intelligence,
            )
            doc_intel, _, _ = analyze_document_intelligence(
                pdf_path, str(out_root))
            di_col_types = build_column_types_from_intelligence(doc_intel)
            di_fdn_types = build_foundation_types_from_intelligence(doc_intel)
            if di_col_types or di_fdn_types:
                doc_intel_used = True
                schedule_pages = doc_intel.get("schedule_pages", {})
                di_buildings = []
                for b in doc_intel.get("buildings", []):
                    floors = []
                    for f in b.get("floors", []):
                        cs = f.get("column_summary", {})
                        floors.append({
                            "level_name": f.get("level_name", ""),
                            "slab_plan_pages": f.get("slab_plan_pages", []),
                            "columns": cs.get("by_symbol", {}),
                            "total_columns": cs.get("total_columns")
                                or sum((cs.get("by_symbol") or {}).values()),
                        })
                    di_buildings.append({"name": b.get("name", ""),
                                         "floors": floors})
                col_result = {
                    "column_types": di_col_types,
                    "buildings": di_buildings,
                    "detail_pages": schedule_pages.get("detail_pages", []),
                    "orphan_columns": {},
                    "foundation_types": di_fdn_types,
                    "footing_plan_pages": (
                        schedule_pages.get("footing_plan_pages", [])
                        or schedule_pages.get("foundation_schedule_pages", [])),
                    "column_schedule_pages": (
                        schedule_pages.get("column_schedule_pages", [])),
                    "detection_confidence": doc_intel.get(
                        "document_summary", {}).get(
                        "detection_confidence", "medium"),
                }
        except Exception as exc:
            print(f"[doc_analyze] document_intelligence fallback failed: {exc}")

    # ── Merge into DocAnalysis ──────────────────────────────────────
    res = DocAnalysis()
    res.raw = {"floor_result": floor_result, "col_result": col_result,
               "wall_result": wall_result}
    if doc_intel_used:
        res.warnings.append(
            "column_census returned empty — used document_intelligence "
            "fallback for column/foundation types")
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
                "columns": cols, "total_columns": total,
                "slab_plan_pages": cf.get("slab_plan_pages", [])}

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

            # Fallback: if floor_result has no pages but col_result does
            # Skip fallback for roof levels — their pages are typically
            # steel marking plans, not concrete slab plans.
            _is_roof = "roof" in level_id.lower()
            if not pages and col_data and not _is_roof:
                from src.ai_floor_analyzer import _EXCLUDE_TITLE_KEYWORDS
                col_fb_pages = col_data.get("slab_plan_pages") or []
                col_fb_titles = col_data.get("page_titles") or []
                filtered_fb = []
                for cp, ct in zip(col_fb_pages,
                                  col_fb_titles + [""] * len(col_fb_pages)):
                    actual_text = (page_text_upper[cp - 1]
                                   if isinstance(cp, int)
                                   and 1 <= cp <= n_pages else "")
                    title_evidence = f"{ct} {actual_text}"
                    if any(kw in title_evidence.upper()
                           for kw in _EXCLUDE_TITLE_KEYWORDS):
                        res.warnings.append(
                            f"{bname}/{level_id}: col fallback page {cp} "
                            f"excluded (non-outline: '{ct}')")
                        continue
                    filtered_fb.append(cp)
                col_pages = _pages_0based(
                    filtered_fb, n_pages,
                    f"{bname}/{level_id} (col fallback)", res.warnings)
                if col_pages:
                    pages = col_pages
                    res.warnings.append(
                        f"{bname}/{level_id}: using col_result pages "
                        f"{[p + 1 for p in col_pages]} as fallback")

            # A semantic level without a verified concrete slab/outline page
            # is not an exportable concrete storey. Keep its datum text in the
            # height planner, but do not create a model floor from steelwork,
            # elevation, schedule or marking-plan evidence.
            if not pages:
                res.warnings.append(
                    f"{bname}/{level_id}: no valid concrete slab plan page; "
                    "level omitted from concrete model")
                continue

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
    try:
        from src.slab_v2.column_census import infer_column_material
    except Exception:
        infer_column_material = None

    for sym, info in col_result.get("column_types", {}).items():
        sym = str(sym).strip()
        if not sym:
            continue
        if isinstance(info, dict):
            material = str(info.get("material") or "").strip().upper()
            if material not in {"RC", "STEEL", "UNKNOWN"}:
                material = (
                    infer_column_material(sym, info)
                    if infer_column_material else "UNKNOWN"
                )
            res.column_types[sym] = ColumnType(
                symbol=sym,
                width_mm=float(info.get("width_mm") or 0),
                depth_mm=float(info.get("depth_mm") or 0),
                count_total=int(info.get("count_total") or 0),
                material=material)
        else:
            res.column_types[sym] = ColumnType(symbol=sym)
    res.column_census_report = dict(col_result.get("consistency_report") or {})
    res.warnings.extend(col_result.get("warnings") or [])

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

    # ── Wall census merge ─────────────────────────────────────────────
    for sym, info in wall_result.get("wall_types", {}).items():
        sym = str(sym).strip()
        if not sym:
            continue
        if isinstance(info, dict):
            res.wall_types[sym] = WallType(
                symbol=sym,
                thickness_mm=float(info.get("thickness_mm") or 0),
                height_mm=float(info.get("height_mm") or 0),
                material=str(info.get("material") or ""),
                wall_category=str(info.get("wall_category") or "wall"),
                count_total=int(info.get("count_total") or 0))
        else:
            res.wall_types[sym] = WallType(symbol=sym)

    res.wall_schedule_pages = _pages_0based(
        wall_result.get("wall_schedule_pages"), n_pages,
        "wall_schedule_pages", res.warnings)
    res.wall_elevation_pages = _pages_0based(
        wall_result.get("wall_elevation_pages"), n_pages,
        "wall_elevation_pages", res.warnings)

    # Populate FloorInfo.walls from wall_result buildings
    wall_buildings: dict[str, dict[str, dict]] = {}
    for wb in wall_result.get("buildings", []):
        wbname = wb.get("name", "")
        for wf in wb.get("floors", []):
            wlevel = wf.get("level_name", "")
            walls = wf.get("walls", {})
            total = wf.get("total_walls", sum(walls.values()))
            wall_buildings.setdefault(wbname, {})[wlevel] = {
                "walls": walls, "total_walls": total}

    for b in res.buildings:
        for f in b.floors:
            wd = _match_col_data(wall_buildings, b.name, f.level_name,
                                 f.level_id)
            # Gemini wall census is ground truth for which walls belong
            # to each floor.  Page text only validates presence within
            # the census scope — it must not add walls the census omits.
            census_walls = wd.get("walls", {}) if wd else {}
            local_walls: dict[str, int] = {}
            search_symbols = set(res.wall_types) | set(census_walls)
            scope_evidence: dict[str, dict[str, int]] = {}
            for page_index in f.pages:
                page_evidence = collect_page_wall_scope_evidence(
                    page_words[page_index], page_content_rects[page_index],
                    search_symbols)
                for symbol, counts in page_evidence.items():
                    merged = scope_evidence.setdefault(symbol, {
                        "current": 0, "under_only": 0, "over_only": 0})
                    for scope, count in counts.items():
                        merged[scope] += count

            for symbol, counts in scope_evidence.items():
                current_count = int(counts.get("current", 0))
                if not current_count or symbol not in res.wall_types:
                    continue
                local_walls[symbol] = 1 if re.match(
                    r"^(?:LW|W)\d+$", symbol.upper()) else current_count

            f.walls = local_walls
            f.total_walls = sum(local_walls.values())
            scoped_refs = {
                symbol: counts for symbol, counts in scope_evidence.items()
                if counts.get("under_only") or counts.get("over_only")
            }
            if scoped_refs:
                details = ", ".join(
                    f"{symbol}:U={counts.get('under_only', 0)},"
                    f"O={counts.get('over_only', 0)}"
                    for symbol, counts in sorted(scoped_refs.items()))
                res.warnings.append(
                    f"{b.name}/{f.level_id}: reference-only wall labels "
                    f"excluded from current-floor export ({details})")
            if wd and census_walls != local_walls:
                res.warnings.append(
                    f"{b.name}/{f.level_id}: wall census reconciled with "
                    "drawing-zone vertical-scope evidence")

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
                             "total_columns": f.total_columns,
                             "walls": f.walls,
                             "total_walls": f.total_walls}
                            for f in b.floors]}
                for b in res.buildings],
            "column_types": {s: vars(t) for s, t in res.column_types.items()},
            "column_schedule_pages_1based":
                [p + 1 for p in res.column_schedule_pages],
            "columns_per_floor": res.columns_per_floor,
            "column_census_report": res.column_census_report,
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
            "wall_types": {s: vars(t) for s, t in res.wall_types.items()},
            "wall_schedule_pages_1based":
                [p + 1 for p in res.wall_schedule_pages],
            "wall_elevation_pages_1based":
                [p + 1 for p in res.wall_elevation_pages],
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
