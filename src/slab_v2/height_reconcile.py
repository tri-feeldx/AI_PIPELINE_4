"""
Multi-source FFL / storey-height reconciliation.

Collects FFL values from three independent sources, cross-validates,
and produces a single authoritative height table for export.

Sources (priority: deterministic > AI > default):
  D — per-page FFL text regex  (most precise, from the plan page itself)
  C — elevation drawing label detection  (deterministic, cross-page)
  A — Gemini doc_analyze  (AI, can hallucinate)
  fallback — config.default_storey_height_mm
"""

from __future__ import annotations

import re
from typing import Optional

import fitz

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import (
    DocAnalysis, FloorHeight, HeightReconciliation,
)
from src.pdf_processor import (
    extract_ffl_values,
    extract_text_blocks,
    extract_storey_heights_from_elevation,
)


def _level_num(level_id: str) -> Optional[int]:
    """Extract numeric floor number from level_id like 'level_03'."""
    m = re.search(r"(\d+)", level_id)
    return int(m.group(1)) if m else None


def reconcile_heights(
    pdf_path: str,
    doc_analysis: DocAnalysis,
    cfg: SlabV2Config,
) -> HeightReconciliation:
    """
    Reconcile FFL and storey heights from all available sources.

    Returns HeightReconciliation with one FloorHeight per non-roof floor,
    ordered by ascending FFL.
    """
    warnings: list[str] = []
    debug_log: list[str] = []
    methods_used: list[str] = []

    doc = fitz.open(pdf_path)

    # ── Source C: elevation drawing detection ────────────────────────────
    elev_heights: dict[int, float] = {}
    try:
        elev_heights = extract_storey_heights_from_elevation(doc)
        if elev_heights:
            methods_used.append("elevation")
            debug_log.append(
                f"Source C (elevation): {elev_heights}")
    except Exception as e:
        debug_log.append(f"Source C failed: {e}")

    # ── Source D: per-page FFL text regex ────────────────────────────────
    page_ffls: dict[int, list[float]] = {}  # page_index -> [ffl_m values]
    for b in doc_analysis.buildings:
        for f in b.floors:
            for pi in f.pages:
                try:
                    blocks = extract_text_blocks(doc[pi])
                    found = extract_ffl_values(blocks)
                    if found:
                        page_ffls[pi] = [v["ffl_m"] for v in found]
                except Exception:
                    pass
    if page_ffls:
        methods_used.append("page_text")
        debug_log.append(
            f"Source D (page text): {page_ffls}")

    doc.close()

    # ── Build floor list from doc_analysis ───────────────────────────────
    floors: list[FloorHeight] = []

    for b in doc_analysis.buildings:
        prev_ffl: Optional[float] = None

        for f in b.floors:

            sources: dict[str, float] = {}
            lvl_num = _level_num(f.level_id)

            # Source A: Gemini FFL
            if f.ffl_m is not None:
                sources["gemini"] = f.ffl_m

            # Source B: Gemini storey_height_mm (from sections)
            if f.storey_height_mm and f.storey_height_mm > 0:
                sources["gemini_height"] = f.storey_height_mm

            # Source D: per-page FFL text
            for pi in f.pages:
                if pi in page_ffls and page_ffls[pi]:
                    sources["page_text"] = page_ffls[pi][0]
                    break

            # Source C: elevation-derived FFL (cumulative from heights)
            if elev_heights and lvl_num is not None:
                cum = 0.0
                for lv in range(1, lvl_num):
                    if lv in elev_heights:
                        cum += elev_heights[lv]
                    else:
                        break
                else:
                    if lvl_num == 1:
                        cum = 0.0
                    sources["elevation"] = round(cum, 3)

            # ── Pick best FFL ────────────────────────────────────────────
            ffl_m: Optional[float] = None
            confidence = "low"

            if "page_text" in sources:
                pt = sources["page_text"]
                ge = sources.get("gemini")
                # validate page_text against Gemini: if Gemini has a value
                # and page_text disagrees by >50%, page_text is likely a
                # spurious regex match (slab thickness, grid ref, etc.)
                if ge is not None and ge != 0 and abs(pt - ge) / max(abs(ge), 0.001) > 0.50:
                    debug_log.append(
                        f"{f.level_id}: page_text FFL={pt:.3f}m rejected "
                        f"(Gemini={ge:.3f}m, >50% off)")
                    del sources["page_text"]
                else:
                    ffl_m = pt
                    confidence = "high"
            elif "elevation" in sources and "gemini" in sources:
                ge = sources["gemini"]
                el = sources["elevation"]
                if ge != 0 and abs(ge - el) / max(abs(ge), 0.001) < 0.05:
                    ffl_m = ge
                    confidence = "high"
                else:
                    ffl_m = el
                    confidence = "medium"
                    if abs(ge - el) > 0.5:
                        warnings.append(
                            f"{f.level_id}: Gemini FFL={ge:.3f}m vs "
                            f"elevation={el:.3f}m — using elevation")
            elif "elevation" in sources:
                ffl_m = sources["elevation"]
                confidence = "medium"
            elif "gemini" in sources:
                ffl_m = sources["gemini"]
                confidence = "medium"

            # Fallback
            if ffl_m is None:
                default_h = cfg.default_storey_height_mm / 1000.0
                ffl_m = (prev_ffl or 0.0) + default_h
                sources["default"] = ffl_m
                warnings.append(
                    f"{f.level_id}: no FFL found — defaulting to "
                    f"{ffl_m:.3f}m")

            prev_ffl = ffl_m

            floors.append(FloorHeight(
                level_id=f.level_id,
                building=b.name,
                ffl_m=ffl_m,
                sources=sources,
                confidence=confidence,
            ))

    # ── De-duplicate FFLs per building ────────────────────────────────────
    # If multiple floors got the same FFL (bad page_text), revert to Gemini
    by_bld: dict[str, list[FloorHeight]] = {}
    for fh in floors:
        by_bld.setdefault(fh.building, []).append(fh)
    for bname, bfloors in by_bld.items():
        seen: dict[float, int] = {}
        has_dup = False
        for fh in bfloors:
            key = round(fh.ffl_m, 3)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                has_dup = True
        if has_dup:
            warnings.append(
                f"{bname}: duplicate FFLs detected after reconciliation "
                f"— reverting to Gemini values")
            for fh in bfloors:
                for b in doc_analysis.buildings:
                    if b.name != bname:
                        continue
                    for f in b.floors:
                        if f.level_id == fh.level_id and f.ffl_m is not None:
                            fh.ffl_m = f.ffl_m
                            fh.sources = {"gemini": f.ffl_m}
                            fh.confidence = "medium"
                            break

    # ── Compute storey heights from reconciled FFLs ──────────────────────
    # Build lookup for Gemini storey_height_mm per floor
    gemini_heights: dict[str, float] = {}
    for b in doc_analysis.buildings:
        for f in b.floors:
            if f.storey_height_mm and f.storey_height_mm > 0:
                gemini_heights[f"{b.name}/{f.level_id}"] = f.storey_height_mm

    for i, fh in enumerate(floors):
        if i + 1 < len(floors) and floors[i + 1].building == fh.building:
            h = (floors[i + 1].ffl_m - fh.ffl_m) * 1000.0
            fh.storey_height_mm = round(h, 0)
        elif i > 0 and floors[i - 1].building == fh.building:
            fh.storey_height_mm = floors[i - 1].storey_height_mm
        else:
            # try Gemini storey_height_mm before default
            gkey = f"{fh.building}/{fh.level_id}"
            if gkey in gemini_heights:
                fh.storey_height_mm = gemini_heights[gkey]
                methods_used.append("gemini_height")
            else:
                fh.storey_height_mm = cfg.default_storey_height_mm

        # Override with elevation source if available
        lvl_num = _level_num(fh.level_id)
        if lvl_num is not None and lvl_num in elev_heights:
            elev_h = elev_heights[lvl_num] * 1000.0
            if abs(elev_h - fh.storey_height_mm) > 100:
                debug_log.append(
                    f"{fh.level_id}: FFL-gap height={fh.storey_height_mm:.0f}mm "
                    f"vs elevation={elev_h:.0f}mm")
            fh.storey_height_mm = round(elev_h, 0)

    # ── Validation ───────────────────────────────────────────────────────
    for fh in floors:
        h = fh.storey_height_mm
        if h < 2000:
            warnings.append(
                f"{fh.level_id}: storey height {h:.0f}mm is suspiciously "
                f"low (<2m) — check scale or FFL")
        elif h > 6000:
            warnings.append(
                f"{fh.level_id}: storey height {h:.0f}mm is unusually "
                f"high (>6m)")

    # Monotonic FFL check (per building)
    buildings_seen: dict[str, list[float]] = {}
    for fh in floors:
        buildings_seen.setdefault(fh.building, []).append(fh.ffl_m)
    for bname, ffls in buildings_seen.items():
        for i in range(len(ffls) - 1):
            if ffls[i + 1] <= ffls[i]:
                warnings.append(
                    f"{bname}: FFL not increasing — "
                    f"{ffls[i]:.3f}m >= {ffls[i + 1]:.3f}m")

    if not methods_used:
        methods_used.append("default_only")

    return HeightReconciliation(
        floors=floors,
        method="+".join(methods_used),
        warnings=warnings,
        debug_log=debug_log,
    )
