"""
Conservative XY floor alignment for stacked structural models.

The goal is to correct small page-origin/layout drift between floor plan pages
without inventing site coordinates. Offsets are only applied when footprint
similarity and translation magnitude are plausible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from shapely.affinity import translate


def _page_floor_map(ai_floor_result: dict | None) -> dict[int, dict]:
    mapping = {}
    for b in (ai_floor_result or {}).get("buildings", []) or []:
        bld = b.get("name") or "(unknown)"
        for floor in b.get("floors", []) or []:
            level = floor.get("level_name") or floor.get("level_id") or ""
            for page_1 in floor.get("slab_plan_pages", []) or []:
                if isinstance(page_1, int) and page_1 > 0:
                    mapping[page_1 - 1] = {"building": bld, "level": level}
    return mapping


def _poly_bounds(poly):
    if poly is None or poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return {
        "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
        "width": maxx - minx, "depth": maxy - miny,
        "area": poly.area,
        "centroid": poly.centroid,
    }


def _primary_slab_by_page(slabs: list) -> dict[int, Any]:
    by_page = defaultdict(list)
    for slab in slabs or []:
        poly = getattr(slab, "real_polygon", None)
        if poly is not None and not poly.is_empty:
            by_page[getattr(slab, "page_index", -1)].append(slab)
    return {
        page: max(items, key=lambda s: getattr(s, "real_polygon", None).area)
        for page, items in by_page.items()
        if page >= 0 and items
    }


def _building_for_page(page_idx: int, page_map: dict, slab=None) -> str:
    if page_idx in page_map:
        return page_map[page_idx].get("building") or "(unknown)"
    label = getattr(slab, "label", "") if slab else ""
    if "—" in label:
        return label.split("—", 1)[0].strip() or "(unknown)"
    return "(unknown)"


def _alignment_confidence(ref_bounds: dict, cur_bounds: dict, dx: float, dy: float) -> tuple[float, str]:
    if not ref_bounds or not cur_bounds:
        return 0.0, "missing_bounds"
    width_ratio = min(ref_bounds["width"], cur_bounds["width"]) / max(ref_bounds["width"], cur_bounds["width"], 1)
    depth_ratio = min(ref_bounds["depth"], cur_bounds["depth"]) / max(ref_bounds["depth"], cur_bounds["depth"], 1)
    area_ratio = min(ref_bounds["area"], cur_bounds["area"]) / max(ref_bounds["area"], cur_bounds["area"], 1)
    diag = max((cur_bounds["width"] ** 2 + cur_bounds["depth"] ** 2) ** 0.5, 1)
    offset_ratio = (dx ** 2 + dy ** 2) ** 0.5 / diag
    score = 0.20 + 0.30 * width_ratio + 0.25 * depth_ratio + 0.20 * area_ratio
    if offset_ratio <= 0.12:
        score += 0.15
    elif offset_ratio > 0.25:
        score -= 0.25
    reason = (
        f"width_ratio={width_ratio:.2f}, depth_ratio={depth_ratio:.2f}, "
        f"area_ratio={area_ratio:.2f}, offset_ratio={offset_ratio:.2f}"
    )
    return max(0.0, min(score, 0.98)), reason


def _translate_element(elem, dx: float, dy: float) -> None:
    poly = getattr(elem, "real_polygon", None)
    if poly is not None and not poly.is_empty:
        elem.real_polygon = translate(poly, xoff=dx, yoff=dy)


def align_floors(slabs: list, columns: list | None = None, foundations: list | None = None,
                 ai_floor_result: dict | None = None,
                 min_confidence: float = 0.72) -> tuple[list[dict], dict[int, tuple[float, float]]]:
    """Compute/apply conservative XY offsets by building/page."""
    columns = columns or []
    foundations = foundations or []
    page_map = _page_floor_map(ai_floor_result)
    primary = _primary_slab_by_page(slabs)
    by_building = defaultdict(list)
    for page_idx, slab in primary.items():
        by_building[_building_for_page(page_idx, page_map, slab)].append((page_idx, slab))

    rows = []
    offsets: dict[int, tuple[float, float]] = {}
    for building, page_slabs in sorted(by_building.items()):
        if not page_slabs:
            continue
        ref_page, ref_slab = max(page_slabs, key=lambda ps: getattr(ps[1], "area_m2", 0.0))
        ref_bounds = _poly_bounds(getattr(ref_slab, "real_polygon", None))
        for page_idx, slab in sorted(page_slabs):
            cur_bounds = _poly_bounds(getattr(slab, "real_polygon", None))
            if page_idx == ref_page:
                rows.append({
                    "Building": building,
                    "Page": page_idx + 1,
                    "Reference": f"P{ref_page + 1}",
                    "dx_mm": 0.0,
                    "dy_mm": 0.0,
                    "Confidence": 1.0,
                    "Applied": False,
                    "Warning": "reference floor",
                })
                continue
            if not ref_bounds or not cur_bounds:
                rows.append({
                    "Building": building, "Page": page_idx + 1, "Reference": f"P{ref_page + 1}",
                    "dx_mm": 0.0, "dy_mm": 0.0, "Confidence": 0.0,
                    "Applied": False, "Warning": "missing footprint bounds",
                })
                continue
            dx = ref_bounds["minx"] - cur_bounds["minx"]
            dy = ref_bounds["miny"] - cur_bounds["miny"]
            confidence, reason = _alignment_confidence(ref_bounds, cur_bounds, dx, dy)
            max_dim = max(cur_bounds["width"], cur_bounds["depth"], 1)
            applied = confidence >= min_confidence and (dx ** 2 + dy ** 2) ** 0.5 <= max_dim * 0.18
            warning = "" if applied else f"not applied: {reason}"
            if applied:
                offsets[page_idx] = (dx, dy)
                for s in slabs:
                    if getattr(s, "page_index", None) == page_idx:
                        _translate_element(s, dx, dy)
                for c in columns:
                    if getattr(c, "page_index", None) == page_idx:
                        _translate_element(c, dx, dy)
                for f in foundations:
                    if getattr(f, "page_index", None) == page_idx:
                        _translate_element(f, dx, dy)
            rows.append({
                "Building": building,
                "Page": page_idx + 1,
                "Reference": f"P{ref_page + 1}",
                "dx_mm": round(dx, 1),
                "dy_mm": round(dy, 1),
                "Confidence": round(confidence, 2),
                "Applied": bool(applied),
                "Warning": warning,
            })
    return rows, offsets
