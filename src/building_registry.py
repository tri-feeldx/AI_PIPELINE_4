"""
Building-level registry for final structural model review/export.

The registry keeps building positions in native drawing coordinates after scale
conversion. It never offsets buildings for presentation.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from shapely.geometry import MultiPolygon
from shapely.ops import unary_union


def _clean_building_name(name: str | None) -> str:
    value = (name or "").strip()
    return value if value and value.lower() not in {"(unknown)", "unknown"} else "(unknown)"


def _building_from_slab_label(label: str | None) -> str:
    label = (label or "").strip()
    parts = re.split(r"\s+[—–-]\s+", label, maxsplit=1)
    if len(parts) == 2:
        return _clean_building_name(parts[0])
    if "-" in label:
        first = label.split("-", 1)[0].strip()
        if first.lower().startswith("building"):
            return _clean_building_name(first)
    return "(unknown)"


def _page_to_floor_map(ai_floor_result: dict | None) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    for bld in (ai_floor_result or {}).get("buildings", []) or []:
        bld_name = _clean_building_name(bld.get("name"))
        for floor in bld.get("floors", []) or []:
            for page_1 in floor.get("slab_plan_pages", []) or []:
                if isinstance(page_1, int) and page_1 > 0:
                    mapping[page_1 - 1] = {
                        "building": bld_name,
                        "level": floor.get("level_name") or floor.get("level_id") or "",
                    }
    return mapping


def _geom_area_m2(poly) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    return float(poly.area) / 1_000_000.0


def _geom_bounds(poly) -> dict | None:
    if poly is None or poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return {
        "min_x_mm": float(minx),
        "min_y_mm": float(miny),
        "max_x_mm": float(maxx),
        "max_y_mm": float(maxy),
        "width_mm": float(maxx - minx),
        "depth_mm": float(maxy - miny),
    }


def _geom_parts(poly):
    if isinstance(poly, MultiPolygon):
        return list(poly.geoms)
    return [poly]


def build_building_registry(slabs: list, columns: list | None = None,
                            foundations: list | None = None,
                            ai_floor_result: dict | None = None) -> dict[str, Any]:
    """Build footprint/bbox/count metadata for Streamlit review and Ruby export."""
    columns = columns or []
    foundations = foundations or []
    page_floor = _page_to_floor_map(ai_floor_result)

    registry = {
        "buildings": {},
        "warnings": [],
        "position_mode": "native_coordinates",
    }

    def ensure(name: str) -> dict:
        name = _clean_building_name(name)
        if name not in registry["buildings"]:
            registry["buildings"][name] = {
                "name": name,
                "levels": defaultdict(lambda: {"pages": set(), "slabs": 0, "columns": 0, "foundations": 0}),
                "slab_pages": set(),
                "column_pages": set(),
                "foundation_pages": set(),
                "slab_count": 0,
                "column_count": 0,
                "foundation_count": 0,
                "footprint_polygon": None,
                "footprint_parts": [],
                "bbox_mm": None,
                "centroid_mm": None,
                "area_m2": 0.0,
                "confidence": "medium",
                "warnings": [],
            }
        return registry["buildings"][name]

    slab_polys_by_building: dict[str, list] = defaultdict(list)
    for slab in slabs or []:
        page_info = page_floor.get(getattr(slab, "page_index", -1), {})
        bld = page_info.get("building") or _building_from_slab_label(getattr(slab, "label", ""))
        level = page_info.get("level") or getattr(slab, "label", "") or ""
        rec = ensure(bld)
        rec["slab_count"] += 1
        rec["slab_pages"].add(getattr(slab, "page_index", -1) + 1)
        rec["levels"][level]["slabs"] += 1
        rec["levels"][level]["pages"].add(getattr(slab, "page_index", -1) + 1)
        poly = getattr(slab, "real_polygon", None)
        if poly is not None and not poly.is_empty:
            slab_polys_by_building[rec["name"]].append(poly)

    for col in columns:
        page_info = page_floor.get(getattr(col, "page_index", -1), {})
        bld = getattr(col, "building", "") or page_info.get("building") or "(unknown)"
        level = getattr(col, "level", "") or page_info.get("level") or ""
        rec = ensure(bld)
        rec["column_count"] += 1
        rec["column_pages"].add(getattr(col, "page_index", -1) + 1)
        rec["levels"][level]["columns"] += 1
        rec["levels"][level]["pages"].add(getattr(col, "page_index", -1) + 1)

    # Assign foundations by nearest slab footprint when they lack explicit building metadata.
    building_footprints = {}
    for bld, polys in slab_polys_by_building.items():
        if polys:
            building_footprints[bld] = unary_union(polys)

    for fdn in foundations:
        bld = getattr(fdn, "building", "") or ""
        if not bld:
            poly = getattr(fdn, "real_polygon", None)
            if poly is not None and not poly.is_empty and building_footprints:
                centroid = poly.centroid
                bld = min(
                    building_footprints,
                    key=lambda name: building_footprints[name].distance(centroid),
                )
        rec = ensure(bld or "(unknown)")
        rec["foundation_count"] += 1
        rec["foundation_pages"].add(getattr(fdn, "page_index", -1) + 1)
        rec["levels"]["foundation"]["foundations"] += 1
        rec["levels"]["foundation"]["pages"].add(getattr(fdn, "page_index", -1) + 1)

    for name, rec in registry["buildings"].items():
        polys = slab_polys_by_building.get(name, [])
        if polys:
            footprint = unary_union(polys)
            rec["footprint_polygon"] = footprint
            rec["footprint_parts"] = _geom_parts(footprint)
            rec["bbox_mm"] = _geom_bounds(footprint)
            c = footprint.centroid
            rec["centroid_mm"] = {"x_mm": float(c.x), "y_mm": float(c.y)}
            rec["area_m2"] = _geom_area_m2(footprint)
        else:
            rec["confidence"] = "low"
            rec["warnings"].append("No slab footprint polygon found for this building.")

    names = list(registry["buildings"])
    for i, a in enumerate(names):
        pa = registry["buildings"][a]["footprint_polygon"]
        if pa is None or pa.is_empty:
            continue
        for b in names[i + 1:]:
            pb = registry["buildings"][b]["footprint_polygon"]
            if pb is None or pb.is_empty:
                continue
            inter = pa.intersection(pb).area
            smaller = min(pa.area, pb.area)
            if smaller > 0 and inter / smaller > 0.25:
                msg = (
                    f"Building footprints overlap: {a} / {b}. "
                    "Likely separate plan sheets or missing site coordinates."
                )
                registry["warnings"].append(msg)
                registry["buildings"][a]["warnings"].append(msg)
                registry["buildings"][b]["warnings"].append(msg)

    for rec in registry["buildings"].values():
        rec["levels"] = {
            k: {
                "pages": sorted(v["pages"]),
                "slabs": v["slabs"],
                "columns": v["columns"],
                "foundations": v["foundations"],
            }
            for k, v in rec["levels"].items()
        }
        rec["slab_pages"] = sorted(p for p in rec["slab_pages"] if p > 0)
        rec["column_pages"] = sorted(p for p in rec["column_pages"] if p > 0)
        rec["foundation_pages"] = sorted(p for p in rec["foundation_pages"] if p > 0)

    if not registry["buildings"]:
        registry["warnings"].append("No building geometry available.")
    return registry


def building_registry_rows(registry: dict) -> list[dict]:
    rows = []
    for rec in (registry or {}).get("buildings", {}).values():
        bbox = rec.get("bbox_mm") or {}
        centroid = rec.get("centroid_mm") or {}
        rows.append({
            "Building": rec.get("name"),
            "Floors": sum(1 for lvl in rec.get("levels", {}) if str(lvl).lower() != "foundation"),
            "Slabs": rec.get("slab_count", 0),
            "Columns": rec.get("column_count", 0),
            "Foundations": rec.get("foundation_count", 0),
            "Area (m2)": round(rec.get("area_m2", 0.0), 2),
            "Centroid X (mm)": round(centroid.get("x_mm", 0.0), 1) if centroid else None,
            "Centroid Y (mm)": round(centroid.get("y_mm", 0.0), 1) if centroid else None,
            "Width (mm)": round(bbox.get("width_mm", 0.0), 1) if bbox else None,
            "Depth (mm)": round(bbox.get("depth_mm", 0.0), 1) if bbox else None,
            "Source Pages": ", ".join(map(str, rec.get("slab_pages", []))),
            "Confidence": rec.get("confidence", "low"),
            "Warnings": " | ".join(rec.get("warnings", [])),
        })
    return rows
