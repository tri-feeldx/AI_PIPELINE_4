"""
Building detection audit outputs for structural model export.

This module does not solve site placement. It records the current semantic
building mapping, geometry footprints, and element assignment evidence so bad
multi-building exports are visible before SketchUp import.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import fitz
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon

from src.building_registry import build_building_registry
from src.visualizer import save_building_footprints


COLORS = [
    "#00C853", "#2962FF", "#FF6D00", "#D500F9",
    "#00B8D4", "#FFD600", "#C51162", "#64DD17",
]


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    try:
        from shapely.geometry import mapping
        if hasattr(value, "geom_type"):
            return mapping(value)
    except Exception:
        pass
    return str(value)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return str(path)


def _parse_status(document_intelligence: dict | None) -> str:
    intel = document_intelligence or {}
    return (
        intel.get("_parse_status")
        or (intel.get("_metadata") or {}).get("parse_status")
        or "unknown"
    )


def _page_map(ai_floor_result: dict | None) -> dict[int, dict]:
    mapping = {}
    for bld in (ai_floor_result or {}).get("buildings", []) or []:
        bld_name = bld.get("name") or "(unknown)"
        for floor in bld.get("floors", []) or []:
            level = floor.get("level_name") or floor.get("level_id") or ""
            for page_1 in floor.get("slab_plan_pages", []) or []:
                if isinstance(page_1, int) and page_1 > 0:
                    mapping[page_1 - 1] = {
                        "building": bld_name,
                        "level": level,
                        "source": "ai_floor_result",
                    }
    return mapping


def _expected_buildings(document_intelligence: dict | None, ai_floor_result: dict | None) -> list[str]:
    names = []
    for source in (document_intelligence or {}, ai_floor_result or {}):
        for bld in source.get("buildings", []) or []:
            name = (bld.get("name") or "").strip()
            if name and name.lower() not in {"unknown", "(unknown)"} and name not in names:
                names.append(name)
    return names


def _bounds(poly) -> dict | None:
    if poly is None or poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return {
        "min_x": float(minx),
        "min_y": float(miny),
        "max_x": float(maxx),
        "max_y": float(maxy),
        "width": float(maxx - minx),
        "depth": float(maxy - miny),
    }


def _poly_parts(poly) -> list:
    if poly is None or poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        return list(poly.geoms)
    return [poly]


def _slab_label_building(label: str | None) -> str:
    value = (label or "").strip()
    if "—" in value:
        return value.split("—", 1)[0].strip() or "(unknown)"
    if " - " in value:
        first = value.split(" - ", 1)[0].strip()
        if first:
            return first
    return "(unknown)"


def _slab_rows(slabs: list, page_mapping: dict[int, dict]) -> list[dict]:
    rows = []
    for idx, slab in enumerate(slabs or []):
        page_idx = getattr(slab, "page_index", -1)
        page_info = page_mapping.get(page_idx, {})
        real_poly = getattr(slab, "real_polygon", None)
        page_poly = getattr(slab, "polygon", None)
        building = page_info.get("building") or _slab_label_building(getattr(slab, "label", ""))
        rows.append({
            "id": getattr(slab, "id", idx),
            "label": getattr(slab, "label", ""),
            "page": page_idx + 1 if page_idx >= 0 else None,
            "building_candidate": building,
            "level_candidate": page_info.get("level") or getattr(slab, "label", ""),
            "area_m2": getattr(slab, "area_m2", 0.0),
            "source": getattr(slab, "source", ""),
            "real_bbox_mm": _bounds(real_poly),
            "page_bbox_pt": _bounds(page_poly),
            "has_real_polygon": bool(real_poly is not None and not real_poly.is_empty),
            "has_page_polygon": bool(page_poly is not None and not page_poly.is_empty),
        })
    return rows


def _registry_json(registry: dict) -> dict:
    buildings = {}
    for name, rec in (registry or {}).get("buildings", {}).items():
        buildings[name] = {
            "name": name,
            "levels": rec.get("levels", {}),
            "slab_pages": rec.get("slab_pages", []),
            "column_pages": rec.get("column_pages", []),
            "foundation_pages": rec.get("foundation_pages", []),
            "slab_count": rec.get("slab_count", 0),
            "column_count": rec.get("column_count", 0),
            "foundation_count": rec.get("foundation_count", 0),
            "bbox_mm": rec.get("bbox_mm"),
            "centroid_mm": rec.get("centroid_mm"),
            "area_m2": rec.get("area_m2", 0.0),
            "confidence": rec.get("confidence", "low"),
            "warnings": rec.get("warnings", []),
        }
    return {
        "position_mode": (registry or {}).get("position_mode", "native_coordinates"),
        "warnings": (registry or {}).get("warnings", []),
        "buildings": buildings,
    }


def _building_footprints(registry: dict) -> dict[str, Any]:
    return {
        name: rec.get("footprint_polygon")
        for name, rec in (registry or {}).get("buildings", {}).items()
        if rec.get("footprint_polygon") is not None and not rec.get("footprint_polygon").is_empty
    }


def _element_polygon(elem):
    real = getattr(elem, "real_polygon", None)
    if real is not None and not real.is_empty:
        return real
    poly = getattr(elem, "polygon", None)
    if poly is not None and not poly.is_empty:
        return poly
    return None


def _assign_elements(elements: list, kind: str, page_mapping: dict[int, dict],
                     footprints: dict[str, Any]) -> list[dict]:
    rows = []
    for idx, elem in enumerate(elements or []):
        page_idx = getattr(elem, "page_index", -1)
        page_info = page_mapping.get(page_idx, {})
        poly = _element_polygon(elem)
        explicit = getattr(elem, "building", "") or ""
        assigned = explicit or page_info.get("building") or "(unknown)"
        reason = "explicit_element_building" if explicit else "page_mapping" if page_info else "unknown"
        distance_mm = None
        inside = False
        if poly is not None and not poly.is_empty and footprints:
            centroid = poly.centroid
            best_name = None
            best_dist = None
            for name, fp in footprints.items():
                dist = float(fp.distance(centroid))
                if best_dist is None or dist < best_dist:
                    best_name = name
                    best_dist = dist
                if fp.contains(centroid) or fp.touches(centroid):
                    if not explicit:
                        assigned = name
                        reason = "inside_building_footprint"
                    inside = True
                    best_name = name
                    best_dist = 0.0
                    break
            if not explicit and assigned == "(unknown)" and best_name:
                assigned = best_name
                reason = "nearest_building_footprint"
            distance_mm = best_dist
        status = "assigned"
        warnings = []
        if assigned == "(unknown)":
            status = "orphan"
            warnings.append("No building mapping or footprint assignment.")
        elif distance_mm is not None and not inside and distance_mm > 2500:
            status = "review"
            warnings.append("Element centroid is far from assigned/nearest building footprint.")
        rows.append({
            "kind": kind,
            "id": getattr(elem, "id", idx),
            "symbol": getattr(elem, "symbol", getattr(elem, "label", "")),
            "page": page_idx + 1 if page_idx >= 0 else None,
            "assigned_building": assigned,
            "assignment_reason": reason,
            "inside_footprint": inside,
            "distance_to_footprint_mm": round(distance_mm, 1) if distance_mm is not None else None,
            "status": status,
            "warnings": warnings,
            "bbox": _bounds(poly),
        })
    return rows


def _page_image(page: fitz.Page, dpi: int = 140):
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    import numpy as np
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def _draw_page_poly(ax, poly, dpi: int, facecolor: str, edgecolor: str,
                    alpha: float = 0.18, linewidth: float = 2.0):
    if poly is None or poly.is_empty:
        return
    for part in _poly_parts(poly):
        xs, ys = part.exterior.xy
        xs = [x * dpi / 72.0 for x in xs]
        ys = [y * dpi / 72.0 for y in ys]
        ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)


def _save_page_overlay(pdf_path: str | None, page_idx: int, slabs: list, elements: list,
                       page_mapping: dict[int, dict], save_path: Path, title: str) -> str | None:
    if not pdf_path or page_idx < 0:
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        dpi = 140
        img = _page_image(page, dpi)
        h, w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        ax.imshow(img, origin="upper")
        bld_to_color = {}
        for slab in slabs:
            if getattr(slab, "page_index", -1) != page_idx:
                continue
            bld = page_mapping.get(page_idx, {}).get("building") or _slab_label_building(getattr(slab, "label", ""))
            if bld not in bld_to_color:
                bld_to_color[bld] = COLORS[len(bld_to_color) % len(COLORS)]
            _draw_page_poly(ax, getattr(slab, "polygon", None), dpi, bld_to_color[bld], bld_to_color[bld], 0.20, 2.0)
            poly = getattr(slab, "polygon", None)
            if poly is not None and not poly.is_empty:
                c = poly.centroid
                ax.text(c.x * dpi / 72, c.y * dpi / 72, bld, color="white", fontsize=7,
                        bbox=dict(facecolor="black", alpha=0.7, edgecolor=bld_to_color[bld], pad=2))
        for elem in elements:
            if getattr(elem, "page_index", -1) != page_idx:
                continue
            color = "#D500F9" if hasattr(elem, "symbol") else "#8E24AA"
            _draw_page_poly(ax, getattr(elem, "polygon", None), dpi, "none", color, 0.0, 1.4)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_title(title, fontsize=8, color="white", backgroundcolor="black")
        fig.tight_layout(pad=0)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        doc.close()
        return str(save_path)
    except Exception:
        return None


def _readiness(document_intelligence: dict | None, ai_floor_result: dict | None,
               registry: dict, assignments: list[dict]) -> dict:
    expected = _expected_buildings(document_intelligence, ai_floor_result)
    detected = list((registry or {}).get("buildings", {}).keys())
    warnings = []
    status = "verified"

    parse_status = _parse_status(document_intelligence)
    if parse_status != "ok":
        warnings.append(f"Document Intelligence parse_status is {parse_status}; semantic building mapping is not trusted.")
        status = "not_verified"
    if expected and len(detected) < len(expected):
        warnings.append(f"Expected {len(expected)} building(s) from semantic data but registry has {len(detected)}.")
        status = "not_verified"
    if "(unknown)" in detected:
        warnings.append("Registry contains unknown building geometry.")
        status = "not_verified"
    overlap_warnings = [w for w in (registry or {}).get("warnings", []) if "overlap" in str(w).lower()]
    if overlap_warnings:
        warnings.extend(overlap_warnings)
        status = "not_verified"
    orphan = [r for r in assignments if r.get("status") == "orphan"]
    review = [r for r in assignments if r.get("status") == "review"]
    if orphan:
        warnings.append(f"{len(orphan)} column/foundation/wall element(s) are orphaned.")
        status = "not_verified"
    if review:
        warnings.append(f"{len(review)} element(s) are far from assigned building footprint and need review.")
        status = "not_verified"

    return {
        "placement_status": status,
        "parse_status": parse_status,
        "expected_buildings": expected,
        "detected_buildings": detected,
        "expected_building_count": len(expected),
        "detected_building_count": len(detected),
        "orphan_element_count": len(orphan),
        "review_element_count": len(review),
        "warnings": warnings,
    }


def run_building_audit(pdf_path: str | None, output_root: str | Path,
                       slabs: list, columns: list | None = None,
                       foundations: list | None = None, walls: list | None = None,
                       document_intelligence: dict | None = None,
                       ai_floor_result: dict | None = None,
                       registry: dict | None = None) -> dict:
    """Write JSON/PNG audit artifacts and return a compact report."""
    columns = columns or []
    foundations = foundations or []
    walls = walls or []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root) / f"building_audit_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    page_mapping = _page_map(ai_floor_result)
    registry = registry or build_building_registry(slabs, columns, foundations, ai_floor_result)
    footprints = _building_footprints(registry)

    doc_payload = document_intelligence or {}
    doc_json = _write_json(out_dir / "01_document_intelligence.json", doc_payload)

    page_candidates = {
        "page_mapping": {str(k + 1): v for k, v in sorted(page_mapping.items())},
        "expected_buildings": _expected_buildings(document_intelligence, ai_floor_result),
        "ai_floor_result": ai_floor_result or {},
    }
    page_json = _write_json(out_dir / "02_page_building_candidates.json", page_candidates)

    slab_rows = _slab_rows(slabs, page_mapping)
    slab_json = _write_json(out_dir / "03_slab_footprints.json", {"slabs": slab_rows})

    page_images = {}
    pages = sorted({getattr(s, "page_index", -1) for s in slabs or [] if getattr(s, "page_index", -1) >= 0})
    for page_idx in pages:
        img = _save_page_overlay(
            pdf_path, page_idx, slabs, [],
            page_mapping,
            out_dir / f"03_slab_footprints_p{page_idx + 1:02d}.png",
            f"Slab Footprints / Building Candidates P{page_idx + 1}",
        )
        if img:
            page_images[f"p{page_idx + 1:02d}_slabs"] = img

    registry_json = _write_json(out_dir / "04_building_registry.json", _registry_json(registry))
    registry_img = out_dir / "04_building_registry.png"
    try:
        save_building_footprints(registry, str(registry_img))
    except Exception:
        registry_img = None

    assignments = []
    assignments.extend(_assign_elements(columns, "column", page_mapping, footprints))
    assignments.extend(_assign_elements(foundations, "foundation", page_mapping, footprints))
    assignments.extend(_assign_elements(walls, "wall", page_mapping, footprints))
    assignment_json = _write_json(out_dir / "05_element_assignment.json", {"elements": assignments})

    element_pages = sorted({
        getattr(e, "page_index", -1)
        for e in [*columns, *foundations, *walls]
        if getattr(e, "page_index", -1) >= 0
    })
    all_elements = [*columns, *foundations, *walls]
    for page_idx in element_pages:
        img = _save_page_overlay(
            pdf_path, page_idx, slabs, all_elements,
            page_mapping,
            out_dir / f"05_element_assignment_p{page_idx + 1:02d}.png",
            f"Element Assignment P{page_idx + 1}",
        )
        if img:
            page_images[f"p{page_idx + 1:02d}_elements"] = img

    readiness = _readiness(document_intelligence, ai_floor_result, registry, assignments)
    readiness_json = _write_json(out_dir / "06_readiness_report.json", readiness)

    summary = {
        "audit_dir": str(out_dir),
        "placement_status": readiness["placement_status"],
        "json_outputs": {
            "document_intelligence": doc_json,
            "page_building_candidates": page_json,
            "slab_footprints": slab_json,
            "building_registry": registry_json,
            "element_assignment": assignment_json,
            "readiness_report": readiness_json,
        },
        "image_outputs": {
            **page_images,
            "building_registry": str(registry_img) if registry_img else None,
        },
        "readiness": readiness,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary
