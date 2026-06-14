"""
Ruby export for slab_v2 — SketchUp script with element-based openings.

The slab is kept GROSS through extraction; here each element footprint
(stair/lift/shaft/void) that intersects a slab is subtracted with shapely
BEFORE code generation, so the generated face already carries its holes as
inner loops — no reliance on SketchUp's boolean engine. Element volumes
(shaft wall rings, stair flights) come from element_geometry.py.

generate_ruby           one page at Z=0 (debug / single-floor preview).
generate_building_ruby  multiple storeys stacked at their FFL elevations;
                        elements span floor-to-floor, same-type footprints
                        on consecutive storeys are paired (IoU) so missing
                        openings above a shaft surface as warnings — holes
                        themselves always come from each page's own X-cross
                        symbols, never invented.

Coordinates: real-world mm, page bottom-left origin, Y-up (same convention
as src/coordinate_mapper.py and v1 model_builder: slab face at Z=FFL,
extruded down).
"""

from __future__ import annotations

from pathlib import Path

import fitz
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.coordinate_mapper import transform_polygon
from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import SlabV2Result
from src.slab_v2.element_geometry import (element_ruby, face_with_holes,
                                          _solid_up)


def _elements_mm(result: SlabV2Result, page: fitz.Page, scale: int):
    ox, oy = page.rect.x0, page.rect.y1
    return [(e.type, e.label,
             transform_polygon(e.polygon, page, scale, ox, oy))
            for e in result.elements]


def _slab_polys_mm(result: SlabV2Result, page: fitz.Page, scale: int):
    ox, oy = page.rect.x0, page.rect.y1
    out = []
    for s in result.slabs:
        mm = s.get("polygon_mm") or transform_polygon(
            s["polygon_pdf"], page, scale, ox, oy)
        out.append((s["label"], mm))
    return out


def _columns_mm(result: SlabV2Result, page: fitz.Page, scale: int):
    ox, oy = page.rect.x0, page.rect.y1
    return [(c.symbol, transform_polygon(c.polygon, page, scale, ox, oy))
            for c in result.columns]


def _slab_lines(label: str, mm, openings, parent_var: str, layer: str,
                thickness_mm: float, z: float = 0.0) -> list[str]:
    if openings is not None:
        mm = mm.difference(openings)
    var = "slab_grp"
    lines = [
        "",
        f"layer = layers.add('{layer}')",
        f"{var} = {parent_var}.entities.add_group",
        f"{var}.layer = layer",
        f"{var}.name = '{label}'",
    ]
    for g in getattr(mm, "geoms", [mm]):
        if not g.is_empty:
            lines += face_with_holes(g, var, thickness_mm, z)
    return lines


def _element_lines(etype: str, label: str, mm, parent_var: str,
                   z_base: float, height: float,
                   cfg: SlabV2Config) -> tuple[list[str], list[str]]:
    body, warns = element_ruby(etype, mm, z_base, height, cfg, "elem_grp")
    if not body:
        return [], warns
    safe_label = label.replace("'", "")
    lines = [
        "",
        f"layer = layers.add('SLAB_V2_{etype}')",
        f"elem_grp = {parent_var}.entities.add_group",
        "elem_grp.layer = layer",
        f"elem_grp.name = '{etype} {safe_label}'",
    ] + body
    return lines, warns


def generate_ruby(
    result: SlabV2Result,
    page: fitz.Page,
    out_path: str,
    cfg: SlabV2Config | None = None,
) -> str:
    """Write a SketchUp Ruby script for one page's slabs + elements (Z=0)."""
    cfg = cfg or SlabV2Config()
    scale = result.scale or 100

    elements_mm = _elements_mm(result, page, scale)
    openings = unary_union([p for _, _, p in elements_mm]) \
        if elements_mm else None

    lines = [
        "# slab_v2 export — page %d, scale 1:%s" % (
            result.page_index + 1,
            f"{scale:.2f}" if isinstance(scale, float) else scale),
        "# slabs are GROSS; openings below were cut from element footprints",
        "model = Sketchup.active_model",
        "model.start_operation('slab_v2 import', true)",
        "layers = model.layers",
    ]

    layer = f"SLAB_V2_P{result.page_index + 1}"
    for label, mm in _slab_polys_mm(result, page, scale):
        lines += _slab_lines(label, mm, openings, "model", layer,
                             cfg.slab_thickness_mm)

    for etype, label, mm in elements_mm:
        elines, warns = _element_lines(etype, label, mm, "model",
                                       0.0, cfg.element_height_mm, cfg)
        lines += elines
        lines += [f"# WARN: {w}" for w in warns]

    lines += [
        "",
        "model.commit_operation",
        f"puts 'slab_v2: imported {len(result.slabs)} slab(s), "
        f"{len(result.elements)} element(s)'",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def _iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def generate_building_ruby(
    storeys: list[dict],
    out_path: str,
    cfg: SlabV2Config | None = None,
    site_offset_mm: tuple[float, float] = (0.0, 0.0),
) -> tuple[str, list[str]]:
    """Write one SketchUp Ruby script stacking several storeys by FFL.

    storeys: [{"result": SlabV2Result, "page": fitz.Page, "ffl_mm": float}]
    (any order — sorted by ffl_mm here).
    site_offset_mm: (dx, dy) translation from site placement audit.
    Returns (path, warnings).
    """
    cfg = cfg or SlabV2Config()
    warnings: list[str] = []

    # group storeys: by level_id if available (prevents FFL collision),
    # else by FFL (zone/part pages of the same floor share one level)
    has_level_id = all("level_id" in st for st in storeys)
    by_key: dict[str, list] = {}
    for st in storeys:
        key = st["level_id"] if has_level_id else str(int(round(st["ffl_mm"])))
        by_key.setdefault(key, []).append(st)
    levels = [{"ffl_mm": v[0]["ffl_mm"], "storeys": v}
              for v in by_key.values()]
    levels.sort(key=lambda lv: lv["ffl_mm"])

    # storey height = gap to the next FFL above; top floor reuses the gap
    # below it (typical constant storey height), else the default
    heights = []
    for i, lv in enumerate(levels):
        if i + 1 < len(levels):
            h = levels[i + 1]["ffl_mm"] - lv["ffl_mm"]
        elif heights:
            h = heights[-1]
        else:
            h = cfg.default_storey_height_mm
            warnings.append(
                f"single storey — element height defaults to {h:.0f}mm")
        heights.append(h)

    # precompute mm geometry per page, merged per level
    for lv in levels:
        lv["elements_mm"], lv["slabs_mm"], lv["columns_mm"] = [], [], []
        lv["pages"] = []
        for st in lv["storeys"]:
            scale = st["result"].scale or 100
            lv["elements_mm"] += _elements_mm(st["result"], st["page"],
                                              scale)
            lv["slabs_mm"] += _slab_polys_mm(st["result"], st["page"],
                                             scale)
            lv["columns_mm"] += _columns_mm(st["result"], st["page"], scale)
            lv["pages"].append(st["result"].page_index + 1)

    # vertical shaft pairing: same type, footprint IoU on consecutive levels
    pairs = 0
    for i in range(len(levels) - 1):
        below, above = levels[i], levels[i + 1]
        for etype, label, mm in below["elements_mm"]:
            if etype == "VOID":
                continue
            best = max((_iou(mm, mm2)
                        for et2, _l2, mm2 in above["elements_mm"]
                        if et2 == etype), default=0.0)
            if best >= cfg.shaft_pair_min_iou:
                pairs += 1
            else:
                warnings.append(
                    f"{etype} '{label}' on page(s) {below['pages']} (FFL "
                    f"{below['ffl_mm'] / 1000:+.3f}) has no matching opening "
                    f"on the storey above (page(s) {above['pages']}) "
                    f"— check step_09")

    lines = [
        "# slab_v2 building export — %d level(s)" % len(levels),
        "# pages: %s" % ", ".join(
            str(s["result"].page_index + 1) for s in storeys),
        "# slab faces sit at Z=FFL (top of slab) and extrude down;",
        "# element/column volumes rise from each FFL to the storey above",
        "model = Sketchup.active_model",
        "model.start_operation('slab_v2 building import', true)",
        "layers = model.layers",
    ]

    for lv, height in zip(levels, heights):
        pages_txt = ",".join(str(p) for p in lv["pages"])
        ffl_m = lv["ffl_mm"] / 1000.0
        level_var = "level_grp"
        lines += [
            "",
            f"# ── level: page(s) {pages_txt}, FFL {ffl_m:+.3f}m, "
            f"height {height:.0f}mm ──",
            f"{level_var} = model.entities.add_group",
            f"{level_var}.name = 'LEVEL FFL{ffl_m:+.3f}'",
        ]
        openings = unary_union([p for _, _, p in lv["elements_mm"]]) \
            if lv["elements_mm"] else None
        layer = f"SLAB_V2_P{pages_txt}"
        for label, mm in lv["slabs_mm"]:
            lines += _slab_lines(label, mm, openings, level_var, layer,
                                 cfg.slab_thickness_mm, z=lv["ffl_mm"])
        for etype, label, mm in lv["elements_mm"]:
            elines, warns = _element_lines(etype, label, mm, level_var,
                                           lv["ffl_mm"], height, cfg)
            lines += elines
            warnings += [f"page(s) {pages_txt}: {w}" for w in warns]
        for symbol, mm in lv["columns_mm"]:
            if mm.is_empty:
                continue
            lines += [
                "",
                "layer = layers.add('SLAB_V2_COLUMN')",
                f"elem_grp = {level_var}.entities.add_group",
                "elem_grp.layer = layer",
                f"elem_grp.name = 'COL_{symbol}'",
            ]
            for g in getattr(mm, "geoms", [mm]):
                lines += _solid_up(g, "elem_grp", lv["ffl_mm"], height)

    # site offset: translate the entire building if multi-building placement
    dx, dy = site_offset_mm
    if abs(dx) > 0.1 or abs(dy) > 0.1:
        lines += [
            "",
            f"# site placement offset: dx={dx:.1f}mm, dy={dy:.1f}mm",
            "all_ents = model.entities.to_a.select {{ |e| e.is_a?(Sketchup::Group) }}",
            f"t = Geom::Transformation.new([{dx:.1f}.mm, {dy:.1f}.mm, 0])",
            "all_ents.each {{ |e| e.transform!(t) }}",
        ]

    n_elems = sum(len(lv["elements_mm"]) for lv in levels)
    n_cols = sum(len(lv["columns_mm"]) for lv in levels)
    lines += [
        "",
        "model.commit_operation",
        f"puts 'slab_v2: imported {len(levels)} level(s), "
        f"{n_elems} element(s), {n_cols} column(s), "
        f"{pairs} vertical shaft pair(s)'",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out), warnings
