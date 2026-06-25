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

import math
from pathlib import Path

import fitz
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.coordinate_mapper import transform_polygon, pdf_point_to_real
from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import OpeningIntent, SlabV2Result
from src.slab_v2.element_geometry import (element_ruby, face_with_holes,
                                          _solid_up)


def _elements_mm(result: SlabV2Result, page: fitz.Page, scale: int):
    ox, oy = page.rect.x0, page.rect.y1
    return [(e.type, e.label,
             transform_polygon(e.polygon, page, scale, ox, oy))
            for e in result.elements]


def _resolved_openings_mm(result: SlabV2Result, page: fitz.Page, scale: int):
    ox, oy = page.rect.x0, page.rect.y1
    allowed = {
        OpeningIntent.SLAB_PENETRATION.value,
        OpeningIntent.VOID.value,
        OpeningIntent.LIFT_SHAFT.value,
    }
    openings = [
        element for element in result.verified_cut_openings
        if (element.opening_intent in allowed
            and bool(element.evidence_ids)
            and not ("STAIR" in element.object_roles
                     and not element.evidence_ids))
    ]
    return [(e.type, e.label,
             transform_polygon(e.polygon, page, scale, ox, oy))
            for e in openings]


def _render_elements_mm(result: SlabV2Result, page: fitz.Page, scale: int,
                        cfg: SlabV2Config):
    ox, oy = page.rect.x0, page.rect.y1
    elements = list(result.render_elements)
    elements = [element for element in elements
                if ((element.type not in {"SHAFT", "LIFT", "CORE"}
                     or cfg.render_shaft_solids)
                    and (element.type != "STAIR" or cfg.render_stair_solids))]
    return [(e.type, e.label,
             transform_polygon(e.polygon, page, scale, ox, oy))
            for e in elements]


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


def _steel_members_mm(result: SlabV2Result, page: fitz.Page, scale: int):
    ox, oy = page.rect.x0, page.rect.y1
    rows = []
    for member in getattr(result, "steel_members", []):
        if (getattr(member, "status", "") != "verified"
                or getattr(member, "polygon", None) is None):
            continue
        rows.append({
            "id": member.id,
            "symbol": member.symbol,
            "member_type": str(member.member_type or "COLUMN").upper(),
            "section": member.section,
            "status": member.status,
            "polygon": transform_polygon(member.polygon, page, scale, ox, oy),
        })
    return rows


def _walls_mm(result: SlabV2Result, page: fitz.Page, scale: int):
    ox, oy = page.rect.x0, page.rect.y1
    rows = []
    for wall in result.walls:
        # The document registry is keyed by wall symbol (W1/W2/W3), while
        # WallFootprint keeps the stable profile_id used in audit outputs.
        # Accept both forms so the verified elevation profile survives the
        # plan-resolver -> exporter handoff.
        profile = {}
        if wall.profile_id:
            profile = result.wall_profiles.get(wall.profile_id, {})
        if not profile:
            profile = result.wall_profiles.get(wall.label, {})
        if not profile and wall.profile_id:
            profile = next(
                (candidate for candidate in result.wall_profiles.values()
                 if candidate.get("profile_id") == wall.profile_id),
                {},
            )
        centerline = [pdf_point_to_real(
            x, y, page.rect.height, scale, ox, oy)
            for x, y in (wall.centerline or [])]
        rows.append({
            "wall": wall, "label": wall.label,
            "polygon": transform_polygon(wall.polygon, page, scale, ox, oy),
            "centerline": centerline,
            "profile": profile,
        })
    return rows


def _ruby_point3(point) -> str:
    return f"[{point[0]:.1f}.mm, {point[1]:.1f}.mm, {point[2]:.1f}.mm]"


def _profile_wall_lines(item: dict, group_var: str, z_base: float,
                        fallback_height: float) -> tuple[list[str], list[str]]:
    """Extrude station-Z elevation panels across wall thickness."""
    wall = item["wall"]
    profile = item.get("profile") or {}
    centerline = item.get("centerline") or []
    panels = profile.get("panels") or []
    if (profile.get("status") != "verified" or len(centerline) != 2
            or not panels or wall.w_mm <= 0):
        lines = []
        for geom in getattr(item["polygon"], "geoms", [item["polygon"]]):
            lines += _solid_up(geom, group_var, z_base, fallback_height)
        return lines, [f"{wall.label}: no verified elevation profile; "
                       "debug full-storey extrusion used"]

    (x0, y0), (x1, y1) = centerline
    dx, dy = x1-x0, y1-y0
    length = math.hypot(dx, dy)
    if length <= 1:
        return [], [f"{wall.label}: degenerate centerline"]
    ux, uy = dx/length, dy/length
    nx, ny = -uy*wall.w_mm/2, ux*wall.w_mm/2
    lines, warnings = [], []
    for panel_index, panel in enumerate(panels, 1):
        raw_station_z = panel.get("polygon_station_z") or []
        station_z = []
        for point in raw_station_z:
            point = (float(point[0]), float(point[1]))
            if not station_z or (abs(point[0]-station_z[-1][0]) > 1e-7 or
                                 abs(point[1]-station_z[-1][1]) > 0.1):
                station_z.append(point)
        if (len(station_z) > 2 and
                abs(station_z[0][0]-station_z[-1][0]) <= 1e-7 and
                abs(station_z[0][1]-station_z[-1][1]) <= 0.1):
            station_z.pop()
        if len(station_z) < 3:
            continue
        front, back = [], []
        for station, z in station_z:
            px, py = x0 + ux*length*station, y0 + uy*length*station
            front.append((px+nx, py+ny, z_base+z))
            back.append((px-nx, py-ny, z_base+z))
        lines += [
            f"# {wall.label} elevation panel {panel_index}",
            "front = [" + ", ".join(_ruby_point3(p) for p in front) + "]",
            "back = [" + ", ".join(_ruby_point3(p) for p in reversed(back)) + "]",
            f"face = {group_var}.entities.add_face(front)",
            f"face = {group_var}.entities.add_face(back)",
        ]
        n = len(front)
        for i in range(n):
            j = (i+1) % n
            quad = [front[i], front[j], back[j], back[i]]
            lines.append(
                f"face = {group_var}.entities.add_face([" +
                ", ".join(_ruby_point3(p) for p in quad) + "])")
    return lines, warnings


def _safe(s: str) -> str:
    return s.replace("'", "")


def _level_short(level_id: str, level_name: str = "") -> str:
    """Convert level_id to short tag prefix: 'level_1' -> 'L1', roof -> 'RF'."""
    lid = level_id.lower()
    if lid.startswith("level_"):
        return "L" + lid[6:]
    if "roof" in lid:
        return "RF"
    if "ground" in lid:
        return "GF"
    if "basement" in lid:
        num = "".join(c for c in lid if c.isdigit())
        return f"B{num}" if num else "B1"
    if level_name:
        import re
        m = re.search(r"(\d+)", level_name)
        if m:
            return "L" + m.group(1).lstrip("0")
    return level_id.upper().replace("LEVEL_", "L")


def _slab_lines(label: str, mm, openings, parent_var: str, tag_name: str,
                thickness_mm: float, z: float = 0.0) -> list[str]:
    if openings is not None:
        mm = mm.difference(openings)
    var = "slab_grp"
    lines = [
        "",
        f"{var} = {parent_var}.entities.add_group",
        f"{var}.layer = layers.add('{tag_name}')",
        f"{var}.name = '{_safe(label)}'",
    ]
    for g in getattr(mm, "geoms", [mm]):
        if not g.is_empty:
            lines += face_with_holes(g, var, thickness_mm, z)
    return lines


def _element_lines(etype: str, label: str, mm, parent_var: str,
                   z_base: float, height: float,
                   cfg: SlabV2Config,
                   tag_name: str = "") -> tuple[list[str], list[str]]:
    body, warns = element_ruby(etype, mm, z_base, height, cfg, "elem_grp")
    if not body:
        return [], warns
    tag = tag_name or f"SLAB_V2_{etype}"
    lines = [
        "",
        f"elem_grp = {parent_var}.entities.add_group",
        f"elem_grp.layer = layers.add('{tag}')",
        f"elem_grp.name = '{etype} {_safe(label)}'",
    ] + body
    return lines, warns


def _tag_folder_lines(all_tags: list[str],
                      folder_name: str = "STRC") -> list[str]:
    """Generate Ruby code: flat tags inside one STRC folder (SketchUp 2021+)."""
    lines = [
        "",
        "# ── Tag folder hierarchy (SketchUp 2021+) ──",
        "begin",
        f"  strc_f = layers.add_folder('{_safe(folder_name)}')",
    ]
    for tag in all_tags:
        lines.append(f"  strc_f.add_layer(layers['{tag}'])")
    lines += [
        "rescue => e",
        "  puts \"Tag folders require SketchUp 2021+ (#{e.message})\"",
        "end",
    ]
    return lines


def generate_ruby(
    result: SlabV2Result,
    page: fitz.Page,
    out_path: str,
    cfg: SlabV2Config | None = None,
) -> str:
    """Write a SketchUp Ruby script for one page's slabs + elements (Z=0)."""
    cfg = cfg or SlabV2Config()
    scale = result.scale or 100
    pnum = result.page_index + 1
    level_label = f"Page {pnum}"

    resolved_mm = _resolved_openings_mm(result, page, scale)
    render_mm = _render_elements_mm(result, page, scale, cfg)
    openings = unary_union([p for _, _, p in resolved_mm]) \
        if resolved_mm else None

    policy = getattr(cfg, "opening_policy_version", "penetration_only_v2")
    lines = [
        f"# Opening policy: {policy}",
        "# slab_v2 export — page %d, scale 1:%s" % (
            pnum,
            f"{scale:.2f}" if isinstance(scale, float) else scale),
        "# slabs are GROSS; openings below were cut from element footprints",
        f"# other floor systems retained but not rendered: "
        f"{len(result.other_floor_systems)}",
        f"# floor-system status: "
        f"{result.floor_system_readiness.get('status', 'review')}",
        "model = Sketchup.active_model",
        "model.start_operation('slab_v2 import', true)",
        "layers = model.layers",
    ]

    tag_floor = f"S P{pnum} FLOOR"
    tag_wall = f"S P{pnum} WALL"
    tag_stair = f"S P{pnum} STAIR_PLACEHOLDER"
    used_tags: list[str] = []

    # slabs → category group
    slab_polys = _slab_polys_mm(result, page, scale)
    if slab_polys:
        lines += [
            "",
            "cat_grp = model.entities.add_group",
            f"cat_grp.name = '{tag_floor}'",
            f"cat_grp.layer = layers.add('{tag_floor}')",
        ]
        for label, mm in slab_polys:
            lines += _slab_lines(label, mm, openings, "cat_grp", tag_floor,
                                 cfg.slab_thickness_mm)
        used_tags.append(tag_floor)

    # stairs → category group
    stair_elems = [(et, lb, m) for et, lb, m in render_mm
                   if et == "STAIR"]
    other_elems = [(et, lb, m) for et, lb, m in render_mm
                   if et != "STAIR"]
    if stair_elems:
        lines += [
            "",
            "cat_grp = model.entities.add_group",
            f"cat_grp.name = '{tag_stair}'",
            f"cat_grp.layer = layers.add('{tag_stair}')",
        ]
        has_stair_geom = False
        for etype, label, mm in stair_elems:
            elines, warns = _element_lines(etype, label, mm, "cat_grp",
                                           0.0, cfg.element_height_mm, cfg,
                                           tag_name=tag_stair)
            lines += elines
            lines += [f"# WARN: {w}" for w in warns]
            if elines:
                has_stair_geom = True
        if has_stair_geom:
            used_tags.append(tag_stair)
    for etype, label, mm in other_elems:
        elines, warns = _element_lines(etype, label, mm, "model",
                                       0.0, cfg.element_height_mm, cfg,
                                       tag_name=tag_floor)
        lines += elines
        lines += [f"# WARN: {w}" for w in warns]

    # walls → category group
    walls_mm_list = [item for item in _walls_mm(result, page, scale)
                     if not item["polygon"].is_empty]
    if walls_mm_list:
        lines += [
            "",
            "cat_grp = model.entities.add_group",
            f"cat_grp.name = '{tag_wall}'",
            f"cat_grp.layer = layers.add('{tag_wall}')",
        ]
        for item in walls_mm_list:
            wlabel = item["label"]
            lines += [
                "",
                "elem_grp = cat_grp.entities.add_group",
                f"elem_grp.layer = layers.add('{tag_wall}')",
                f"elem_grp.name = 'WALL_{_safe(wlabel)}'",
                "elem_grp.material = [142, 36, 170]",
            ]
            wall_lines, wall_warns = _profile_wall_lines(
                item, "elem_grp", 0.0, cfg.element_height_mm)
            lines += wall_lines
            lines += [f"# WARN: {w}" for w in wall_warns]
        used_tags.append(tag_wall)

    # tag folder hierarchy: flat STRC folder
    lines += _tag_folder_lines(used_tags, "STRC")

    lines += [
        "",
        "model.commit_operation",
        f"puts 'slab_v2: imported {len(result.slabs)} slab(s), "
        f"{len(result.elements)} raw element(s), "
        f"{len(result.verified_cut_openings)} verified cut opening(s), "
        f"{len(result.walls)} wall(s), "
        f"{len(result.columns)} column(s), "
        f"{len(result.other_floor_systems)} other floor system(s) retained, "
        f"judge accepted {len(result.opening_judgement.get('opening_ids', []))}, "
        f"excluded {len(result.opening_judgement.get('exclude_ids', []))}'",
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
    building_name: str = "",
    readiness_report=None,
) -> tuple[str, list[str]]:
    """Write one SketchUp Ruby script stacking several storeys by FFL.

    storeys: [{"result": SlabV2Result, "page": fitz.Page, "ffl_mm": float,
               "level_id": str}]
    (any order — sorted by ffl_mm here).
    site_offset_mm: (dx, dy) translation from site placement audit.
    building_name: used for tag folder hierarchy.
    Returns (path, warnings).
    """
    cfg = cfg or SlabV2Config()
    warnings: list[str] = []
    bname = building_name or "Building"

    # group storeys: by level_id if available (prevents FFL collision),
    # else by FFL (zone/part pages of the same floor share one level)
    has_level_id = all("level_id" in st for st in storeys)
    by_key: dict[str, list] = {}
    for st in storeys:
        key = st["level_id"] if has_level_id else str(int(round(st["ffl_mm"])))
        by_key.setdefault(key, []).append(st)
    levels = [{"ffl_mm": v[0]["ffl_mm"], "storeys": v,
               "level_id": v[0].get("level_id", ""),
               "level_name": v[0].get("level_name", "")}
              for v in by_key.values()]
    levels.sort(key=lambda lv: lv["ffl_mm"])

    # storey height = gap to the next FFL above; top floor reuses the gap
    # below it (typical constant storey height), else the default
    heights = []
    for i, lv in enumerate(levels):
        provided = [s.get("height_mm") for s in lv["storeys"]
                    if s.get("height_mm") and s.get("height_mm") > 0]
        statuses = [s.get("height_status") for s in lv["storeys"]
                    if s.get("height_status")]
        lv["height_status"] = statuses[0] if statuses else "default_unsafe"
        if provided:
            h = provided[0]
        elif i + 1 < len(levels):
            h = levels[i + 1]["ffl_mm"] - lv["ffl_mm"]
            lv["height_status"] = "debug_ffl_gap"
            warnings.append(
                f"{lv['level_id']}: no LevelDatum height; debug FFL gap used")
        elif heights:
            h = heights[-1]
            lv["height_status"] = "debug_reused_below"
            warnings.append(
                f"{lv['level_id']}: no top-storey height; debug value reused")
        else:
            h = cfg.default_storey_height_mm
            lv["height_status"] = "default_unsafe"
            warnings.append(
                f"single storey — element height defaults to {h:.0f}mm")
        heights.append(h)

    # precompute mm geometry per page, merged per level
    for lv in levels:
        lv["elements_mm"], lv["openings_mm"], lv["render_mm"] = [], [], []
        lv["slabs_mm"], lv["columns_mm"] = [], []
        lv["steel_members_mm"] = []
        lv["walls_mm"] = []
        lv["pages"] = []
        for st in lv["storeys"]:
            scale = st["result"].scale or 100
            lv["elements_mm"] += _elements_mm(st["result"], st["page"],
                                              scale)
            lv["openings_mm"] += _resolved_openings_mm(
                st["result"], st["page"], scale)
            lv["render_mm"] += _render_elements_mm(
                st["result"], st["page"], scale, cfg)
            lv["slabs_mm"] += _slab_polys_mm(st["result"], st["page"],
                                             scale)
            lv["columns_mm"] += _columns_mm(st["result"], st["page"], scale)
            lv["steel_members_mm"] += _steel_members_mm(
                st["result"], st["page"], scale)
            lv["walls_mm"] += _walls_mm(st["result"], st["page"], scale)
            lv["pages"].append(st["result"].page_index + 1)

    from shapely.affinity import translate as _translate
    def _centered(p: Polygon) -> Polygon:
        cx, cy = p.centroid.coords[0]
        return _translate(p, -cx, -cy)
    _seen_walls: dict[str, Polygon] = {}
    for lv in levels:
        deduped = []
        for item in lv["walls_mm"]:
            label = item["label"]
            if label.upper().startswith("LW"):
                deduped.append(item)
                continue
            norm = _centered(item["polygon"])
            if label in _seen_walls:
                if _iou(norm, _seen_walls[label]) > 0.90:
                    continue
            else:
                _seen_walls[label] = norm
            deduped.append(item)
        lv["walls_mm"] = deduped

    model_status = getattr(readiness_report, "model_status", "debug")
    readiness_reasons = getattr(readiness_report, "reasons", [])
    opening_policy = getattr(
        cfg, "opening_policy_version", "penetration_only_v2")

    # vertical shaft pairing: same type, footprint IoU on consecutive levels
    pairs = 0
    for i in range(len(levels) - 1):
        below, above = levels[i], levels[i + 1]
        for etype, label, mm in below["openings_mm"]:
            if etype == "VOID":
                continue
            best = max((_iou(mm, mm2)
                        for et2, _l2, mm2 in above["openings_mm"]
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
        "# slab_v2 building export — %s, %d level(s)" % (bname, len(levels)),
        "# Opening policy: %s" % opening_policy,
        ("# FINAL VERIFIED MODEL" if model_status == "final"
         else "# UNVERIFIED MODEL - DEBUG USE ONLY"),
        "# readiness: %s" % model_status,
        "# readiness reasons: %s" % ("; ".join(readiness_reasons) or "none"),
        "# other floor systems are retained in audit JSON but not rendered",
        "# other floor system count: %d" % sum(
            len(s["result"].other_floor_systems) for s in storeys),
        "# steel export policy: verified_only",
        "# steel member count: %d" % sum(
            len(getattr(s["result"], "steel_members", [])) for s in storeys),
        "# pages: %s" % ", ".join(
            str(s["result"].page_index + 1) for s in storeys),
        "# slab faces sit at Z=FFL (top of slab) and extrude down;",
        "# element/column volumes rise from each FFL to the storey above",
        "model = Sketchup.active_model",
        "model.start_operation('slab_v2 building import', true)",
        "layers = model.layers",
    ]

    # ── Tag naming: S {LEVEL_SHORT} {CATEGORY}  inside flat STRC folder ──
    all_tags: list[str] = []

    for lv, height in zip(levels, heights):
        pages_txt = ",".join(str(p) for p in lv["pages"])
        ffl_m = lv["ffl_mm"] / 1000.0
        level_id = lv.get("level_id") or f"FFL{ffl_m:+.3f}"
        level_name = lv.get("level_name") or ""
        lv_short = _level_short(level_id, level_name)
        level_label = f"{level_id} (FFL {ffl_m:+.3f}m)"
        level_var = "level_grp"

        tag_floor = f"S {lv_short} FLOOR"
        tag_wall = f"S {lv_short} WALL"
        tag_col_conc = f"S {lv_short} COLUMN CONCRETE"
        tag_steel_col = f"S {lv_short} STEEL COLUMN"
        tag_steel_beam = f"S {lv_short} STEEL BEAM"
        tag_steel_bracing = f"S {lv_short} STEEL BRACING"
        tag_steel_floor = f"S {lv_short} STEEL FLOOR"
        # Stair detail sheets are not yet geometrically reconciled with the
        # plan footprint.  The opening is authoritative; generated steps are
        # explicitly tagged as placeholders until detail evidence is verified.
        tag_stair = f"S {lv_short} STAIR_PLACEHOLDER"

        lines += [
            "",
            f"# ── {level_label}, page(s) {pages_txt}, "
            f"height {height:.0f}mm ──",
            f"# Height evidence status: {lv.get('height_status')}",
            f"{level_var} = model.entities.add_group",
            f"{level_var}.name = '{_safe(level_label)}'",
        ]
        openings = unary_union([p for _, _, p in lv["openings_mm"]]) \
            if lv["openings_mm"] else None

        # slabs → S Lx FLOOR (category parent group)
        if lv["slabs_mm"]:
            lines += [
                "",
                f"cat_grp = {level_var}.entities.add_group",
                f"cat_grp.name = '{tag_floor}'",
                f"cat_grp.layer = layers.add('{tag_floor}')",
            ]
            for label, mm in lv["slabs_mm"]:
                lines += _slab_lines(label, mm, openings, "cat_grp",
                                     tag_floor, cfg.slab_thickness_mm,
                                     z=lv["ffl_mm"])
            if tag_floor not in all_tags:
                all_tags.append(tag_floor)

        # elements: stairs → S Lx STAIR, others → S Lx FLOOR (cut only)
        if lv["render_mm"]:
            stair_items = [(et, lb, m) for et, lb, m in lv["render_mm"]
                           if et == "STAIR"]
            other_items = [(et, lb, m) for et, lb, m in lv["render_mm"]
                           if et != "STAIR"]
            if stair_items:
                lines += [
                    "",
                    f"cat_grp = {level_var}.entities.add_group",
                    f"cat_grp.name = '{tag_stair}'",
                    f"cat_grp.layer = layers.add('{tag_stair}')",
                ]
                has_stair_geom = False
                for etype, label, mm in stair_items:
                    elines, warns = _element_lines(
                        etype, label, mm, "cat_grp",
                        lv["ffl_mm"], height, cfg, tag_name=tag_stair)
                    lines += elines
                    warnings += [f"page(s) {pages_txt}: {w}" for w in warns]
                    if elines:
                        has_stair_geom = True
                if has_stair_geom and tag_stair not in all_tags:
                    all_tags.append(tag_stair)
            for etype, label, mm in other_items:
                elines, warns = _element_lines(
                    etype, label, mm, level_var,
                    lv["ffl_mm"], height, cfg, tag_name=tag_floor)
                lines += elines
                warnings += [f"page(s) {pages_txt}: {w}" for w in warns]

        # RC columns stay separate from the steel subsystem.  Older exports
        # guessed steel from UC prefixes here; verified steel now comes from
        # result.steel_members only.
        if lv["columns_mm"]:
            conc_cols = [(s, m) for s, m in lv["columns_mm"]
                         if not m.is_empty]
            if conc_cols:
                lines += [
                    "",
                    f"cat_grp = {level_var}.entities.add_group",
                    f"cat_grp.name = '{tag_col_conc}'",
                    f"cat_grp.layer = layers.add('{tag_col_conc}')",
                ]
                for symbol, mm in conc_cols:
                    lines += [
                        "",
                        "elem_grp = cat_grp.entities.add_group",
                        f"elem_grp.layer = layers.add('{tag_col_conc}')",
                        f"elem_grp.name = 'COL_{_safe(symbol)}'",
                    ]
                    for g in getattr(mm, "geoms", [mm]):
                        lines += _solid_up(g, "elem_grp",
                                           lv["ffl_mm"], height)
                if tag_col_conc not in all_tags:
                    all_tags.append(tag_col_conc)

        steel_columns = [
            item for item in lv.get("steel_members_mm", [])
            if item.get("member_type") == "COLUMN"
            and not item.get("polygon").is_empty
        ]
        if steel_columns:
            lines += [
                "",
                f"cat_grp = {level_var}.entities.add_group",
                f"cat_grp.name = '{tag_steel_col}'",
                f"cat_grp.layer = layers.add('{tag_steel_col}')",
            ]
            for item in steel_columns:
                symbol = item.get("symbol") or "STEEL"
                member_id = item.get("id") or "steel"
                section = item.get("section") or "UNKNOWN_SECTION"
                status = item.get("status") or "verified"
                lines += [
                    "",
                    "elem_grp = cat_grp.entities.add_group",
                    f"elem_grp.layer = layers.add('{tag_steel_col}')",
                    f"elem_grp.name = 'STEEL_COL_{_safe(symbol)}_{_safe(member_id)}'",
                    "elem_grp.material = [55, 135, 210]",
                    f"# steel section: {_safe(section)}; status: {_safe(status)}",
                ]
                for g in getattr(item["polygon"], "geoms", [item["polygon"]]):
                    lines += _solid_up(g, "elem_grp",
                                       lv["ffl_mm"], height)
            if tag_steel_col not in all_tags:
                all_tags.append(tag_steel_col)
        _ = (tag_steel_beam, tag_steel_bracing, tag_steel_floor)

        # walls → S Lx WALL (category parent group)
        wall_items = [item for item in lv["walls_mm"]
                      if not item["polygon"].is_empty] if lv["walls_mm"] else []
        if wall_items:
            lines += [
                "",
                f"cat_grp = {level_var}.entities.add_group",
                f"cat_grp.name = '{tag_wall}'",
                f"cat_grp.layer = layers.add('{tag_wall}')",
            ]
            for item in wall_items:
                wlabel = item["label"]
                lines += [
                    "",
                    "elem_grp = cat_grp.entities.add_group",
                    f"elem_grp.layer = layers.add('{tag_wall}')",
                    f"elem_grp.name = 'WALL_{_safe(wlabel)}'",
                    "elem_grp.material = [142, 36, 170]",
                ]
                wall_lines, wall_warns = _profile_wall_lines(
                    item, "elem_grp", lv["ffl_mm"], height)
                lines += wall_lines
                lines += [f"# WARN: {w}" for w in wall_warns]
            if tag_wall not in all_tags:
                all_tags.append(tag_wall)

    # tag folder hierarchy: flat STRC folder (SketchUp 2021+)
    lines += _tag_folder_lines(all_tags, "STRC")

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
    n_openings = sum(len(lv["openings_mm"]) for lv in levels)
    n_cols = sum(len(lv["columns_mm"]) for lv in levels)
    n_steel_members = sum(len(lv.get("steel_members_mm", [])) for lv in levels)
    n_steel_cols = sum(
        sum(1 for item in lv.get("steel_members_mm", [])
            if item.get("member_type") == "COLUMN") for lv in levels)
    steel_statuses = sorted({
        str(st["result"].steel_readiness.get("status", "not_required"))
        for st in storeys
        if getattr(st["result"], "steel_readiness", None)
    })
    steel_status = ",".join(steel_statuses) or "not_required"
    n_walls = sum(len(lv["walls_mm"]) for lv in levels)
    n_shaft_solids = sum(
        sum(1 for etype, _label, _mm in lv["render_mm"]
            if etype in {"SHAFT", "LIFT", "CORE"}) for lv in levels)
    n_stair_solids = sum(
        sum(1 for etype, _label, _mm in lv["render_mm"]
            if etype == "STAIR") for lv in levels)
    judge_accepted = sum(
        len(st["result"].opening_judgement.get("opening_ids", []))
        for st in storeys)
    judge_excluded = sum(
        len(st["result"].opening_judgement.get("exclude_ids", []))
        for st in storeys)
    expected_rc = sum(
        sum(st["result"].column_detection_report.get("expected", {}).values())
        for st in storeys)
    detected_rc = sum(
        sum(st["result"].column_detection_report.get("detected", {}).values())
        for st in storeys)
    n_other_floors = sum(
        len(st["result"].other_floor_systems) for st in storeys)
    n_stair_context = sum(
        len(st["result"].opening_context_objects) for st in storeys)
    n_prevented_stairs = sum(
        len(st["result"].opening_report.get(
            "prevented_stair_cut_ids", [])) for st in storeys)
    n_mixed_review = sum(
        len(st["result"].opening_report.get(
            "unresolved_mixed_ids", [])) for st in storeys)
    lines += [
        "",
        "model.commit_operation",
        f"puts 'slab_v2: imported {len(levels)} level(s), "
        f"{n_elems} raw element(s), {n_openings} verified cut opening(s), "
        f"{n_cols} RC column(s), {n_steel_cols} steel column(s), "
        f"{n_walls} wall(s), "
        f"{n_shaft_solids} shaft solid(s), "
        f"{n_stair_solids} stair solid(s), "
        f"{n_other_floors} other floor system(s) retained, "
        f"{pairs} vertical shaft pair(s), "
        f"judge accepted {judge_accepted}, excluded {judge_excluded}, "
        f"RC columns {detected_rc}/{expected_rc}'",
        f"puts 'Opening policy: {opening_policy}; stair context: "
        f"{n_stair_context}; prevented stair cuts: {n_prevented_stairs}; "
        f"unresolved mixed candidates: {n_mixed_review}; stair solids: 0'",
        f"puts 'Steel readiness: {steel_status}; steel export: "
        f"verified_only; steel members: {n_steel_members}'",
        f"puts 'MODEL READINESS: {model_status.upper()}'",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out), warnings
