"""
SketchUp Ruby script generator — slabs, columns & foundations.

Generates a .rb script with:
  1. Layers per floor level (slabs) + Columns + Foundations
  2. Slabs: Face → pushpull down per thickness
  3. Columns: Face → pushpull UP from slab FFL
  4. Foundations: Face → pushpull DOWN below GL
  5. Material colors per type, grouped elements
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from typing import Optional
import re
from shapely.geometry import MultiPolygon as _MultiPolygon


def _get_polygon(geom):
    if isinstance(geom, _MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


LEVEL_COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D", "#F06292",
    "#CE93D8", "#80DEEA", "#FFCC02", "#FF8A65",
]
COLUMN_COLOR = "#9E9E9E"
FOOTING_COLOR = "#795548"
WALL_COLOR = "#8E24AA"


def _ruby_point(x_mm, y_mm, z_mm):
    return f"Geom::Point3d.new({x_mm:.2f}.mm, {y_mm:.2f}.mm, {z_mm:.2f}.mm)"


def _sanitize(n):
    return re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", n)


def _clean_ring_coords(poly, tol_mm=1.0):
    """Return exterior coords safe for SketchUp add_face."""
    if poly is None or poly.is_empty:
        return []
    try:
        geom = poly.buffer(0)
        if geom.is_empty:
            geom = poly
    except Exception:
        geom = poly
    geom = _get_polygon(geom)
    coords = list(geom.exterior.coords)
    cleaned = []
    for x, y in coords:
        if cleaned:
            px, py = cleaned[-1]
            if abs(x - px) <= tol_mm and abs(y - py) <= tol_mm:
                continue
        cleaned.append((float(x), float(y)))
    if len(cleaned) > 1:
        fx, fy = cleaned[0]
        lx, ly = cleaned[-1]
        if abs(fx - lx) <= tol_mm and abs(fy - ly) <= tol_mm:
            cleaned = cleaned[:-1]
    unique = []
    seen = set()
    for x, y in cleaned:
        key = (round(x / tol_mm), round(y / tol_mm))
        if key in seen:
            continue
        seen.add(key)
        unique.append((x, y))
    return unique if len(unique) >= 3 else []


def _clean_hole_coords(interior, tol_mm=1.0):
    cleaned = []
    for x, y in list(interior.coords):
        if cleaned:
            px, py = cleaned[-1]
            if abs(x - px) <= tol_mm and abs(y - py) <= tol_mm:
                continue
        cleaned.append((float(x), float(y)))
    if len(cleaned) > 1:
        fx, fy = cleaned[0]
        lx, ly = cleaned[-1]
        if abs(fx - lx) <= tol_mm and abs(fy - ly) <= tol_mm:
            cleaned = cleaned[:-1]
    return cleaned if len(cleaned) >= 3 else []


def generate_ruby_script(slab_regions, output_path, thickness_mm=200.0,
                         generated_by="Feeldx Slab Extractor", **kwargs):
    return generate_full_ruby_script(slab_regions=slab_regions,
                                     column_regions=[], foundation_regions=[],
                                     output_path=output_path,
                                     slab_thickness_mm=thickness_mm,
                                     generated_by=generated_by)


def generate_slab_csv(slab_regions, output_path, **kwargs):
    rows = [{"ID": s.id, "Label": s.label, "FFL_m": s.ffl_m or "",
             "FFL_mm": s.ffl_mm or "", "Thickness_mm": 200,
             "Area_m2": f"{s.area_m2:.3f}" if s.area_m2 else "",
             "Page": s.page_index + 1, "Source": s.source} for s in slab_regions]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return output_path


def compute_storey_heights(slab_regions) -> dict:
    """Compute page-index -> height to next FFL in metres."""
    page_ffl = {}
    for s in slab_regions:
        if s.ffl_m is not None and s.page_index not in page_ffl:
            page_ffl[s.page_index] = float(s.ffl_m)
    ordered = sorted(page_ffl.items(), key=lambda kv: kv[1])
    heights = {}
    for i, (page_idx, ffl) in enumerate(ordered):
        if i + 1 < len(ordered):
            heights[page_idx] = round(ordered[i + 1][1] - ffl, 4)
    return heights


def generate_full_ruby_script(slab_regions, column_regions, foundation_regions,
                              output_path, slab_thickness_mm=200.0,
                              column_height_mm=3000.0,
                              storey_height_by_page_mm=None,
                              storey_height_report=None,
                              building_registry=None,
                              site_placement_report=None,
                              single_model=True,
                              preserve_native_building_position=True,
                              generated_by="Feeldx Pipeline",
                              wall_regions=None):
    """Generate complete Ruby script for all structural elements."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wall_regions = wall_regions or []
    total = len(slab_regions) + len(column_regions) + len(foundation_regions) + len(wall_regions)
    storey_height_by_page_mm = storey_height_by_page_mm or {}
    storey_height_report = storey_height_report or []
    building_registry = building_registry or {"buildings": {}, "warnings": []}
    site_placement_report = site_placement_report or {}
    site_readiness = site_placement_report.get("readiness") or {}
    site_transform = site_placement_report.get("site_transform") or {}
    site_transform_map = site_transform.get("building_transforms") or {}
    site_scale = site_transform.get("scale") or site_readiness.get("scale") or {}
    site_verified = site_readiness.get("site_placement_status") == "verified"
    building_offsets = {}
    if site_verified:
        for bld, transform in site_transform_map.items():
            if transform.get("status") != "verified":
                continue
            dx = transform.get("dx_mm")
            dy = transform.get("dy_mm")
            if dx is None or dy is None:
                continue
            building_offsets[bld] = (float(dx), float(dy))
    building_count = len(building_registry.get("buildings", {}))
    floor_count = sum(
        sum(1 for lvl in b.get("levels", {}) if str(lvl).lower() != "foundation")
        for b in building_registry.get("buildings", {}).values()
    )
    height_status_counts = Counter((r.get("Status") or "unknown") for r in storey_height_report)

    lines += [
        "# " + "=" * 60,
        f"# Generated by: {generated_by}",
        f"# Date: {now}",
        f"# Buildings: {building_count} | Floors: {floor_count}",
        f"# Slabs: {len(slab_regions)} | Columns: {len(column_regions)} | Foundations: {len(foundation_regions)} | Walls: {len(wall_regions)}",
        f"# Total elements: {total}",
        f"# Model mode: {'single_model' if single_model else 'multi_export'} | Building position: "
        f"{'site_keyplan_verified' if site_verified and building_offsets else ('native_coordinates' if preserve_native_building_position else 'presentation_offset')}",
        f"# Site/keyplan source page: {site_readiness.get('primary_source_page', 'N/A')} | "
        f"Scale: {site_scale.get('source') or 'N/A'} | "
        f"Placed buildings: {len(building_offsets)}",
        "# Storey heights: "
        f"verified={height_status_counts.get('verified', 0)} | "
        f"inferred={height_status_counts.get('inferred', 0)} | "
        f"missing/default={height_status_counts.get('missing', 0) + height_status_counts.get('default', 0)}",
        "# Paste into SketchUp Ruby Console (Window > Ruby Console)",
        "# " + "=" * 60,
        "",
        "model = Sketchup.active_model",
        "model.start_operation('Import Structural Model', true)",
        "entities = model.active_entities",
        "layers = model.layers",
        "materials = model.materials",
        "model.options['UnitsOptions']['LengthUnit'] = 4  # mm",
        "",
    ]

    # Page-level dominant FFL
    page_ffl_counts = defaultdict(Counter)
    for s in slab_regions:
        if s.ffl_m is not None:
            page_ffl_counts[s.page_index][round(s.ffl_m, 3)] += 1
    page_primary_ffl = {p: c.most_common(1)[0][0]
                        for p, c in page_ffl_counts.items() if c}

    # Layers & materials
    lines.append("# --- Layers & Materials ---")
    level_map = {}
    for s in slab_regions:
        lvl = page_primary_ffl.get(s.page_index)
        key = f"P{s.page_index + 1:02d}_FFL_{lvl:.1f}m" if lvl is not None else f"P{s.page_index + 1:02d}"
        if key not in level_map:
            idx = len(level_map)
            color = LEVEL_COLORS[idx % len(LEVEL_COLORS)]
            level_map[key] = {"idx": idx, "color": color, "ffl_mm": (lvl or 0) * 1000}
            safe = _sanitize(key)
            lines.append(f'layer_slab_{idx} = layers.add("{safe}")')
            lines.append(f'mat_slab_{idx} = materials.add("mat_{safe}"); mat_slab_{idx}.color = Sketchup::Color.new("{color}")')
    lines.append('layer_col = layers.add("Columns")')
    lines.append(f'mat_col = materials.add("mat_columns"); mat_col.color = Sketchup::Color.new("{COLUMN_COLOR}")')
    lines.append('layer_fdn = layers.add("Foundations")')
    lines.append(f'mat_fdn = materials.add("mat_foundations"); mat_fdn.color = Sketchup::Color.new("{FOOTING_COLOR}")')
    lines.append('layer_wall = layers.add("Walls")')
    lines.append(f'mat_wall = materials.add("mat_walls"); mat_wall.color = Sketchup::Color.new("{WALL_COLOR}")')
    lines.append("")

    # Building/level/type group containers. Coordinates are preserved inside these groups.
    page_to_building = defaultdict(list)
    for b in building_registry.get("buildings", {}).values():
        for page_1 in b.get("slab_pages", []) or []:
            if isinstance(page_1, int) and page_1 > 0:
                page_to_building[page_1 - 1].append(b.get("name") or "(unknown)")
    container_entities = {}
    container_idx = 0

    def _context_for_slab(slab):
        label = getattr(slab, "label", "") or ""
        bld = None
        candidates = page_to_building.get(getattr(slab, "page_index", -1), [])
        if len(candidates) == 1:
            bld = candidates[0]
            lvl_name = re.sub(rf"^{re.escape(bld)}\s*[—–\-_]*\s*", "", label).strip() or label
            lvl_name = re.sub(r"^[^A-Za-z0-9]+", "", lvl_name).strip()
            return bld or "(unknown)", lvl_name or "Level"
        parts = re.split(r"\s+[—–-]\s+", label, maxsplit=1)
        if len(parts) == 2:
            bld = parts[0].strip()
            lvl_name = parts[1].strip()
        else:
            bld = candidates[0] if len(candidates) == 1 else "(unknown)"
            lvl_name = label or f"Page {getattr(slab, 'page_index', -1) + 1}"
        return bld or "(unknown)", lvl_name or "Level"

    def _context_for_element(elem, fallback_type="element"):
        bld = getattr(elem, "building", "") or ""
        lvl_name = getattr(elem, "level", "") or ""
        if not bld:
            candidates = page_to_building.get(getattr(elem, "page_index", -1), [])
            bld = candidates[0] if len(candidates) == 1 else "(unknown)"
        if not lvl_name:
            lvl_name = fallback_type
        return bld or "(unknown)", lvl_name or fallback_type

    def _offset_for_building(building):
        if not site_verified:
            return 0.0, 0.0
        return building_offsets.get(building or "", (0.0, 0.0))

    def _emit_points(var_name, coords, z_mm, building, indent="  "):
        dx, dy = _offset_for_building(building)
        lines.append(f"{indent}{var_name} = [")
        for x, y in coords:
            lines.append(f"{indent}  {_ruby_point(x + dx, y + dy, z_mm)},")
        lines[-1] = lines[-1].rstrip(",")
        lines.append(f"{indent}]")

    def _ensure_container(building, level, elem_type):
        nonlocal container_idx
        key = (building or "(unknown)", level or "Level", elem_type)
        if key in container_entities:
            return container_entities[key]
        idx = container_idx
        container_idx += 1
        b_safe = _sanitize(key[0])
        l_safe = _sanitize(key[1])
        t_safe = _sanitize(key[2])
        lines.extend([
            f'grp_container_{idx} = entities.add_group',
            f'grp_container_{idx}.name = "{b_safe} / {l_safe} / {t_safe}"',
            f'ents_container_{idx} = grp_container_{idx}.entities',
        ])
        container_entities[key] = f"ents_container_{idx}"
        return container_entities[key]

    # --- SLABS ---
    lines.append("# === SLABS ===")
    for i, slab in enumerate(slab_regions):
        poly = _get_polygon(getattr(slab, "real_polygon", None))
        if poly is None or poly.is_empty:
            lines.append(f"# SKIP slab {slab.label}: no polygon")
            continue
        lvl = page_primary_ffl.get(slab.page_index)
        key = f"P{slab.page_index + 1:02d}_FFL_{lvl:.1f}m" if lvl is not None else f"P{slab.page_index + 1:02d}"
        if key not in level_map:
            continue
        info = level_map[key]
        z_mm = slab.ffl_mm if slab.ffl_mm else 0.0
        safe = _sanitize(slab.label)
        coords = _clean_ring_coords(poly)
        if len(coords) < 3:
            continue
        area_txt = f"{slab.area_m2:.2f}m²" if slab.area_m2 else "?"
        bld, lvl_name = _context_for_slab(slab)
        slab_entities = _ensure_container(bld, lvl_name, "Slabs")
        lines += ["", f"# Slab {i + 1}: {slab.label} | Area: {area_txt}", "begin",
                  f"  # building offset mm: dx={_offset_for_building(bld)[0]:.2f}, dy={_offset_for_building(bld)[1]:.2f}"]
        _emit_points(f"pts_slab_{i}", coords, z_mm, bld)
        lines += [f"  grp_slab_{i} = {slab_entities}.add_group",
                  f"  face_slab_{i} = grp_slab_{i}.entities.add_face(pts_slab_{i})",
                  f"  if face_slab_{i} && face_slab_{i}.valid?",
                  f"    hole_faces_{i} = []"]
        for h_idx, interior in enumerate(getattr(poly, "interiors", [])):
            hole_coords = _clean_hole_coords(interior)
            if len(hole_coords) < 3:
                continue
            _emit_points(f"pts_slab_{i}_hole_{h_idx}", hole_coords, z_mm, bld, indent="    ")
            lines += [
                f"    hole_face_{i}_{h_idx} = grp_slab_{i}.entities.add_face(pts_slab_{i}_hole_{h_idx})",
                f"    hole_faces_{i} << hole_face_{i}_{h_idx} if hole_face_{i}_{h_idx} && hole_face_{i}_{h_idx}.valid?",
            ]
        lines += [f"    hole_faces_{i}.each {{ |hf| hf.erase! if hf && hf.valid? }}",
                  f"    face_slab_{i}.pushpull(-{slab_thickness_mm:.0f}.mm)",
                  f"    grp_slab_{i}.layer = layer_slab_{info['idx']}",
                  f"    grp_slab_{i}.material = mat_slab_{info['idx']}",
                  f"    grp_slab_{i}.name = '{safe}'",
                  "  end", "rescue => e",
                  f"  puts 'Error slab {safe}: ' + e.message", "end"]

    # --- COLUMNS ---
    lines += ["", "# === COLUMNS ==="]
    col_idx = 0
    for col in column_regions:
        poly = _get_polygon(getattr(col, "real_polygon", None)) or getattr(col, "polygon", None)
        if poly is None or poly.is_empty:
            lines.append(f"# SKIP col {col.symbol}: no polygon")
            continue
        z_base = 0.0
        if slab_regions:
            cx, cy = poly.centroid.x, poly.centroid.y
            best = float("inf")
            for s in slab_regions:
                if s.page_index != col.page_index:
                    continue
                sp = _get_polygon(getattr(s, "real_polygon", None))
                if sp is None or sp.is_empty:
                    continue
                d = ((cx - sp.centroid.x) ** 2 + (cy - sp.centroid.y) ** 2) ** 0.5
                if d < best and s.ffl_mm is not None:
                    best, z_base = d, s.ffl_mm
        if z_base == 0:
            z_base = page_primary_ffl.get(col.page_index, 0) * 1000
        safe = _sanitize(col.symbol)
        w, d = col.width_mm or 200, col.depth_mm or 200
        col_height = (
            getattr(col, "height_mm", 0)
            or storey_height_by_page_mm.get(col.page_index)
            or column_height_mm
        )
        bld, lvl_name = _context_for_element(col, "Columns")
        col_entities = _ensure_container(bld, lvl_name, "Columns")
        coords = _clean_ring_coords(poly)
        if len(coords) < 3:
            cx, cy = poly.centroid.x, poly.centroid.y
            hw, hd = w / 2, d / 2
            coords = [(cx - hw, cy - hd), (cx + hw, cy - hd),
                      (cx + hw, cy + hd), (cx - hw, cy + hd)]
        if len(coords) < 3:
            continue
        lines += ["", f"# Col {col.symbol} Z={z_base:.0f}mm {w:.0f}x{d:.0f}mm H={col_height:.0f}mm", "begin",
                  f"  # building offset mm: dx={_offset_for_building(bld)[0]:.2f}, dy={_offset_for_building(bld)[1]:.2f}"]
        _emit_points(f"pts_col_{col_idx}", coords, z_base, bld)
        lines += [f"  grp_col_{col_idx} = {col_entities}.add_group",
                  f"  face_col_{col_idx} = grp_col_{col_idx}.entities.add_face(pts_col_{col_idx})",
                  f"  if face_col_{col_idx} && face_col_{col_idx}.valid?",
                  f"    face_col_{col_idx}.pushpull({col_height:.0f}.mm)",
                  f"    grp_col_{col_idx}.layer = layer_col",
                  f"    grp_col_{col_idx}.material = mat_col",
                  f"    grp_col_{col_idx}.name = 'COL_{safe}'",
                  "  end", "rescue => e",
                  f"  puts 'Error col {safe}: ' + e.message", "end"]
        col_idx += 1

    # --- FOUNDATIONS ---
    lines += ["", "# === FOUNDATIONS ==="]
    fdn_idx = 0
    for fdn in foundation_regions:
        poly = _get_polygon(getattr(fdn, "real_polygon", None)) or getattr(fdn, "polygon", None)
        if poly is None or poly.is_empty:
            lines.append(f"# SKIP fdn {fdn.symbol}: no polygon")
            continue
        safe = _sanitize(fdn.symbol)
        thk = fdn.depth_mm or 400
        z_bot = -(fdn.depth_below_gl_mm or 1500)
        z_top = z_bot + thk
        w, d = fdn.width_mm or 1000, fdn.depth_mm or 1000
        bld, lvl_name = _context_for_element(fdn, "Foundations")
        fdn_entities = _ensure_container(bld, lvl_name, "Foundations")
        coords = _clean_ring_coords(poly)
        if len(coords) < 3:
            cx, cy = poly.centroid.x, poly.centroid.y
            coords = [(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2),
                      (cx + w / 2, cy + d / 2), (cx - w / 2, cy + d / 2)]
        if len(coords) < 3:
            continue
        lines += ["", f"# FDN {fdn.symbol} Ztop={z_top:.0f}mm Thk={thk:.0f}mm", "begin",
                  f"  # building offset mm: dx={_offset_for_building(bld)[0]:.2f}, dy={_offset_for_building(bld)[1]:.2f}"]
        _emit_points(f"pts_fdn_{fdn_idx}", coords, z_top, bld)
        lines += [f"  grp_fdn_{fdn_idx} = {fdn_entities}.add_group",
                  f"  face_fdn_{fdn_idx} = grp_fdn_{fdn_idx}.entities.add_face(pts_fdn_{fdn_idx})",
                  f"  if face_fdn_{fdn_idx} && face_fdn_{fdn_idx}.valid?",
                  f"    face_fdn_{fdn_idx}.pushpull(-{thk:.0f}.mm)",
                  f"    grp_fdn_{fdn_idx}.layer = layer_fdn",
                  f"    grp_fdn_{fdn_idx}.material = mat_fdn",
                  f"    grp_fdn_{fdn_idx}.name = 'FDN_{safe}'",
                  "  end", "rescue => e",
                  f"  puts 'Error fdn {safe}: ' + e.message", "end"]
        fdn_idx += 1

    # --- WALLS ---
    lines += ["", "# === WALLS ==="]
    wall_idx = 0
    for wall in wall_regions:
        poly = _get_polygon(getattr(wall, "real_polygon", None)) or getattr(wall, "polygon", None)
        if poly is None or poly.is_empty:
            lines.append(f"# SKIP wall {getattr(wall, 'label', '?')}: no polygon")
            continue
        z_base = page_primary_ffl.get(getattr(wall, "page_index", -1), 0) * 1000
        wall_height = (
            getattr(wall, "height_mm", 0)
            or storey_height_by_page_mm.get(getattr(wall, "page_index", -1))
            or column_height_mm
        )
        safe = _sanitize(getattr(wall, "label", "") or f"wall_{wall_idx}")
        bld, lvl_name = _context_for_element(wall, "Walls")
        wall_entities = _ensure_container(bld, lvl_name, "Walls")
        coords = _clean_ring_coords(poly)
        if len(coords) < 3:
            continue
        lines += ["", f"# Wall {wall_idx + 1}: {safe} Z={z_base:.0f}mm H={wall_height:.0f}mm", "begin",
                  f"  # building offset mm: dx={_offset_for_building(bld)[0]:.2f}, dy={_offset_for_building(bld)[1]:.2f}"]
        _emit_points(f"pts_wall_{wall_idx}", coords, z_base, bld)
        lines += [f"  grp_wall_{wall_idx} = {wall_entities}.add_group",
                  f"  face_wall_{wall_idx} = grp_wall_{wall_idx}.entities.add_face(pts_wall_{wall_idx})",
                  f"  if face_wall_{wall_idx} && face_wall_{wall_idx}.valid?",
                  f"    face_wall_{wall_idx}.pushpull({wall_height:.0f}.mm)",
                  f"    grp_wall_{wall_idx}.layer = layer_wall",
                  f"    grp_wall_{wall_idx}.material = mat_wall",
                  f"    grp_wall_{wall_idx}.name = 'WALL_{safe}'",
                  "  end", "rescue => e",
                  f"  puts 'Error wall {safe}: ' + e.message", "end"]
        wall_idx += 1

    # Finalize
    lines += ["", "# --- Finalize ---",
              "model.commit_operation",
              "Sketchup.active_model.active_view.zoom_extents",
              f"puts 'Done! {total} elements imported.'"]

    script = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script)
    return script


def generate_columns_ruby(column_regions, output_path, storey_heights=None, height_map=None):
    """Debug/export helper for columns only."""
    return generate_full_ruby_script(
        slab_regions=[],
        column_regions=column_regions,
        foundation_regions=[],
        output_path=output_path,
        generated_by="Feeldx Columns Export",
    )


def generate_foundations_ruby(foundation_regions, output_path):
    """Debug/export helper for foundations only."""
    return generate_full_ruby_script(
        slab_regions=[],
        column_regions=[],
        foundation_regions=foundation_regions,
        output_path=output_path,
        generated_by="Feeldx Foundations Export",
    )


def generate_columns_csv(column_regions, output_path):
    rows = [{
        "ID": c.id,
        "Symbol": c.symbol,
        "Family": getattr(c, "family", ""),
        "Status": getattr(c, "status", ""),
        "Building": c.building,
        "Level": c.level,
        "Width_mm": f"{c.width_mm:.0f}" if c.width_mm else "",
        "Depth_mm": f"{c.depth_mm:.0f}" if c.depth_mm else "",
        "Height_mm": f"{getattr(c, 'height_mm', 0):.0f}" if getattr(c, "height_mm", 0) else "",
        "Page": c.page_index + 1,
        "Confidence": getattr(c, "detection_confidence", ""),
    } for c in column_regions]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return output_path


def _compute_ffl_height_map(slab_regions, census=None) -> dict:
    """Compatibility helper used by app.py; returns symbol height overrides when known."""
    return {}
