"""
Ruby geometry for element volumes — shaft walls and stair flights.

LIFT / SHAFT / DUCT: the footprint minus an inward buffer becomes a wall
ring extruded over the storey height, so the volume stands exactly on the
opening already cut in the slab. Footprints too narrow for the buffer fall
back to a solid volume (warning).

STAIR: a straight flight approximated as a stepped solid mass — riser count
from the storey height (riser <= cfg.stair_max_riser_mm), run along the
long axis of the footprint's minimum rotated rectangle, full footprint
width. Correct position, elevation, and step count; suggestive geometry,
not a shop drawing.

VOID: no volume — the opening in the slab is the whole story.

All coordinates are real-world mm, page bottom-left origin, Y-up, Z=height
(the convention of src/coordinate_mapper.py and v1 model_builder).
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from src.slab_v2.config import SlabV2Config


def _dedup_consecutive(coords):
    """Remove consecutive duplicate points (SketchUp rejects them).
    Compares at 1-decimal precision to match the .1f formatting."""
    if not coords:
        return coords
    out = [coords[0]]
    for p in coords[1:]:
        if round(p[0], 1) != round(out[-1][0], 1) or \
           round(p[1], 1) != round(out[-1][1], 1):
            out.append(p)
    return out


def ring_to_ruby(coords, z: float = 0.0) -> str:
    clean = _dedup_consecutive(coords)
    pts = ", ".join(f"[{x:.1f}.mm, {y:.1f}.mm, {z:.1f}.mm]"
                    for x, y in clean)
    return f"[{pts}]"


def face_with_holes(poly: Polygon, group_var: str, thickness_mm: float,
                    z: float = 0.0) -> list[str]:
    """Face at height z with inner loops erased, extruded DOWN thickness_mm."""
    lines = []
    ext = list(poly.exterior.coords)[:-1]
    lines.append(
        f"face = {group_var}.entities.add_face({ring_to_ruby(ext, z)})")
    for hole in poly.interiors:
        ring = list(hole.coords)[:-1]
        lines.append(f"hole_face = {group_var}.entities.add_face("
                     f"{ring_to_ruby(ring, z)})")
        lines.append("hole_face.erase! if hole_face && hole_face.valid?")
    lines.append("face.reverse! if face.normal.z < 0")
    lines.append(f"face.pushpull(-{thickness_mm:.1f}.mm) if face.valid?")
    return lines


def _solid_up(poly: Polygon, group_var: str, z_base: float,
              height: float) -> list[str]:
    """Face (with holes) at z_base extruded UP by height."""
    lines = []
    ext = list(poly.exterior.coords)[:-1]
    lines.append(
        f"face = {group_var}.entities.add_face({ring_to_ruby(ext, z_base)})")
    for hole in poly.interiors:
        ring = list(hole.coords)[:-1]
        lines.append(f"hole_face = {group_var}.entities.add_face("
                     f"{ring_to_ruby(ring, z_base)})")
        lines.append("hole_face.erase! if hole_face && hole_face.valid?")
    lines.append("face.reverse! if face.normal.z < 0")
    lines.append(f"face.pushpull({height:.1f}.mm) if face.valid?")
    return lines


def _shaft_ruby(poly: Polygon, z_base: float, height: float,
                cfg: SlabV2Config, var: str) -> tuple[list[str], list[str]]:
    t = cfg.shaft_wall_thickness_mm
    inner = poly.buffer(-t)
    if inner.is_empty or inner.area <= 0:
        return (_solid_up(poly, var, z_base, height),
                [f"shaft footprint too narrow for {t:.0f}mm walls — "
                 f"exported as solid volume"])
    ring = poly.difference(inner)
    lines = []
    for g in getattr(ring, "geoms", [ring]):
        if not g.is_empty:
            lines += _solid_up(g, var, z_base, height)
    return lines, []


def _stair_ruby(poly: Polygon, z_base: float, height: float,
                cfg: SlabV2Config, var: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    mrr = poly.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":          # degenerate footprint
        return _solid_up(poly, var, z_base, height), \
            ["stair footprint degenerate — exported as solid volume"]
    c = list(mrr.exterior.coords)[:4]
    e1 = (c[1][0] - c[0][0], c[1][1] - c[0][1])
    e2 = (c[2][0] - c[1][0], c[2][1] - c[1][1])
    len1, len2 = math.hypot(*e1), math.hypot(*e2)
    if len1 >= len2:
        # run along c0->c1, width toward c2 (= e2)
        origin, run_vec, run_len, wide_vec = c[0], e1, len1, e2
    else:
        # run along c1->c2, width back toward c0 (= -e1)
        origin, run_vec, run_len = c[1], e2, len2
        wide_vec = (-e1[0], -e1[1])
    if run_len < 1.0 or height <= 0:
        return _solid_up(poly, var, z_base, height), \
            ["stair footprint degenerate — exported as solid volume"]
    ux, uy = run_vec[0] / run_len, run_vec[1] / run_len

    n = max(1, math.ceil(height / cfg.stair_max_riser_mm))
    going = run_len / n
    if going < cfg.stair_min_going_mm:
        n_fit = max(1, math.floor(run_len / cfg.stair_min_going_mm))
        if n_fit < 3 and n_fit < n:
            # footprint can't carry a recognizable flight (small stair
            # penetration / mislabeled void) — a solid mass reads better
            # than one or two giant steps
            return _solid_up(poly, var, z_base, height), \
                [f"stair run {run_len:.0f}mm too short for a flight — "
                 f"exported as solid volume"]
        warnings.append(
            f"stair run {run_len:.0f}mm too short for {n} steps "
            f"(going {going:.0f}mm < {cfg.stair_min_going_mm:.0f}mm) — "
            f"reduced to {n_fit} steps")
        n = n_fit
        going = run_len / n
    riser = height / n

    lines = [f"# stair flight: {n} steps, riser {riser:.0f}mm, "
             f"going {going:.0f}mm (suggestive mass, not a shop drawing)"]
    for i in range(n):
        a = (origin[0] + ux * going * i, origin[1] + uy * going * i)
        b = (origin[0] + ux * going * (i + 1),
             origin[1] + uy * going * (i + 1))
        step = [a, b, (b[0] + wide_vec[0], b[1] + wide_vec[1]),
                (a[0] + wide_vec[0], a[1] + wide_vec[1])]
        z_top = z_base + riser * (i + 1)
        depth = riser * (i + 1)
        lines.append(f"face = {var}.entities.add_face("
                     f"{ring_to_ruby(step, z_top)})")
        lines.append("face.reverse! if face.normal.z < 0")
        lines.append(f"face.pushpull(-{depth:.1f}.mm) if face.valid?")
    return lines, warnings


def element_ruby(etype: str, poly_mm: Polygon, z_base_mm: float,
                 height_mm: float, cfg: SlabV2Config,
                 var: str) -> tuple[list[str], list[str]]:
    """Ruby lines for one element volume. Returns (lines, warnings)."""
    if poly_mm.is_empty:
        return [], []
    if etype == "VOID":
        return [], []
    if etype == "STAIR":
        return _stair_ruby(poly_mm, z_base_mm, height_mm, cfg, var)
    return _shaft_ruby(poly_mm, z_base_mm, height_mm, cfg, var)
