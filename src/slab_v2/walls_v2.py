"""
Wall detection v2 — census-aware text-anchor-then-shape.

Pass 1: find text labels matching wall symbols on the page, search for
long rectangular shapes near each label, assign symbol precisely using
census thickness for size-matching.

Pass 2: WALL-class face fallback (wall_extract.py logic) for unlabeled
walls, skipping polygons already claimed by Pass 1.

Census cross-check compares detected counts vs expected per-floor counts.
"""

from __future__ import annotations

import math
from collections import defaultdict

import fitz
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import (
    ClassElection, FaceGraph, WallFootprint, WallType,
)

PT_TO_MM = 25.4 / 72.0

_MIN_ASPECT_RATIO = 3.0
_MERGE_BUFFER_PT = 1.5
_MERGE_DEBUFFER_PT = 1.2
_MIN_AREA_FRAC = 0.0002
_MAX_AREA_FRAC = 0.15
_LABEL_RADIUS_PT = 80.0

_CHAIN_GAP_PT = 20.0
_CHAIN_ANGLE_TOL = math.radians(15)
_CHAIN_THK_TOL_MM = 30.0


def _mrr_sides(poly: Polygon) -> tuple[float, float]:
    """Return (short_side, long_side) of the minimum rotated rectangle."""
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    sides = []
    for i in range(4):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        sides.append(math.hypot(dx, dy))
    sides.sort()
    return sides[0], sides[2]


def _thickness_match(
    short_mm: float,
    wt: WallType,
    tol_mm: float = 60.0,
) -> bool:
    """Check if a rectangle's short side matches the census thickness."""
    if wt.thickness_mm <= 0:
        return True
    return abs(short_mm - wt.thickness_mm) <= tol_mm


def _is_wall_shaped(short_pt: float, long_pt: float) -> bool:
    """Check aspect ratio >= 3:1 and minimum short side."""
    if short_pt < 0.5:
        return False
    return long_pt / short_pt >= _MIN_ASPECT_RATIO


def _wall_axis_angle(poly: Polygon) -> float:
    """Return wall's long-axis angle in radians [0, pi)."""
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    s1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    s2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    if s1 >= s2:
        dx, dy = coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]
    else:
        dx, dy = coords[2][0] - coords[1][0], coords[2][1] - coords[1][1]
    return math.atan2(dy, dx) % math.pi


def _angles_close(a: float, b: float, tol: float) -> bool:
    """Check if two angles in [0, pi) are within tolerance (wrap-aware)."""
    diff = abs(a - b)
    return diff < tol or abs(diff - math.pi) < tol


def _chain_adjacent(
    pass1_walls: list[WallFootprint],
    claimed: set[int],
    all_cands: list[tuple],
    cand_tree: STRtree,
    to_mm: float,
) -> tuple[list[WallFootprint], set[int]]:
    """Expand each Pass 1 wall by chaining adjacent unclaimed segments
    that share the same thickness and axis direction."""
    for wall in pass1_walls:
        seed_angle = _wall_axis_angle(wall.polygon)
        seed_short, _ = _mrr_sides(wall.polygon)
        seed_thk_mm = seed_short * to_mm
        merged_poly = wall.polygon

        changed = True
        while changed:
            changed = False
            search_buf = merged_poly.buffer(_CHAIN_GAP_PT)
            hits = cand_tree.query(search_buf)
            for idx in hits:
                idx = int(idx)
                if idx in claimed:
                    continue
                cpoly, c_short_mm, _ = all_cands[idx]
                if abs(c_short_mm - seed_thk_mm) > _CHAIN_THK_TOL_MM:
                    continue
                c_angle = _wall_axis_angle(cpoly)
                if not _angles_close(seed_angle, c_angle, _CHAIN_ANGLE_TOL):
                    continue
                if merged_poly.distance(cpoly) > _CHAIN_GAP_PT:
                    continue
                merged_poly = unary_union([merged_poly, cpoly])
                claimed.add(idx)
                changed = True

        if not merged_poly.equals(wall.polygon):
            # bridge micro-gaps: buffer out then back in
            if merged_poly.geom_type == "MultiPolygon":
                merged_poly = merged_poly.buffer(
                    _CHAIN_GAP_PT / 2).buffer(-_CHAIN_GAP_PT / 2)
            if merged_poly.geom_type == "MultiPolygon":
                merged_poly = max(merged_poly.geoms, key=lambda g: g.area)
            wall.polygon = merged_poly
            short_pt, long_pt = _mrr_sides(merged_poly)
            wall.w_mm = round(short_pt * to_mm, 1)
            wall.l_mm = round(long_pt * to_mm, 1)

    return pass1_walls, claimed


def _classify_wall_category(label: str, wt: WallType | None) -> str:
    """Determine wall_type string for WallFootprint."""
    if wt and wt.wall_category:
        return wt.wall_category
    upper = label.upper()
    if upper.startswith("SW") or "SHEAR" in upper:
        return "shear_wall"
    if upper.startswith("RW") or "RETAINING" in upper:
        return "retaining_wall"
    if "CORE" in upper:
        return "core_wall"
    return "wall"


def extract_walls_v2(
    page: fitz.Page,
    paths: list,
    slab_union,
    scale: float,
    wall_types: dict[str, WallType],
    cfg: SlabV2Config,
    fg_all: FaceGraph | None = None,
    election: ClassElection | None = None,
    elements: list | None = None,
    column_polys: list | None = None,
    walls_per_floor_census: dict[str, int] | None = None,
    classes: list | None = None,
) -> tuple[list[WallFootprint], list[str]]:
    """Census-aware wall detection with WALL-class face fallback."""
    warnings: list[str] = []
    if not scale:
        return [], ["wall detection skipped: no scale"]
    to_mm = PT_TO_MM * scale

    openings = unary_union([e.polygon for e in elements]) \
        if elements else None
    col_union = unary_union(column_polys) if column_polys else None

    # ── build all long rectangular candidates from vector paths ─────────
    from src.slab_v2.columns import _path_polygon, _rect_sides_pt, _normalize_label
    all_cands = []  # (poly, short_mm, long_mm)
    for p in paths:
        if p.outside_content:
            continue
        if classes and 0 <= p.style_id < len(classes):
            if classes[p.style_id].key.dashes:
                continue
        poly = _path_polygon(p)
        if poly is None:
            continue
        sides = _rect_sides_pt(poly)
        if sides is None:
            continue
        short_pt, long_pt = sorted(sides)
        short_mm, long_mm = short_pt * to_mm, long_pt * to_mm
        if not _is_wall_shaped(short_pt, long_pt):
            continue
        if short_mm < 50.0 or short_mm > 600.0:
            continue
        all_cands.append((poly, short_mm, long_mm))

    # ── PASS 1: text-anchor-then-shape (census-guided) ──────────────────
    # only search for symbols expected on this floor (avoids false positives
    # from legend text like "BW1" on pages where that wall doesn't exist)
    if walls_per_floor_census:
        symbols_norm = {_normalize_label(s): s for s in wall_types
                        if s in walls_per_floor_census}
    else:
        symbols_norm = {_normalize_label(s): s for s in wall_types}
    text_anchors = []  # (symbol, cx, cy)
    if symbols_norm:
        slab_buffer = slab_union.buffer(80) if slab_union is not None else None
        for w in page.get_text("words"):
            txt = _normalize_label(w[4])
            if txt in symbols_norm:
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                if slab_buffer is not None and \
                        not slab_buffer.contains(Point(cx, cy)):
                    continue
                text_anchors.append((symbols_norm[txt], cx, cy))

    claimed: set[int] = set()
    pass1_walls: list[WallFootprint] = []
    cand_tree = None

    if all_cands and text_anchors:
        cand_geoms = [c[0] for c in all_cands]
        cand_tree = STRtree(cand_geoms)

        for sym, cx, cy in text_anchors:
            wt = wall_types.get(sym)
            if wt is None:
                continue
            search_box = Point(cx, cy).buffer(_LABEL_RADIUS_PT)
            hits = cand_tree.query(search_box)
            best_idx, best_dist = None, _LABEL_RADIUS_PT + 1
            for idx in hits:
                idx = int(idx)
                if idx in claimed:
                    continue
                poly, short_mm, long_mm = all_cands[idx]
                if not _thickness_match(short_mm, wt):
                    continue
                dist = poly.distance(Point(cx, cy))
                if dist < best_dist:
                    best_idx, best_dist = idx, dist
            if best_idx is not None:
                poly, short_mm, long_mm = all_cands[best_idx]
                claimed.add(best_idx)
                wall_cat = _classify_wall_category(sym, wt)
                pass1_walls.append(WallFootprint(
                    label=sym, polygon=poly,
                    w_mm=round(short_mm, 1), l_mm=round(long_mm, 1),
                    wall_type=wall_cat))

    # chain adjacent segments with matching thickness + axis
    if pass1_walls and cand_tree is not None:
        pass1_walls, claimed = _chain_adjacent(
            pass1_walls, claimed, all_cands, cand_tree, to_mm)

    # dedupe pass1 (same polygon claimed by multiple nearby text anchors)
    if len(pass1_walls) > 1:
        p1_geoms = [w.polygon for w in pass1_walls]
        p1_tree = STRtree(p1_geoms)
        p1_keep = set()
        for i, wall in enumerate(pass1_walls):
            dup = False
            for j in p1_tree.query(wall.polygon):
                j = int(j)
                if j == i or j not in p1_keep:
                    continue
                inter = wall.polygon.intersection(p1_geoms[j]).area
                union = wall.polygon.area + p1_geoms[j].area - inter
                if union > 0 and inter / union > 0.5:
                    dup = True
                    break
            if not dup:
                p1_keep.add(i)
        pass1_walls = [pass1_walls[i] for i in sorted(p1_keep)]

    # ── PASS 2: WALL-class face fallback ────────────────────────────────
    pass2_walls: list[WallFootprint] = []
    if fg_all is not None and election is not None:
        wall_class_ids = {
            cid for cid, role in election.roles.items() if role == "WALL"
        }
        if wall_class_ids:
            content_area = page.rect.width * page.rect.height
            min_area = _MIN_AREA_FRAC * content_area
            max_area = _MAX_AREA_FRAC * content_area

            wall_faces = []
            for f in fg_all.faces:
                if not (f.style_ids & wall_class_ids):
                    continue
                if f.area_pt2 < min_area or f.area_pt2 > max_area:
                    continue
                wall_faces.append(f)

            if col_union is not None:
                wall_faces = [
                    f for f in wall_faces
                    if not col_union.contains(
                        f.polygon.representative_point())
                ]

            # exclude faces already claimed by pass1
            if pass1_walls and wall_faces:
                p1_union = unary_union([w.polygon for w in pass1_walls])
                wall_faces = [
                    f for f in wall_faces
                    if not p1_union.contains(
                        f.polygon.representative_point())
                ]

            if wall_faces:
                merged = unary_union(
                    [f.polygon.buffer(_MERGE_BUFFER_PT) for f in wall_faces]
                ).buffer(-_MERGE_DEBUFFER_PT)

                parts = []
                for g in getattr(merged, "geoms", [merged]):
                    if g.is_empty or g.area < min_area:
                        continue
                    parts.append(g)

                # filter by aspect ratio + exclude pass1 duplicates
                p1_union = unary_union([w.polygon for w in pass1_walls]) \
                    if pass1_walls else None
                unlabeled_idx = 0
                for poly in parts:
                    short_pt, long_pt = _mrr_sides(poly)
                    if not _is_wall_shaped(short_pt, long_pt):
                        continue
                    short_mm_p2 = short_pt * to_mm
                    if short_mm_p2 < 50.0 or short_mm_p2 > 600.0:
                        continue
                    if p1_union is not None and p1_union.contains(
                            poly.representative_point()):
                        continue
                    short_mm = short_pt * to_mm
                    long_mm = long_pt * to_mm

                    # try matching to census by thickness (floor-filtered)
                    matched_sym = None
                    for sym, wt in wall_types.items():
                        if walls_per_floor_census and sym not in walls_per_floor_census:
                            continue
                        if wt.thickness_mm <= 0:
                            continue
                        if _thickness_match(short_mm, wt):
                            matched_sym = sym
                            break

                    if matched_sym:
                        label = matched_sym
                        wt = wall_types[matched_sym]
                        wall_cat = _classify_wall_category(label, wt)
                    else:
                        unlabeled_idx += 1
                        label = f"WALL_{unlabeled_idx}"
                        wall_cat = "wall"

                    pass2_walls.append(WallFootprint(
                        label=label, polygon=poly,
                        w_mm=round(short_mm, 1), l_mm=round(long_mm, 1),
                        wall_type=wall_cat))

    walls = pass1_walls + pass2_walls
    if walls_per_floor_census and all_cands and text_anchors:
        detected_labels = {w.label for w in walls}
        existing = [w.polygon for w in walls]
        for sym, cx, cy in text_anchors:
            if sym in detected_labels or sym not in walls_per_floor_census:
                continue
            wt = wall_types.get(sym)
            if wt is None:
                continue
            anchor = Point(cx, cy)
            best = None
            best_dist = 120.0
            for poly, short_mm, long_mm in all_cands:
                if not _thickness_match(short_mm, wt):
                    continue
                dist = poly.distance(anchor)
                if dist >= best_dist:
                    continue
                duplicate = False
                for eg in existing:
                    inter = poly.intersection(eg).area
                    if inter / max(min(poly.area, eg.area), 1e-9) > 0.75:
                        duplicate = True
                        break
                if duplicate:
                    continue
                best = (poly, short_mm, long_mm)
                best_dist = dist
            if best is None:
                continue
            poly, short_mm, long_mm = best
            wall_cat = _classify_wall_category(sym, wt)
            walls.append(WallFootprint(
                label=sym, polygon=poly,
                w_mm=round(short_mm, 1), l_mm=round(long_mm, 1),
                wall_type=wall_cat))
            existing.append(poly)
            detected_labels.add(sym)
            warnings.append(
                f"recovered {sym} from nearest wall candidate at {best_dist:.1f}pt")

    # ── census cross-check ───────────────────────────────────────────────
    if walls_per_floor_census:
        detected: dict[str, int] = defaultdict(int)
        for w in walls:
            detected[w.label] += 1
        for sym, expected in walls_per_floor_census.items():
            got = detected.get(sym, 0)
            if got != expected:
                warnings.append(
                    f"census expects {expected}× {sym}, detected {got}")
        for sym, got in detected.items():
            if sym not in walls_per_floor_census and \
                    not sym.startswith("WALL_"):
                warnings.append(
                    f"detected {got}× {sym} not in census for this floor")

    labeled_count = sum(1 for w in walls
                        if not w.label.startswith("WALL_"))
    if text_anchors and labeled_count:
        warnings.insert(0,
            f"text-anchor pass: {labeled_count} wall(s) identified by "
            f"label, {len(walls) - labeled_count} by face fallback")

    return walls, warnings
