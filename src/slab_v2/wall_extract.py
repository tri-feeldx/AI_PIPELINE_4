"""
Wall extraction from face graph + Gemini WALL-tagged style classes.

ZERO additional Gemini calls — uses the existing Round 1 class election
which already tags style classes with role="WALL". Extracts faces from
those classes, filters by geometry (aspect ratio >= 3:1), merges adjacent
faces (core walls = L/U clusters), assigns text labels, converts dims.

Public API:
    extract_walls(page, fg_all, election, cfg, content_area, scale)
        -> list[WallFootprint]
"""

from __future__ import annotations

import math
import re

import fitz
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ClassElection, FaceGraph, WallFootprint

_WALL_LABEL_RE = re.compile(
    r"^(S?W\d+[A-Z]?|BW\d+|RC\s*WALL|RETAINING\s*WALL|CORE\s*WALL|"
    r"CONCRETE\s*WALL|SHEAR\s*WALL)$",
    re.IGNORECASE,
)

_RETAINING_RE = re.compile(r"RETAINING|RW\d+|BW\d+", re.IGNORECASE)

# geometry thresholds
_MIN_AREA_FRAC = 0.0002
_MAX_AREA_FRAC = 0.15
_MIN_ASPECT_RATIO = 3.0
_MERGE_BUFFER_PT = 1.5
_MERGE_DEBUFFER_PT = 1.2
_LABEL_RADIUS_PT = 60.0


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
    return sides[0], sides[1]


def _classify_wall_type(label: str) -> str:
    if _RETAINING_RE.search(label):
        return "retaining_wall"
    return "wall"


def extract_walls(
    page: fitz.Page,
    fg_all: FaceGraph,
    election: ClassElection,
    cfg: SlabV2Config,
    content_area_pt2: float,
    scale: float | int | None,
    column_polys: list[Polygon] | None = None,
) -> list[WallFootprint]:
    """Extract wall footprints from WALL-tagged faces. CPU-only, ~0.1s/page."""

    wall_class_ids = {
        cid for cid, role in election.roles.items() if role == "WALL"
    }
    if not wall_class_ids:
        return []

    # filter faces belonging to WALL classes
    min_area = _MIN_AREA_FRAC * content_area_pt2
    max_area = _MAX_AREA_FRAC * content_area_pt2
    wall_faces = []
    for f in fg_all.faces:
        if not (f.style_ids & wall_class_ids):
            continue
        if f.area_pt2 < min_area or f.area_pt2 > max_area:
            continue
        wall_faces.append(f)

    if not wall_faces:
        return []

    # exclude faces already claimed as columns
    if column_polys:
        col_union = unary_union(column_polys)
        wall_faces = [
            f for f in wall_faces
            if not col_union.contains(f.polygon.representative_point())
        ]

    if not wall_faces:
        return []

    # merge adjacent wall faces (core walls = many small faces)
    merged = unary_union(
        [f.polygon.buffer(_MERGE_BUFFER_PT) for f in wall_faces]
    ).buffer(-_MERGE_DEBUFFER_PT)

    parts = []
    for g in getattr(merged, "geoms", [merged]):
        if g.is_empty or g.area < min_area:
            continue
        parts.append(g)

    if not parts:
        return []

    # filter by aspect ratio
    wall_polys = []
    for p in parts:
        short, long = _mrr_sides(p)
        if short < 0.1:
            continue
        ratio = long / short
        if ratio >= _MIN_ASPECT_RATIO:
            wall_polys.append(p)

    if not wall_polys:
        return []

    # collect text labels near wall polygons
    words = page.get_text("words")
    label_candidates = []
    for w in words:
        text = w[4].strip()
        if _WALL_LABEL_RE.match(text):
            cx = (w[0] + w[2]) / 2
            cy = (w[1] + w[3]) / 2
            label_candidates.append((text, Point(cx, cy)))

    # assign labels + convert dimensions
    walls: list[WallFootprint] = []
    unlabeled_idx = 0
    scale_factor = (scale or 100) * 25.4 / 72.0

    for poly in wall_polys:
        best_label = ""
        best_dist = _LABEL_RADIUS_PT
        for text, pt in label_candidates:
            d = poly.distance(pt)
            if d < best_dist:
                best_dist = d
                best_label = text

        if not best_label:
            unlabeled_idx += 1
            best_label = f"WALL_{unlabeled_idx}"

        short_pt, long_pt = _mrr_sides(poly)
        w_mm = short_pt * scale_factor
        l_mm = long_pt * scale_factor
        wall_type = _classify_wall_type(best_label)

        walls.append(WallFootprint(
            label=best_label,
            polygon=poly,
            w_mm=round(w_mm, 1),
            l_mm=round(l_mm, 1),
            wall_type=wall_type,
        ))

    return walls
