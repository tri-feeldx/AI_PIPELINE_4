"""
Slab boundary extraction from vector PDF paths.

Strategy:
  1. fitz.Page.get_drawings() → all vector paths (lines, rects, curves, filled shapes)
  2. Collect closed filled paths → candidate polygons
  3. Also reconstruct closed polygons by chaining open line segments
  4. Filter by area threshold (>= min_area_pdf_units²)
  5. Associate with nearest text labels and FFL values
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import fitz
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import numpy as np

from src.pipeline_logger import log_extraction_counts, log_warn, get_logger


def _ensure_polygon(geom) -> Optional[Polygon]:
    """Return a simple Polygon from any geometry, taking largest part if Multi."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return None


@dataclass
class SlabRegion:
    id: int
    polygon: Polygon          # in PDF coordinate space (points)
    label: str = ""
    ffl_m: Optional[float] = None
    ffl_mm: Optional[float] = None
    area_pdf: float = 0.0     # area in PDF point² units
    area_m2: float = 0.0      # area in real-world m² (set after coordinate mapping)
    page_index: int = 0
    source: str = "filled"    # "filled" | "reconstructed"
    color: tuple = field(default_factory=lambda: (0.3, 0.7, 1.0, 0.4))


def extract_paths(page: fitz.Page) -> list[dict]:
    """Return all vector drawing items from the page."""
    return page.get_drawings()


def _pts_from_items(items: list) -> list[tuple[float, float]]:
    """Extract all point coordinates from fitz path items."""
    pts = []
    for item in items:
        kind = item[0]
        if kind == "l":   # line: ("l", p1, p2)
            pts.extend([item[1], item[2]])
        elif kind == "c": # curve: ("c", p1, p2, p3, p4)
            pts.extend([item[1], item[4]])
        elif kind == "re": # rect: ("re", rect)
            r = item[1]
            pts.extend([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
        elif kind == "qu": # quad
            q = item[1]
            pts.extend([q.ul, q.ur, q.lr, q.ll])
    return [(float(p.x), float(p.y)) if hasattr(p, "x") else (float(p[0]), float(p[1])) for p in pts]


def _is_closed(items: list, tol: float = 1.0) -> bool:
    """Check if path items form a closed loop."""
    if not items:
        return False
    pts = _pts_from_items(items)
    if len(pts) < 3:
        return False
    first, last = pts[0], pts[-1]
    return math.dist(first, last) < tol


def _rect_to_polygon(rect: fitz.Rect) -> Optional[Polygon]:
    if rect.is_empty or rect.is_infinite:
        return None
    return Polygon([
        (rect.x0, rect.y0), (rect.x1, rect.y0),
        (rect.x1, rect.y1), (rect.x0, rect.y1),
    ])


def build_polygons_from_drawings(drawings: list[dict], min_pts: int = 4) -> list[Polygon]:
    """
    Convert fitz drawing items to Shapely Polygons.
    Handles: filled closed paths, rect items, quads.
    """
    polygons = []

    for d in drawings:
        # Direct rect
        if d.get("rect") and not d.get("items"):
            poly = _rect_to_polygon(d["rect"])
            if poly and poly.is_valid and not poly.is_empty:
                polygons.append(poly)
            continue

        items = d.get("items", [])
        if not items:
            continue

        # Collect all coordinates
        pts = _pts_from_items(items)
        # Deduplicate consecutive duplicates
        unique_pts = [pts[0]] if pts else []
        for p in pts[1:]:
            if math.dist(p, unique_pts[-1]) > 0.5:
                unique_pts.append(p)

        if len(unique_pts) < min_pts:
            continue

        # Only keep closed paths (start ≈ end) or filled shapes
        is_filled = bool(d.get("fill"))
        closed = _is_closed(items)

        if is_filled or closed:
            try:
                poly = Polygon(unique_pts)
                poly = _ensure_polygon(poly.buffer(0))  # always normalize — fixes subtle self-intersections
                if poly and poly.is_valid and not poly.is_empty and poly.area > 0:
                    polygons.append(poly)
            except Exception:
                pass

    return polygons


def _segment_endpoints(drawings: list[dict]) -> list[tuple]:
    """Extract all line segment endpoints (p1, p2) for chain reconstruction."""
    segments = []
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                p1 = (float(item[1].x), float(item[1].y))
                p2 = (float(item[2].x), float(item[2].y))
                if math.dist(p1, p2) > 1.0:
                    segments.append((p1, p2))
    return segments


def reconstruct_closed_polygons(drawings: list[dict], tol: float = 2.0) -> list[Polygon]:
    """
    Chain line segments into closed polygons (for structural outline drawings
    where the slab boundary is drawn with individual lines, not filled paths).
    Uses a greedy endpoint-matching algorithm.
    """
    segments = _segment_endpoints(drawings)
    if not segments:
        return []

    used = [False] * len(segments)
    polygons = []

    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue
        chain = list(segments[start_idx])
        used[start_idx] = True
        current_end = chain[-1]

        for _ in range(len(segments)):
            found = False
            for j, seg in enumerate(segments):
                if used[j]:
                    continue
                p1, p2 = seg
                if math.dist(current_end, p1) < tol:
                    chain.append(p2)
                    current_end = p2
                    used[j] = True
                    found = True
                    break
                elif math.dist(current_end, p2) < tol:
                    chain.append(p1)
                    current_end = p1
                    used[j] = True
                    found = True
                    break
            if not found:
                break

        # Check if the chain closes back to start
        if len(chain) >= 4 and math.dist(chain[0], chain[-1]) < tol * 2:
            try:
                poly = Polygon(chain)
                if not poly.is_valid:
                    fixed = _ensure_polygon(poly.buffer(0))
                    if fixed and fixed.is_valid and not fixed.is_empty:
                        get_logger().info("Reconstructed polygon invalid → buffer(0) applied → fixed")
                        poly = fixed
                    else:
                        get_logger().warning("Reconstructed polygon invalid → could not fix → dropped")
                        poly = None
                else:
                    poly = _ensure_polygon(poly)
                if poly and poly.is_valid and not poly.is_empty and poly.area > 100:
                    polygons.append(poly)
            except Exception:
                pass

    return polygons


def filter_slab_candidates(
    polygons: list[Polygon],
    page: fitz.Page,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.95,
) -> list[Polygon]:
    """
    Filter polygons to likely slab regions, then merge into the true floor outline.

    Strategy:
    1. Drop trivially tiny / overly-elongated shapes (dimension lines, thin borders)
    2. Deduplicate overlapping polygons
    3. unary_union all survivors → touching structural bays merge into one floor shape
    4. If the union produces multiple disconnected components (e.g. main slab + legend panel),
       keep only those ≥ 10% of the largest component area
    """
    page_area = page.rect.width * page.rect.height
    min_area = page_area * min_area_fraction
    max_area = page_area * max_area_fraction

    result = []
    for poly in polygons:
        area = poly.area
        if area < min_area or area > max_area:
            continue
        # Exclude very elongated shapes (lines disguised as thin rects)
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        if w < 5 or h < 5:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 15:
            continue
        result.append(poly)

    # Remove duplicates / heavily overlapping polygons
    result = _deduplicate_polygons(result)

    if not result:
        return []

    # Merge touching structural bays into the complete floor outline.
    # Each floor plan page represents ONE floor; if we detect N>1 polygons they are
    # structural bay fills that must be unioned to recover the true slab boundary.
    try:
        merged = unary_union(result)
    except Exception as e:
        get_logger().warning(
            f"  unary_union failed ({type(e).__name__}): {e} "
            f"— falling back to largest polygon"
        )
        return [max(result, key=lambda p: p.area)]

    if merged is None or merged.is_empty:
        return [max(result, key=lambda p: p.area)]

    if isinstance(merged, MultiPolygon):
        # Multiple disconnected components — keep only significant ones.
        # Small components (title block boxes, legend panels) are < 5% of main floor.
        max_comp_area = max(g.area for g in merged.geoms)
        kept = [g for g in merged.geoms if g.area >= max_comp_area * 0.05]
        if not kept:
            kept = [max(merged.geoms, key=lambda g: g.area)]
        get_logger().info(
            f"  Union merge: {len(result)} polys → {len(merged.geoms)} components → "
            f"kept {len(kept)} (≥5% of {max_comp_area:.0f}pt²)"
        )
        return kept
    else:
        if len(result) > 1:
            get_logger().info(
                f"  Union merge: {len(result)} polys → 1 component"
            )
        return [merged]


def _deduplicate_polygons(polygons: list[Polygon], overlap_thresh: float = 0.8) -> list[Polygon]:
    """Remove polygons that are nearly identical or contained within another."""
    if not polygons:
        return []
    # Sort by area descending
    sorted_polys = sorted(polygons, key=lambda p: p.area, reverse=True)
    kept = []
    for poly in sorted_polys:
        dominated = False
        for keeper in kept:
            try:
                intersection = poly.intersection(keeper)
                if intersection.area / max(poly.area, 1) > overlap_thresh:
                    dominated = True
                    break
            except Exception:
                pass
        if not dominated:
            kept.append(poly)
    return kept


def assign_labels(
    candidates: list[Polygon],
    text_blocks: list[dict],
    ffl_values: list[dict],
    slab_labels: list[dict],
) -> list[SlabRegion]:
    """
    For each candidate polygon, find:
    - The nearest slab label text whose center falls inside or near the polygon
    - The nearest FFL value
    """
    regions = []

    # Build centroid points for text items
    def bbox_center(bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    for idx, poly in enumerate(candidates):
        centroid = poly.centroid

        # Find label: prefer label whose bbox center is inside polygon
        best_label = ""
        best_label_dist = float("inf")
        for sl in slab_labels:
            cx, cy = bbox_center(sl["bbox"])
            pt = Point(cx, cy)
            if poly.contains(pt):
                best_label = sl["label"]
                best_label_dist = 0
                break
            else:
                d = centroid.distance(pt)
                if d < best_label_dist:
                    best_label_dist = d
                    best_label = sl["label"]

        # Only assign label if it's reasonably close
        if best_label_dist > poly.length:
            best_label = f"S{idx + 1}"

        # Find FFL: nearest to polygon centroid
        best_ffl = None
        best_ffl_dist = float("inf")
        for ffl in ffl_values:
            cx, cy = bbox_center(ffl["bbox"])
            d = centroid.distance(Point(cx, cy))
            if d < best_ffl_dist:
                best_ffl_dist = d
                best_ffl = ffl

        region = SlabRegion(
            id=idx,
            polygon=poly,
            label=best_label or f"S{idx + 1}",
            ffl_m=best_ffl["ffl_m"] if best_ffl else None,
            ffl_mm=best_ffl["ffl_mm"] if best_ffl else None,
            area_pdf=poly.area,
        )
        regions.append(region)

    return regions


def extract_slabs_from_page(
    page: fitz.Page,
    text_blocks: list[dict],
    ffl_values: list[dict],
    slab_labels: list[dict],
) -> tuple[list[SlabRegion], list[dict]]:
    """
    Full pipeline for one page:
      drawings → polygons (filled + reconstructed) → filter → label → SlabRegion list
    Returns (slab_regions, raw_drawings) for visualization.
    """
    drawings = extract_paths(page)

    filled_polys = build_polygons_from_drawings(drawings)
    reconstructed_polys = reconstruct_closed_polygons(drawings)

    all_polys = filled_polys + reconstructed_polys
    candidates = filter_slab_candidates(all_polys, page)

    log_extraction_counts(page.number, len(filled_polys), len(reconstructed_polys), len(candidates))

    if not candidates:
        log_warn(page.number, "No slab candidates after filtering — check scale and page selection")

    regions = assign_labels(candidates, text_blocks, ffl_values, slab_labels)

    # Tag source and log FFL warnings
    n_filled = len(filled_polys)
    for i, r in enumerate(regions):
        r.source = "filled" if i < n_filled else "reconstructed"
        r.page_index = page.number
        if r.ffl_m is None:
            log_warn(page.number, f"Slab {r.label}: no FFL found → default FFL=0.000m will be used")

    return regions, drawings
