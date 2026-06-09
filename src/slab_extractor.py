"""
Slab boundary extraction from vector PDF paths.

Strategy:
  1. fitz.Page.get_drawings() → all vector paths (lines, rects, curves, filled shapes)
  2. Collect closed filled paths → candidate polygons (with fill color)
  3. Also reconstruct closed polygons by chaining open line segments
  4. Filter by area threshold (>= min_area_pdf_units²)
  5. Dominant-color filter: fill color with largest total area = slab color
  6. Associate with nearest text labels and FFL values
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import fitz
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union

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


@dataclass
class SlabExtractionResult:
    """Structured slab extraction output for gross-slab -> net-slab processing."""
    gross_slabs: list[Polygon] = field(default_factory=list)
    net_slabs: list[Polygon] = field(default_factory=list)
    dominant_fill: Optional[tuple] = None
    appendages: list[Polygon] = field(default_factory=list)
    ignored_regions: list[dict] = field(default_factory=list)
    void_candidates: list[dict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


# ── Color helpers ──────────────────────────────────────────────────────────────

def _round_color(c: tuple, step: float = 0.05) -> tuple:
    """Round RGB values to nearest step to cluster similar shades."""
    return tuple(round(v / step) * step for v in c)


def _color_close(c1, c2, tol: float = 0.08) -> bool:
    """True if two RGB tuples are within tol of each other on all channels."""
    if c1 is None or c2 is None:
        return False
    return all(abs(a - b) <= tol for a, b in zip(c1, c2))


# ── Path extraction ────────────────────────────────────────────────────────────

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


def build_polygons_from_drawings(
    drawings: list[dict], min_pts: int = 4
) -> list[tuple[Polygon, Optional[tuple]]]:
    """
    Convert fitz drawing items to (Shapely Polygon, fill_color) pairs.
    fill_color is an RGB tuple (r,g,b) with values 0.0–1.0, or None.
    Handles: filled closed paths, rect items, quads.
    """
    polygons = []

    for d in drawings:
        fill_color = d.get("fill")  # RGB tuple or None

        # Direct rect
        if d.get("rect") and not d.get("items"):
            poly = _rect_to_polygon(d["rect"])
            if poly and poly.is_valid and not poly.is_empty:
                polygons.append((poly, fill_color))
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
                poly = _ensure_polygon(poly.buffer(0))  # normalize — fixes self-intersections
                if poly and poly.is_valid and not poly.is_empty and poly.area > 0:
                    polygons.append((poly, fill_color))
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


def _bbox_from_text_block(block: dict) -> Polygon:
    x0, y0, x1, y1 = block["bbox"]
    return box(float(x0), float(y0), float(x1), float(y1))


def _iter_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _merge_to_components(polygons: list[Polygon]) -> list[Polygon]:
    if not polygons:
        return []
    return _iter_polygons(unary_union(polygons))


def _semantic_void_allowlist() -> dict:
    return {
        "cut_keywords": ("STAIR", "LIFT", "CORE", "VOID", "OPENING", "SHAFT", "PENETRATION"),
        "keep_keywords": ("C.J", "P.M.J", "T.M.J", "SETDOWN", "STEP", "THICKNESS", "COLUMN"),
    }


def _text_hits_for_voids(text_blocks: list[dict], page: fitz.Page) -> list[dict]:
    spec = _semantic_void_allowlist()
    hits = []
    for block in text_blocks or []:
        txt = (block.get("text") or "").upper().strip()
        if not txt:
            continue
        if any(k in txt for k in spec["cut_keywords"]):
            try:
                bbox_poly = _bbox_from_text_block(block)
            except Exception:
                continue
            cx = bbox_poly.centroid.x
            cy = bbox_poly.centroid.y
            if cx > page.rect.width * 0.82 or cy > page.rect.height * 0.88:
                continue
            hits.append({"text": txt, "bbox": block["bbox"], "polygon": bbox_poly})
    return hits


def _detect_semantic_void_candidates(
    non_slab_pairs: list[tuple[Polygon, Optional[tuple]]],
    gross_slabs: list[Polygon],
    text_blocks: list[dict],
    page: fitz.Page,
    min_confidence: float = 0.75,
    auto_cut_voids: bool = True,
) -> list[dict]:
    if not non_slab_pairs or not gross_slabs:
        return []

    gross_union = unary_union(gross_slabs)
    text_hits = _text_hits_for_voids(text_blocks, page)
    if not text_hits:
        return []

    page_area = page.rect.width * page.rect.height
    min_area = page_area * 0.00008
    max_area = page_area * 0.08
    search_radius = max(page.rect.width, page.rect.height) * 0.045

    def _is_steelwork_color(color) -> bool:
        if color is None or len(color) < 3:
            return False
        r, g, b = color[:3]
        if r > 0.90 and g > 0.90 and b > 0.90:
            return False
        return b > r + 0.12 and g > 0.45

    candidates = []
    used = set()
    for hit_idx, hit in enumerate(text_hits):
        txt = hit["text"]
        is_stair = "STAIR" in txt
        is_void = any(k in txt for k in ("VOID", "OPENING", "LIFT", "CORE", "SHAFT", "PENETRATION"))
        nearby = []
        for poly_idx, (poly, color) in enumerate(non_slab_pairs):
            if color is None or poly_idx in used:
                continue
            if is_stair and not _is_steelwork_color(color):
                continue
            if poly.is_empty or poly.area > max_area:
                continue
            if poly.distance(hit["polygon"]) > search_radius:
                continue
            try:
                if not gross_union.buffer(2).intersects(poly):
                    continue
            except Exception:
                continue
            nearby.append((poly_idx, poly, color))

        if not nearby:
            continue

        try:
            cluster = unary_union([p for _, p, _ in nearby]).buffer(2).envelope
            cut_poly = cluster.intersection(gross_union).buffer(0)
        except Exception:
            continue
        if cut_poly.is_empty or cut_poly.area < min_area or cut_poly.area > max_area:
            continue

        confidence = 0.62
        if is_stair:
            confidence += 0.23
        if is_void:
            confidence += 0.20
        if len(nearby) >= 3:
            confidence += 0.05
        confidence = min(confidence, 0.95)
        for poly_idx, _, _ in nearby:
            used.add(poly_idx)
        candidates.append({
            "polygon": cut_poly,
            "reason": "stair_steelwork_cluster" if is_stair else "semantic_void_cluster",
            "confidence": confidence,
            "auto_cut": bool(auto_cut_voids and confidence >= min_confidence),
            "text": txt,
            "color": "cluster",
        })

    return candidates


def _filter_slab_candidates_legacy(
    poly_color_pairs: list,   # list[tuple[Polygon, Optional[tuple]]]
    page: fitz.Page,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.95,
) -> list[Polygon]:
    """
    Filter polygons to likely slab regions, then merge into the true floor outline.

    Strategy:
    1. Drop trivially tiny / overly-elongated shapes (dimension lines, thin borders)
    2. Deduplicate overlapping polygons
    3. Dominant-color filter: the fill color covering the largest total area = slab color.
       Keep only polygons of that color (ignores steelwork, walls, column caps etc.)
    4. unary_union all survivors → touching structural bays merge into one floor shape
    5. If the union produces multiple disconnected components, keep only those ≥ 5% of largest
    """
    page_area = page.rect.width * page.rect.height
    min_area = page_area * min_area_fraction
    max_area = page_area * max_area_fraction

    result_pairs = []
    for poly, color in poly_color_pairs:
        area = poly.area
        if area < min_area or area > max_area:
            continue
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        if w < 5 or h < 5:
            continue
        # Convexity ratio: slabs and thin connector strips are near-convex (≥0.55).
        # Stairwells / lift cores are L-shaped or irregular → convexity ≈ 0.4–0.7.
        # This replaces the old aspect-ratio check which incorrectly rejected wide thin
        # connector strips (e.g. 60m × 2m corridor has aspect=30 but convexity=1.0).
        try:
            convexity = poly.convex_hull.area / poly.area
        except Exception:
            convexity = 1.0
        if convexity < 0.55:
            continue
        result_pairs.append((poly, color))

    result_pairs = _deduplicate_pairs(result_pairs)

    if not result_pairs:
        return []

    # Dominant-color filter: pick fill color with largest total area = slab color.
    # Works without reading the legend — slab always dominates the page area.
    color_area: dict = defaultdict(float)
    for poly, color in result_pairs:
        if color is not None:
            key = _round_color(color)
            color_area[key] += poly.area

    if color_area:
        dominant = max(color_area, key=color_area.get)
        color_filtered = [
            p for p, c in result_pairs
            if c is not None and _color_close(_round_color(c), dominant, tol=0.06)
        ]
        if color_filtered:
            get_logger().info(
                f"  Color filter: dominant={dominant}, "
                f"kept {len(color_filtered)}/{len(result_pairs)} polys"
            )
            result = color_filtered
        else:
            result = [p for p, _ in result_pairs]  # fallback: no color match
    else:
        result = [p for p, _ in result_pairs]  # fallback: no fill colors (line drawings)

    # Merge touching structural bays into the complete floor outline.
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
        max_comp_area = max(g.area for g in merged.geoms)
        kept = [g for g in merged.geoms if g.area >= max_comp_area * 0.02]
        if not kept:
            kept = [max(merged.geoms, key=lambda g: g.area)]
        get_logger().info(
            f"  Union merge: {len(result)} polys → {len(merged.geoms)} components → "
            f"kept {len(kept)} (≥5% of {max_comp_area:.0f}pt²)"
        )
        return kept
    else:
        if len(result) > 1:
            get_logger().info(f"  Union merge: {len(result)} polys → 1 component")
        return [merged]


def filter_slab_candidates_structured(
    poly_color_pairs: list,
    page: fitz.Page,
    text_blocks: Optional[list[dict]] = None,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.95,
    recover_slab_appendages: bool = True,
    auto_cut_voids: bool = True,
    cut_walls: bool = False,
    min_void_confidence: float = 0.75,
) -> SlabExtractionResult:
    """
    Gross-slab -> net-slab extraction.

    Gross slab keeps valid dominant-fill regions, including small edge appendages.
    Net slab subtracts only high-confidence semantic void candidates.
    """
    _ = cut_walls
    page_area = page.rect.width * page.rect.height
    min_area = page_area * min_area_fraction
    max_area = page_area * max_area_fraction

    result_pairs = []
    ignored_regions = []
    for poly, color in poly_color_pairs:
        area = poly.area
        if area < min_area or area > max_area:
            ignored_regions.append({"polygon": poly, "reason": "area_filter", "color": color})
            continue
        bounds = poly.bounds
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        if w < 5 or h < 5:
            ignored_regions.append({"polygon": poly, "reason": "thin_filter", "color": color})
            continue
        try:
            convexity = poly.convex_hull.area / poly.area
        except Exception:
            convexity = 1.0
        if convexity < 0.55:
            ignored_regions.append({"polygon": poly, "reason": "convexity_filter", "color": color})
            continue
        result_pairs.append((poly, color))

    result_pairs = _deduplicate_pairs(result_pairs)
    if not result_pairs:
        return SlabExtractionResult(ignored_regions=ignored_regions)

    color_area: dict = defaultdict(float)
    for poly, color in result_pairs:
        if color is not None:
            key = _round_color(color)
            color_area[key] += poly.area

    dominant = None
    non_slab_pairs = []
    if color_area:
        dominant = max(color_area, key=color_area.get)
        slab_pairs = [
            (p, c) for p, c in result_pairs
            if c is not None and _color_close(_round_color(c), dominant, tol=0.06)
        ]
        non_slab_pairs = [
            (p, c) for p, c in result_pairs
            if not (c is not None and _color_close(_round_color(c), dominant, tol=0.06))
        ]
        if not slab_pairs:
            slab_pairs = result_pairs
            non_slab_pairs = []
            dominant = None
    else:
        slab_pairs = result_pairs

    if dominant is not None:
        raw_non_slab_pairs = []
        raw_min_area = page_area * 0.00001
        for poly, color in poly_color_pairs:
            if color is None or poly.area < raw_min_area:
                continue
            if _color_close(_round_color(color), dominant, tol=0.06):
                continue
            cx, cy = poly.centroid.x, poly.centroid.y
            if cx > page.rect.width * 0.82 or cy > page.rect.height * 0.88:
                continue
            raw_non_slab_pairs.append((poly, color))
        if raw_non_slab_pairs:
            non_slab_pairs = raw_non_slab_pairs

    slab_polys = [p for p, _ in slab_pairs]
    try:
        components = _merge_to_components(slab_polys)
    except Exception as e:
        get_logger().warning(f"  gross slab union failed ({type(e).__name__}): {e}")
        largest = max(slab_polys, key=lambda p: p.area)
        return SlabExtractionResult(
            gross_slabs=[largest],
            net_slabs=[largest],
            dominant_fill=dominant,
            ignored_regions=ignored_regions,
            debug={"fallback": "largest_after_gross_union_failure"},
        )

    if not components:
        components = [max(slab_polys, key=lambda p: p.area)]

    largest = max(components, key=lambda g: g.area)
    max_comp_area = max(largest.area, 1.0)
    attach_distance = max(page.rect.width, page.rect.height) * 0.018
    kept = []
    appendages = []
    far_same_fill = []

    for comp in sorted(components, key=lambda g: g.area, reverse=True):
        is_major = comp.area >= max_comp_area * 0.02
        is_attached = recover_slab_appendages and comp.distance(largest) <= attach_distance
        is_useful_appendage = comp.area >= max_comp_area * 0.0015
        if is_major or (is_attached and is_useful_appendage):
            kept.append(comp)
            if not is_major:
                appendages.append(comp)
        else:
            far_same_fill.append(comp)
            ignored_regions.append({
                "polygon": comp,
                "reason": "far_or_tiny_same_fill",
                "color": dominant,
            })

    if not kept:
        kept = [largest]

    gross_slabs = _merge_to_components(kept) or kept
    void_candidates = _detect_semantic_void_candidates(
        non_slab_pairs=non_slab_pairs,
        gross_slabs=gross_slabs,
        text_blocks=text_blocks or [],
        page=page,
        min_confidence=min_void_confidence,
        auto_cut_voids=auto_cut_voids,
    )

    cut_polys = [c["polygon"] for c in void_candidates if c.get("auto_cut")]
    net_slabs = gross_slabs
    if cut_polys:
        try:
            cut_union = unary_union(cut_polys)
            net_geom = unary_union(gross_slabs).difference(cut_union).buffer(0)
            net_slabs = [
                g for g in _iter_polygons(net_geom)
                if g.area >= max_comp_area * 0.001
            ] or gross_slabs
        except Exception as e:
            get_logger().warning(f"  Void subtraction failed ({type(e).__name__}): {e}")
            net_slabs = gross_slabs

    get_logger().info(
        f"  Gross/net slab: dominant={dominant}, gross={len(gross_slabs)}, "
        f"net={len(net_slabs)}, appendages={len(appendages)}, "
        f"void_candidates={len(void_candidates)}, auto_cuts={len(cut_polys)}"
    )

    return SlabExtractionResult(
        gross_slabs=gross_slabs,
        net_slabs=net_slabs,
        dominant_fill=dominant,
        appendages=appendages,
        ignored_regions=ignored_regions,
        void_candidates=void_candidates,
        debug={
            "dominant_pairs": len(slab_pairs),
            "non_slab_pairs": len(non_slab_pairs),
            "components_before_recovery": len(components),
            "components_kept": len(kept),
            "appendages": len(appendages),
            "ignored_same_fill": len(far_same_fill),
            "auto_cut_voids": len(cut_polys),
            "semantic_spec_source": "default_allowlist",
            "cut_walls": False,
        },
    )


def filter_slab_candidates(
    poly_color_pairs: list,
    page: fitz.Page,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.95,
) -> list[Polygon]:
    """Compatibility wrapper: return final net slab polygons only."""
    result = filter_slab_candidates_structured(
        poly_color_pairs,
        page,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
    )
    return result.net_slabs


def _deduplicate_pairs(
    pairs: list, overlap_thresh: float = 0.8
) -> list:
    """Remove (Polygon, color) pairs where polygon is nearly identical to or contained in another."""
    if not pairs:
        return []
    sorted_pairs = sorted(pairs, key=lambda pc: pc[0].area, reverse=True)
    kept = []
    for poly, color in sorted_pairs:
        dominated = False
        for kpoly, _ in kept:
            try:
                intersection = poly.intersection(kpoly)
                if intersection.area / max(poly.area, 1) > overlap_thresh:
                    dominated = True
                    break
            except Exception:
                pass
        if not dominated:
            kept.append((poly, color))
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
      drawings → polygons with color → dominant-color filter → label → SlabRegion list
    Returns (slab_regions, raw_drawings) for visualization.
    """
    drawings = extract_paths(page)

    filled_pairs = build_polygons_from_drawings(drawings)        # list[(Polygon, color)]
    recon_pairs  = [(p, None) for p in reconstruct_closed_polygons(drawings)]

    all_pairs = filled_pairs + recon_pairs
    slab_result = filter_slab_candidates_structured(all_pairs, page, text_blocks=text_blocks)
    candidates = slab_result.net_slabs

    log_extraction_counts(page.number, len(filled_pairs), len(recon_pairs), len(candidates))

    if not candidates:
        log_warn(page.number, "No slab candidates after filtering — check scale and page selection")

    regions = assign_labels(candidates, text_blocks, ffl_values, slab_labels)

    for i, r in enumerate(regions):
        r.source = "filled-net" if slab_result.dominant_fill is not None else "reconstructed"
        r.page_index = page.number
        if r.ffl_m is None:
            log_warn(page.number, f"Slab {r.label}: no FFL found → default FFL=0.000m will be used")

    return regions, drawings
