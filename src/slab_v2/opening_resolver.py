"""Resolve and semantically judge stair/core/shaft opening candidates.

Geometry is always produced by PDF vectors.  Gemini may choose candidate IDs,
but never supplies coordinates.  A deterministic decision remains available
when the judge fails or returns low confidence.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field

import fitz
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.slab_v2.models import (ElementFootprint, OpeningIntent,
                                ResolvedPenetration)

PT_TO_MM = 25.4 / 72.0

_STAIR_RE = re.compile(r"\bSTAIRS?\b|\bST[- ]?\d{1,2}\b", re.I)
_CORE_RE = re.compile(r"\b(LIFT|ELEV|SHAFT|CORE|LV ?\d{1,2})\b", re.I)
_EQUIPMENT_RE = re.compile(r"\b(FB|FLOOR\s*BOX)\b", re.I)
_STEEL_LABEL_RE = re.compile(r"^(?:SH|CH|UC|UB|SHS|CHS|RHS)\w*$", re.I)


@dataclass
class OpeningResolution:
    resolved_openings: list[ElementFootprint] = field(default_factory=list)
    verified_cut_openings: list[ElementFootprint] = field(default_factory=list)
    context_objects: list[ElementFootprint] = field(default_factory=list)
    review_candidates: list[dict] = field(default_factory=list)
    stair_footprints: list[ElementFootprint] = field(default_factory=list)
    core_shaft_footprints: list[ElementFootprint] = field(default_factory=list)
    resolved_penetrations: list[ResolvedPenetration] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    judgement: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)


def _word_anchors(page: fitz.Page, rx: re.Pattern, content_rect: fitz.Rect):
    anchors = []
    words = list(page.get_text("words"))
    for i, w in enumerate(words):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not content_rect.contains(fitz.Point(cx, cy)):
            continue
        if not rx.search(str(text)):
            continue
        label = str(text)
        if label.upper() == "STAIR" and i + 1 < len(words):
            nxt = str(words[i + 1][4])
            if re.fullmatch(r"\d{1,2}", nxt):
                label = f"{label} {nxt}"
        anchors.append((label, (x0, y0, x1, y1), Point(cx, cy)))
    return anchors


def _nearby_text(page: fitz.Page, polygon, radius: float = 35.0) -> list[str]:
    zone = polygon.buffer(radius)
    hits = []
    for w in page.get_text("words"):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if zone.contains(Point(cx, cy)):
            hits.append(str(w[4]))
        if len(hits) >= 12:
            break
    return hits


def _is_stair_fill(fill: tuple | None) -> bool:
    if not fill:
        return False
    r, g, b = fill
    return b >= 0.72 and g >= 0.55 and r <= 0.75


def _fill_polygons(paths: list, classes: list | None, predicate):
    out = []
    for p in paths:
        if p.outside_content or p.fill_polygon is None:
            continue
        fill = None
        if classes and 0 <= p.style_id < len(classes):
            fill = classes[p.style_id].key.fill
        if predicate(fill):
            out.append(p.fill_polygon)
    return out


def _candidate(candidate_id: str, kind: str, label: str, source: str,
               polygon, page: fitz.Page, confidence: float,
               default_action: str) -> dict:
    is_stair = kind.startswith("STAIR") or bool(_STAIR_RE.search(label or ""))
    return {
        "id": candidate_id,
        "kind_hint": kind,
        "label": label,
        "source": source,
        "polygon": polygon,
        "area_pt2": float(polygon.area),
        "bbox": tuple(float(x) for x in polygon.bounds),
        "nearby_text": _nearby_text(page, polygon),
        "confidence": float(confidence),
        "default_action": default_action,
        "destructive_allowed": default_action == "opening",
        "verification_status": (
            "verified" if default_action == "opening" and confidence >= 0.85
            else "review"),
        "object_roles": ["STAIR"] if is_stair else [],
        "object_evidence_ids": (["stair_label_or_geometry"] if is_stair else []),
        "opening_intent": OpeningIntent.NONE.value,
        "opening_evidence_ids": [],
        "cut_eligible": False,
        "reject_reason": "",
    }


def _connected_fill_cluster(fills: list, seed_idx: int,
                            tolerance: float = 1.0) -> list[int]:
    """Flood-fill touching/near-touching vector fills from one stair seed."""
    tree = STRtree(fills)
    seen = {seed_idx}
    queue = [seed_idx]
    while queue:
        idx = queue.pop()
        search = fills[idx].buffer(tolerance)
        for nxt in tree.query(search):
            nxt = int(nxt)
            if nxt in seen:
                continue
            if search.intersects(fills[nxt]):
                seen.add(nxt)
                queue.append(nxt)
    return sorted(seen)


def _stair_candidates(page, paths, classes, content_rect, slab_union,
                      scale, protected_solids=None, cfg=None
                      ) -> tuple[list[dict], list[str], list[str]]:
    warnings: list[str] = []
    default_ids: list[str] = []
    anchors = _word_anchors(page, _STAIR_RE, content_rect)
    fills = _fill_polygons(paths, classes, _is_stair_fill)
    if not anchors or not fills:
        return [], default_ids, warnings

    tree = STRtree(fills)
    to_mm = PT_TO_MM * (scale or 100)
    slab_buffer = slab_union.buffer(80) if slab_union is not None else None
    candidates: list[dict] = []
    used_clusters: set[tuple[int, ...]] = set()
    for ordinal, (label, _bbox, anchor) in enumerate(anchors, 1):
        best_idx, best_score = None, float("inf")
        for idx in tree.query(anchor.buffer(180)):
            idx = int(idx)
            poly = fills[idx]
            bx = poly.bounds
            w_mm = (bx[2] - bx[0]) * to_mm
            h_mm = (bx[3] - bx[1]) * to_mm
            short, long = min(w_mm, h_mm), max(w_mm, h_mm)
            # A flight may be encoded as many narrow tread fills.  Seed from
            # one small tread, then validate/use the connected union below.
            if short < 100 or long < 150 or long > 9000:
                continue
            if slab_buffer is not None and not poly.intersects(slab_buffer):
                continue
            score = poly.distance(anchor)
            if score < best_score:
                best_idx, best_score = idx, score
        if best_idx is None:
            warnings.append(f"{label}: no qualifying blue stair fill")
            continue

        token = re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")
        prefix = f"stair_{token or ordinal}"
        landing = fills[best_idx]
        candidates.append(_candidate(
            f"{prefix}_landing", "STAIR_LANDING", label,
            "nearest_blue_fill", landing, page, 0.55, "review"))

        cluster_ids = tuple(_connected_fill_cluster(fills, best_idx))
        cluster = unary_union([fills[i] for i in cluster_ids]).buffer(0)
        if cluster.geom_type == "MultiPolygon":
            cluster = max(cluster.geoms, key=lambda g: g.area)
        full_cluster_area = cluster.area
        if slab_union is not None:
            clipped = cluster.intersection(slab_union)
            if not clipped.is_empty:
                if clipped.geom_type == "MultiPolygon":
                    clipped = max(clipped.geoms, key=lambda g: g.area)
                cluster = clipped
        cluster_key = tuple(round(v, 1) for v in cluster.bounds)
        if cluster_key in used_clusters:
            continue
        used_clusters.add(cluster_key)
        cluster_id = f"{prefix}_flight_union"
        overlap_ratio = cluster.area / max(full_cluster_area, 1e-9)
        is_edge_interface = overlap_ratio < 0.20
        label_distance_mm = cluster.distance(anchor) * to_mm
        area_m2 = cluster.area * to_mm * to_mm / 1_000_000.0
        structural_ratio = (
            cluster.intersection(protected_solids).area
            / max(cluster.area, 1e-9)
            if protected_solids is not None and not protected_solids.is_empty
            else 0.0)
        min_overlap = float(getattr(
            cfg, "stair_opening_min_slab_overlap", 0.85))
        max_structural = float(getattr(
            cfg, "stair_opening_max_structural_intersection_ratio", 0.01))
        min_area = float(getattr(cfg, "stair_opening_min_area_m2", 0.25))
        max_area = float(getattr(cfg, "stair_opening_max_area_m2", 40.0))
        max_label_distance = float(getattr(
            cfg, "stair_opening_max_label_distance_mm", 2500.0))
        verified = (
            not is_edge_interface
            and overlap_ratio >= min_overlap
            and structural_ratio <= max_structural
            and min_area <= area_m2 <= max_area
            and label_distance_mm <= max_label_distance
            and len(cluster_ids) >= 1
            and cluster.geom_type == "Polygon"
            and cluster.is_valid)
        candidate = _candidate(
            cluster_id,
            "STAIR_EDGE_INTERFACE" if is_edge_interface else "STAIR_OPENING",
            label, "connected_blue_fill_union", cluster, page,
            0.55 if is_edge_interface else (0.94 if verified else 0.72),
            "opening" if verified else "review")
        candidate.update({
            "destructive_allowed": verified,
            "verification_status": "verified" if verified else "review",
            "slab_overlap_ratio": overlap_ratio,
            "fill_component_count": len(cluster_ids),
            "label_anchor_distance_mm": label_distance_mm,
            "area_m2": area_m2,
            "structural_intersection_ratio": structural_ratio,
            "boundary_evidence": ["connected_blue_fill_union",
                                  "stair_text_anchor"],
            "geometry_audit": {
                "slab_overlap_ratio": overlap_ratio,
                "fill_component_count": len(cluster_ids),
                "label_anchor_distance_mm": label_distance_mm,
                "area_m2": area_m2,
                "structural_intersection_ratio": structural_ratio,
            },
        })
        candidates.append(candidate)
        if verified:
            default_ids.append(cluster_id)
        warnings.append(
            f"label '{label}' (STAIR): generated landing and connected "
            f"flight candidates; "
            + (f"verified={cluster_id}" if verified
               else "edge/geometry evidence marked review"))
    return candidates, default_ids, warnings


def _is_verified_stairwell_penetration(candidate: dict) -> bool:
    """Return True when a stairwell candidate has independent cut evidence.

    A stair label/flight is only object context. The P10-style case is
    different: a closed vector enclosure contains an X penetration seed and
    has a verified boundary snap to the slab edge. That is penetration
    geometry that happens to contain stair graphics.
    """
    if str(candidate.get("kind_hint", "")).upper() != "STAIRWELL":
        return False
    if candidate.get("verification_status") != "verified":
        return False
    if candidate.get("default_action") != "opening":
        return False
    if not bool(candidate.get("destructive_allowed", False)):
        return False
    try:
        coverage = float(candidate.get("boundary_coverage", 0.0) or 0.0)
    except (TypeError, ValueError):
        coverage = 0.0
    if coverage < 0.70:
        return False

    audit = candidate.get("geometry_audit", {}) or {}
    snap = audit.get("boundary_snap", {}) if isinstance(audit, dict) else {}
    if snap.get("status") != "verified_snap":
        return False

    source = str(candidate.get("source", "")).lower()
    seed_text = " ".join(
        str(seed).lower() for seed in candidate.get("contained_seed_ids", []))
    has_x_seed = (
        "x_seed" in source
        or "xcross" in seed_text
        or "x_cross" in seed_text
    )
    return has_x_seed


def _apply_multi_intent_policy(candidates: list[dict]) -> dict:
    """Separate object identity from independently proven opening intent.

    Stair graphics can describe an object but never authorize subtraction.
    A mixed stair/penetration candidate is cut only when its penetration,
    void, or shaft evidence remains valid without using stair evidence.
    """
    prevented_stairs = []
    restored_boundaries = []
    rejected_x_hulls = []
    mixed = []
    verified = []
    review = []
    for candidate in candidates:
        kind = str(candidate.get("kind_hint", "")).upper()
        nearby = " ".join(candidate.get("nearby_text", []))
        roles = set(candidate.get("object_roles", []))
        if kind.startswith("STAIR") or _STAIR_RE.search(
                str(candidate.get("label", "")) + " " + nearby):
            roles.add("STAIR")

        intent = OpeningIntent.NONE.value
        opening_evidence = list(candidate.get("opening_evidence_ids", []))
        if kind == "SLAB_PENETRATION":
            intent = OpeningIntent.SLAB_PENETRATION.value
            opening_evidence.extend([
                "closed_x_cross_vector_signature",
                "legend_slab_penetration_family",
                "slab_containment_guard",
            ])
        elif kind == "SLAB_OPENING":
            intent = OpeningIntent.SLAB_PENETRATION.value
            opening_evidence.extend([
                "closed_x_cross_vector_signature",
                "slab_containment_guard",
            ])
        elif kind in {"SHAFT", "LIFT", "CORE"}:
            intent = OpeningIntent.LIFT_SHAFT.value
            opening_evidence.extend([
                "wall_bounded_shaft_or_core_context",
                "closed_vector_geometry",
            ])
        elif kind == "VOID":
            intent = OpeningIntent.VOID.value
            opening_evidence.append("explicit_void_candidate")

        if _is_verified_stairwell_penetration(candidate):
            intent = OpeningIntent.SLAB_PENETRATION.value
            opening_evidence.extend([
                "closed_stairwell_penetration_boundary",
                "contained_x_cross_seed",
                "verified_boundary_snap",
                "slab_containment_guard",
            ])
            candidate["mixed_intent_reason"] = (
                "stair graphics are context only; closed boundary plus X seed "
                "and verified slab-edge snap prove slab penetration intent")
            restored_boundaries.append(candidate["id"])

        # Explicit local text can independently prove intent even where a
        # stair object is present. Stair labels/fill/treads are never counted.
        if re.search(r"\bSLAB\s+PENETRATION\b|\bSLAB\s+OPENING\b", nearby, re.I):
            intent = OpeningIntent.SLAB_PENETRATION.value
            opening_evidence.append("local_explicit_penetration_text")
        elif re.search(r"\bVOID\b|\bNO\s+SLAB\b", nearby, re.I):
            intent = OpeningIntent.VOID.value
            opening_evidence.append("local_explicit_void_text")
        elif re.search(r"\bLIFT\b|\bSHAFT\b", nearby, re.I):
            intent = OpeningIntent.LIFT_SHAFT.value
            opening_evidence.append("local_explicit_lift_shaft_text")

        verified_geometry = (
            candidate.get("verification_status") == "verified"
            and candidate.get("default_action") == "opening"
            and bool(candidate.get("destructive_allowed", False)))
        cut_eligible = (intent in {
            OpeningIntent.SLAB_PENETRATION.value,
            OpeningIntent.VOID.value,
            OpeningIntent.LIFT_SHAFT.value,
        } and verified_geometry and bool(opening_evidence))

        if "STAIR" in roles and intent == OpeningIntent.NONE.value:
            prevented_stairs.append(candidate["id"])
            source = str(candidate.get("source", "")).lower()
            if ("convex_hull" in source or "x_cross" in source
                    or "xcross" in source or kind == "STAIR_PENETRATION"):
                rejected_x_hulls.append(candidate["id"])
            candidate["reject_reason"] = (
                "stair object has no independent penetration/void/shaft intent")
            candidate["verification_status"] = "context_only"
            candidate["default_action"] = "review"
            candidate["destructive_allowed"] = False
        elif "STAIR" in roles and intent != OpeningIntent.NONE.value:
            mixed.append(candidate["id"])
        if cut_eligible:
            verified.append(candidate["id"])
        elif intent != OpeningIntent.NONE.value:
            review.append(candidate["id"])

        candidate["object_roles"] = sorted(roles)
        candidate["opening_intent"] = intent
        candidate["opening_evidence_ids"] = sorted(set(opening_evidence))
        candidate["cut_eligible"] = cut_eligible
        candidate["destructive_allowed"] = cut_eligible
    return {
        "verified_cut_ids": verified,
        "prevented_stair_cut_ids": prevented_stairs,
        "stair_context_blocked_ids": prevented_stairs,
        "penetration_boundary_restored_ids": restored_boundaries,
        "x_hull_rejected_ids": rejected_x_hulls,
        "mixed_stair_penetration_ids": mixed,
        "unresolved_mixed_ids": [cid for cid in mixed if cid not in verified],
        "review_ids": review,
    }


def _stair_xcross_candidates(page, paths, content_rect, slab_union,
                             scale) -> tuple[list[dict], list[str], list[str]]:
    """Detect large stairwell penetrations from finite X-cross envelopes.

    The generic X-cross detector intentionally caps shaft size at 4m. Stair
    interfaces can be larger, so this branch requires independent STAIR text
    evidence and corner-consistent crossing diagonal vectors.
    """
    anchors = _word_anchors(page, _STAIR_RE, content_rect)
    if not anchors:
        return [], [], []
    to_mm = PT_TO_MM * (scale or 100)
    diagonal_rows = []
    for path in paths:
        if path.outside_content:
            continue
        for start, end in path.segments:
            dx, dy = end[0]-start[0], end[1]-start[1]
            length = math.hypot(dx, dy)
            if length < 5:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx))) % 180
            angle = min(angle, 180-angle)
            if 12 <= angle <= 78:
                diagonal_rows.append((LineString([start, end]), length))
    candidates, default_ids, warnings = [], [], []
    seen = []
    for i, (first, first_length) in enumerate(diagonal_rows):
        for second, second_length in diagonal_rows[i+1:]:
            if min(first_length, second_length)/max(first_length, second_length) < 0.55:
                continue
            if not first.crosses(second):
                continue
            intersection = first.intersection(second)
            if intersection.geom_type != "Point":
                continue
            if (intersection.distance(first.interpolate(0.5, normalized=True))
                    > 0.28*first_length
                    or intersection.distance(second.interpolate(0.5, normalized=True))
                    > 0.28*second_length):
                continue
            points = list(first.coords) + list(second.coords)
            minx = min(point[0] for point in points)
            miny = min(point[1] for point in points)
            maxx = max(point[0] for point in points)
            maxy = max(point[1] for point in points)
            # Stair interfaces are often trapezoidal: the two diagonal lines
            # need not terminate at the corners of one axis-aligned box. The
            # finite convex hull preserves their actual vector endpoints.
            polygon = MultiPoint(points).convex_hull
            short_mm = min(maxx-minx, maxy-miny)*to_mm
            long_mm = max(maxx-minx, maxy-miny)*to_mm
            if short_mm < 500 or long_mm > 12000 or long_mm/short_mm > 5.0:
                continue
            if polygon.geom_type != "Polygon" or len(polygon.exterior.coords) < 5:
                continue
            nearest = min(anchors, key=lambda row: polygon.distance(row[2]))
            label, _bbox, anchor = nearest
            # The label must genuinely anchor this X, not merely be the
            # nearest stair elsewhere on a dense core plan.
            if polygon.distance(anchor) > max(
                    30.0, 0.25*max(first_length, second_length)):
                continue
            if slab_union is not None:
                polygon = polygon.intersection(slab_union)
                if polygon.is_empty:
                    continue
                if polygon.geom_type == "MultiPolygon":
                    polygon = max(polygon.geoms, key=lambda geom: geom.area)
            if any(polygon.intersection(existing).area /
                   max(min(polygon.area, existing.area), 1e-9) > 0.8
                   for existing in seen):
                continue
            seen.append(polygon)
            token = re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")
            candidate_id = (
                f"stair_{token}_xcross_penetration_{len(candidates)+1:02d}")
            candidates.append(_candidate(
                candidate_id, "STAIR_PENETRATION", label,
                "stair_label+x_cross_vector_envelope", polygon, page,
                0.92, "opening"))
            default_ids.append(candidate_id)
            warnings.append(
                f"{label}: large X-cross stair penetration candidate generated")
    return candidates, default_ids, warnings


def _interval_coverage(intervals: list[tuple[float, float]], start: float,
                       end: float, tolerance: float) -> float:
    clipped = []
    for a, b in intervals:
        lo, hi = max(start, min(a, b)), min(end, max(a, b))
        if hi > lo:
            clipped.append((lo, hi))
    if not clipped or end <= start:
        return 0.0
    clipped.sort()
    total, lo, hi = 0.0, clipped[0][0], clipped[0][1]
    for nxt_lo, nxt_hi in clipped[1:]:
        if nxt_lo <= hi + tolerance:
            hi = max(hi, nxt_hi)
        else:
            total += hi - lo
            lo, hi = nxt_lo, nxt_hi
    return min(1.0, (total + hi - lo) / (end - start))


def _axis_aligned_exterior_segments(geometry) -> list[dict]:
    """Return horizontal/vertical segments from slab exterior rings only."""
    rows = []
    for polygon in getattr(geometry, "geoms", [geometry]):
        if polygon.geom_type != "Polygon":
            continue
        coords = list(polygon.exterior.coords)
        for a, b in zip(coords, coords[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            if abs(dx) <= 0.05 and abs(dy) > 0.1:
                rows.append({"axis": "vertical", "value": (a[0]+b[0])/2,
                             "start": min(a[1], b[1]),
                             "end": max(a[1], b[1])})
            elif abs(dy) <= 0.05 and abs(dx) > 0.1:
                rows.append({"axis": "horizontal", "value": (a[1]+b[1])/2,
                             "start": min(a[0], b[0]),
                             "end": max(a[0], b[0])})
    return rows


def _snap_penetration_to_slab_edge(
    polygon,
    slab_union,
    scale: float,
    cfg,
    side_coverage: list[float] | None = None,
    protected_solids=None,
    other_openings: list | None = None,
):
    """Extend one verified stairwell side to a nearby slab exterior.

    This is intentionally fail-closed.  The intermediate strip must already
    be slab, have no protected structural geometry, and be bounded by a slab
    exterior that overlaps the complete opening side.
    """
    audit = {
        "status": "not_snapped",
        "before_bbox": list(polygon.bounds),
        "after_bbox": list(polygon.bounds),
        "before_polygon": [list(point) for point in polygon.exterior.coords],
        "after_polygon": [list(point) for point in polygon.exterior.coords],
        "reason": "no qualifying slab-edge attachment",
        "evidence_ids": [],
        "prevented_candidates": [],
    }
    if (polygon is None or polygon.is_empty or polygon.geom_type != "Polygon"
            or slab_union is None or slab_union.is_empty):
        return polygon, audit

    to_mm = PT_TO_MM * float(scale or 100)
    max_gap_pt = float(getattr(
        cfg, "penetration_edge_snap_max_mm", 600.0)) / max(to_mm, 1e-9)
    min_overlap = float(getattr(
        cfg, "penetration_edge_snap_min_overlap", 0.90))
    min_endpoint = float(getattr(
        cfg, "penetration_edge_snap_min_endpoint_coverage", 0.65))
    max_protected = float(getattr(
        cfg, "penetration_edge_snap_max_protected_ratio", 0.01))
    cover = list(side_coverage or [0.0, 0.0, 0.0, 0.0])
    while len(cover) < 4:
        cover.append(0.0)

    minx, miny, maxx, maxy = polygon.bounds
    width, height = maxx-minx, maxy-miny
    exterior = _axis_aligned_exterior_segments(slab_union)
    options = []

    def consider(side, axis, current, target, seg_start, seg_end,
                 side_start, side_end, endpoint_indices):
        side_length = max(side_end-side_start, 1e-9)
        overlap = max(0.0, min(seg_end, side_end)-max(seg_start, side_start))
        overlap_ratio = overlap/side_length
        gap = abs(current-target)
        if gap <= 1e-6 or gap > max_gap_pt or overlap_ratio < min_overlap:
            return
        if min(cover[index] for index in endpoint_indices) < min_endpoint:
            audit["prevented_candidates"].append({
                "side": side, "gap_mm": gap*to_mm,
                "reason": "endpoint vector coverage below threshold"})
            return
        if side == "left":
            strip = box(target, miny, minx, maxy)
        elif side == "right":
            strip = box(maxx, miny, target, maxy)
        elif side == "top":
            strip = box(minx, target, maxx, miny)
        else:
            strip = box(minx, maxy, maxx, target)
        if strip.is_empty or strip.area <= 0:
            return
        inside_ratio = strip.intersection(slab_union).area/max(strip.area, 1e-9)
        protected_ratio = (strip.intersection(protected_solids).area
                           / max(strip.area, 1e-9)
                           if protected_solids is not None
                           and not protected_solids.is_empty else 0.0)
        other_ratio = max((strip.intersection(other).area/max(strip.area, 1e-9)
                           for other in (other_openings or [])), default=0.0)
        if inside_ratio < 0.98 or protected_ratio > max_protected or other_ratio > 0.01:
            audit["prevented_candidates"].append({
                "side": side, "gap_mm": gap*to_mm,
                "inside_slab_ratio": inside_ratio,
                "protected_intersection_ratio": protected_ratio,
                "other_opening_intersection_ratio": other_ratio,
                "reason": "geometry guard rejected boundary snap"})
            return
        options.append({
            "side": side, "gap_pt": gap, "gap_mm": gap*to_mm,
            "overlap_ratio": overlap_ratio, "strip": strip,
            "strip_area_pt2": strip.area,
            "inside_slab_ratio": inside_ratio,
            "protected_intersection_ratio": protected_ratio,
        })

    for segment in exterior:
        if segment["axis"] == "vertical":
            x = segment["value"]
            if x < minx:
                consider("left", "vertical", minx, x, segment["start"],
                         segment["end"], miny, maxy, (2, 3))
            elif x > maxx:
                consider("right", "vertical", maxx, x, segment["start"],
                         segment["end"], miny, maxy, (2, 3))
        else:
            y = segment["value"]
            if y < miny:
                consider("top", "horizontal", miny, y, segment["start"],
                         segment["end"], minx, maxx, (0, 1))
            elif y > maxy:
                consider("bottom", "horizontal", maxy, y, segment["start"],
                         segment["end"], minx, maxx, (0, 1))

    if not options:
        return polygon, audit
    options.sort(key=lambda row: row["gap_pt"])
    chosen = options[0]
    expanded = unary_union([polygon, chosen.pop("strip")]).buffer(0)
    if expanded.geom_type != "Polygon":
        audit["reason"] = "snap produced non-polygon geometry"
        return polygon, audit
    audit.update(chosen)
    audit.update({
        "status": "verified_snap",
        "after_bbox": list(expanded.bounds),
        "after_polygon": [list(point) for point in expanded.exterior.coords],
        "area_added_pt2": float(expanded.area-polygon.area),
        "evidence_ids": ["slab_exterior", "vector_supported_endpoints",
                         "protected_solids_clear"],
        "reason": "bounded penetration attached to verified slab exterior",
    })
    return expanded, audit


def _stairwell_boundary_candidates(page, paths, classes, slab_union, scale,
                                   xcross_candidates, stair_candidates, cfg,
                                   protected_solids=None):
    """Build finite stairwell enclosures around X and flight seeds.

    The X-cross is deliberately only an inside seed.  The returned geometry
    is bounded by real horizontal/vertical PDF vectors; its convex hull is
    never accepted as the final opening when an enclosure is available.
    """
    if not xcross_candidates:
        return [], [], [], []
    to_mm = PT_TO_MM * float(scale or 100)
    axis_tol = max(0.5, float(getattr(
        cfg, "penetration_axis_tolerance_mm", 150.0)) / to_mm)
    out, defaults, penetrations, warnings = [], [], [], []

    for xseed in xcross_candidates:
        label = xseed["label"]
        related = [c for c in stair_candidates
                   if c["label"] == label and
                   c["kind_hint"] in {"STAIR_OPENING", "STAIR_LANDING"}]
        flight = next((c for c in related
                       if c["kind_hint"] == "STAIR_OPENING"), None)
        seed_polys = [xseed["polygon"]] + ([flight["polygon"]] if flight else [])
        seed_union = unary_union(seed_polys).buffer(0)
        minx, miny, maxx, maxy = seed_union.bounds
        width, height = maxx-minx, maxy-miny
        margin = max(30.0, min(100.0, 0.55*max(width, height)))
        search = box(minx-margin, miny-margin, maxx+margin, maxy+margin)

        horizontal, vertical = [], []
        for path in paths:
            if path.outside_content:
                continue
            if classes and 0 <= path.style_id < len(classes):
                if classes[path.style_id].key.dashes:
                    continue
            for a, b in path.segments:
                line = LineString([a, b])
                if not line.intersects(search):
                    continue
                dx, dy = b[0]-a[0], b[1]-a[1]
                length = math.hypot(dx, dy)
                if length < max(8.0, 0.10*min(width, height)):
                    continue
                if abs(dy) <= max(0.35, 0.01*length):
                    horizontal.append(((a[1]+b[1])/2,
                                       min(a[0], b[0]), max(a[0], b[0])))
                elif abs(dx) <= max(0.35, 0.01*length):
                    vertical.append(((a[0]+b[0])/2,
                                     min(a[1], b[1]), max(a[1], b[1])))

        lefts = sorted({round(x, 3) for x, _, _ in vertical
                        if minx-margin <= x <= minx+axis_tol},
                       key=lambda x: minx-x)[:6]
        rights = sorted({round(x, 3) for x, _, _ in vertical
                         if maxx-axis_tol <= x <= maxx+margin},
                        key=lambda x: x-maxx)[:6]
        tops = sorted({round(y, 3) for y, _, _ in horizontal
                       if miny-margin <= y <= miny+axis_tol},
                      key=lambda y: miny-y)[:6]
        bottoms = sorted({round(y, 3) for y, _, _ in horizontal
                          if maxy-axis_tol <= y <= maxy+margin},
                         key=lambda y: y-maxy)[:6]

        rows = []
        for left in lefts:
            for right in rights:
                for top in tops:
                    for bottom in bottoms:
                        if right-left < width*0.92 or bottom-top < height*0.92:
                            continue
                        candidate = box(left, top, right, bottom)
                        if any(candidate.intersection(seed).area /
                               max(seed.area, 1e-9) < 0.94
                               for seed in seed_polys):
                            continue
                        vleft = [(a, b) for x, a, b in vertical
                                 if abs(x-left) <= axis_tol]
                        vright = [(a, b) for x, a, b in vertical
                                  if abs(x-right) <= axis_tol]
                        htop = [(a, b) for y, a, b in horizontal
                                if abs(y-top) <= axis_tol]
                        hbottom = [(a, b) for y, a, b in horizontal
                                   if abs(y-bottom) <= axis_tol]
                        coverages = [
                            _interval_coverage(vleft, top, bottom, axis_tol),
                            _interval_coverage(vright, top, bottom, axis_tol),
                            _interval_coverage(htop, left, right, axis_tol),
                            _interval_coverage(hbottom, left, right, axis_tol),
                        ]
                        coverage = sum(coverages)/4.0
                        supported_sides = sum(value >= 0.20 for value in coverages)
                        if supported_sides < 3:
                            continue
                        area_ratio = candidate.area/max(seed_union.envelope.area, 1e-9)
                        if area_ratio > 4.0:
                            continue
                        # Prefer the outer finite enclosure when its real-line
                        # support is comparable; this prevents falling back to
                        # the smaller X hull or a single stair landing.
                        score = (0.68*coverage + 0.07*supported_sides
                                 + 0.08*min(area_ratio, 2.0)/2.0)
                        rows.append((score, coverage, coverages, candidate))
        if not rows:
            warnings.append(
                f"{label}: no vector-confirmed closed stairwell enclosure; "
                "X hull retained for review only")
            continue
        rows.sort(key=lambda row: (row[0], row[3].area), reverse=True)
        score, coverage, side_coverage, polygon = rows[0]
        # One stairwell can contain another labelled flight. Preserve the
        # finite enclosure, then include only vector flights that touch it.
        connected_flights = [
            candidate for candidate in stair_candidates
            if candidate.get("kind_hint") == "STAIR_OPENING"
            and candidate["polygon"].intersects(polygon.buffer(2.0))]
        if connected_flights:
            polygon = unary_union(
                [polygon] + [candidate["polygon"]
                             for candidate in connected_flights]).buffer(0)
            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda geom: geom.area)
        other_openings = [
            candidate["polygon"] for candidate in stair_candidates
            if candidate.get("label") != label
            and candidate.get("kind_hint") == "STAIR_OPENING"]
        polygon, snap_audit = _snap_penetration_to_slab_edge(
            polygon, slab_union, scale, cfg, side_coverage,
            protected_solids=protected_solids,
            other_openings=other_openings)
        if slab_union is not None:
            clipped = polygon.intersection(slab_union)
            if clipped.is_empty:
                continue
            if clipped.geom_type == "MultiPolygon":
                clipped = max(clipped.geoms, key=lambda geom: geom.area)
            polygon = clipped
        confidence = min(0.97, 0.62 + 0.42*coverage)
        token = re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")
        cid = f"stair_{token}_closed_stairwell"
        candidate = _candidate(
            cid, "STAIRWELL", label,
            "x_seed+flight_seed+orthogonal_vector_enclosure", polygon,
            page, confidence,
            "opening" if confidence >= getattr(
                cfg, "penetration_min_confidence", 0.85) else "review")
        candidate["boundary_coverage"] = coverage
        candidate["side_coverage"] = side_coverage
        candidate["contained_seed_ids"] = [xseed["id"]] + [
            connected["id"] for connected in connected_flights]
        candidate["rejected_hull_id"] = xseed["id"]
        candidate["geometry_audit"] = {"boundary_snap": snap_audit}
        out.append(candidate)
        if candidate["default_action"] == "opening":
            defaults.append(cid)
        penetrations.append(ResolvedPenetration(
            id=cid, kind="STAIRWELL", polygon=polygon,
            source_candidate_ids=[cid],
            contained_seed_ids=candidate["contained_seed_ids"],
            boundary_coverage=coverage, confidence=confidence,
            status="verified" if cid in defaults else "review",
            warnings=[] if cid in defaults else ["boundary confidence low"],
            geometry_audit={"boundary_snap": snap_audit}))
        warnings.append(
            f"{label}: closed stairwell enclosure coverage={coverage:.2f}, "
            f"confidence={confidence:.2f}")
    return out, defaults, penetrations, warnings


def _verified_core_wall_opening_candidates(
    walls, raw_elements, page, content_rect, slab_union, scale, cfg,
    paths=None,
):
    """Return wall-bounded shaft faces; retain the LW hull as context only."""
    lw_walls = [w for w in walls if w.label.upper().startswith("LW")]
    if len(lw_walls) < 4:
        return [], [], []
    wall_union = unary_union([w.polygon for w in lw_walls]).buffer(0)
    core_footprint = wall_union.convex_hull
    if core_footprint.is_empty or core_footprint.geom_type != "Polygon":
        return [], [], []
    to_mm = PT_TO_MM * float(scale or 100)
    area_m2 = core_footprint.area*to_mm*to_mm/1_000_000.0
    if not 1.0 <= area_m2 <= 100.0:
        return [], [], []
    if slab_union is not None and not core_footprint.intersects(slab_union):
        return [], [], []

    context = _candidate(
        "core_lw_wall_enclosed", "CORE_CONTEXT", "CORE/LW",
        "lw_wall_envelope_context_only", core_footprint, page, 0.99,
        "exclude")
    context.update({
        "destructive_allowed": False,
        "verification_status": "context_only",
        "reject_reason": "LW envelope contains structural wall solids",
        "wall_intersection_ratio": (
            core_footprint.intersection(wall_union).area
            / max(core_footprint.area, 1e-9)),
    })

    boundary_tol = max(0.5, 75.0/max(to_mm, 1e-9))
    min_coverage = float(getattr(
        cfg, "core_opening_min_boundary_coverage", 0.70))
    max_wall_ratio = float(getattr(
        cfg, "core_opening_max_wall_intersection_ratio", 0.01))
    candidates = [context]
    default_ids = []
    verified_count = 0
    accepted_polygons = []
    for element in raw_elements:
        if element.type not in {"VOID", "SHAFT", "LIFT"}:
            continue
        polygon = element.polygon.buffer(0)
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        contained_ratio = (polygon.intersection(core_footprint).area
                           / max(polygon.area, 1e-9))
        if contained_ratio < 0.98:
            continue
        wall_ratio = (polygon.intersection(wall_union).area
                      / max(polygon.area, 1e-9))
        boundary_coverage = (
            polygon.boundary.intersection(
                wall_union.buffer(boundary_tol)).length
            / max(polygon.boundary.length, 1e-9))
        face_area_m2 = polygon.area*to_mm*to_mm/1_000_000.0
        verified = (0.25 <= face_area_m2 <= 100.0
                    and wall_ratio <= max_wall_ratio
                    and boundary_coverage >= min_coverage)
        cid = f"core_interior_{len(candidates):02d}"
        candidate = _candidate(
            cid, "SHAFT", "CORE/SHAFT",
            "closed_raw_face+lw_wall_ring", polygon, page,
            min(0.98, 0.72+0.30*boundary_coverage),
            "opening" if verified else "review")
        candidate.update({
            "destructive_allowed": verified,
            "verification_status": "verified" if verified else "review",
            "wall_intersection_ratio": wall_ratio,
            "core_containment_ratio": contained_ratio,
            "boundary_coverage": boundary_coverage,
            "source_element_type": element.type,
            "geometry_audit": {
                "wall_intersection_ratio": wall_ratio,
                "core_containment_ratio": contained_ratio,
                "boundary_coverage": boundary_coverage,
                "area_m2": face_area_m2,
            },
        })
        candidates.append(candidate)
        if verified:
            default_ids.append(cid)
            verified_count += 1
            accepted_polygons.append(polygon)

    # Some drawings use a single corner-to-corner diagonal inside a closed
    # shaft face instead of a complete X.  The generic element detector
    # intentionally requires two diagonals, so recover only faces whose
    # geometry is already fully enclosed by the LW topology and whose one
    # diagonal spans the face boundary.  The diagonal is evidence; the wall-
    # bounded free face remains the geometry authority.
    free_space = core_footprint.difference(wall_union).buffer(0)
    diagonal_lines = []
    for path in paths or []:
        if path.outside_content:
            continue
        for start, end in path.segments:
            dx, dy = end[0]-start[0], end[1]-start[1]
            length = math.hypot(dx, dy)
            if length <= 2.0:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            angle = min(angle, 180.0-angle)
            if 15.0 <= angle <= 75.0:
                diagonal_lines.append(LineString([start, end]))

    for face in getattr(free_space, "geoms", [free_space]):
        if face.is_empty or face.geom_type != "Polygon":
            continue
        face_area_m2 = face.area*to_mm*to_mm/1_000_000.0
        if not 0.25 <= face_area_m2 <= 100.0:
            continue
        if any(face.intersection(existing).area / max(face.area, 1e-9) >= 0.90
               for existing in accepted_polygons):
            continue
        boundary_coverage = (
            face.boundary.intersection(
                wall_union.buffer(boundary_tol)).length
            / max(face.boundary.length, 1e-9))
        if boundary_coverage < min_coverage:
            continue
        minx, miny, maxx, maxy = face.bounds
        face_diagonal = math.hypot(maxx-minx, maxy-miny)
        spanning = []
        for line in diagonal_lines:
            clipped = line.intersection(face)
            if clipped.is_empty:
                continue
            span_ratio = clipped.length / max(face_diagonal, 1e-9)
            coords = list(line.coords)
            endpoint_error = max(
                Point(coords[0]).distance(face.boundary),
                Point(coords[-1]).distance(face.boundary))
            if span_ratio >= 0.75 and endpoint_error <= boundary_tol:
                spanning.append({
                    "line_bounds": [float(value) for value in line.bounds],
                    "span_ratio": float(span_ratio),
                    "endpoint_error_pt": float(endpoint_error),
                })
        if not spanning:
            continue
        cid = f"core_interior_{len(candidates):02d}"
        confidence = min(0.98, 0.82+0.12*boundary_coverage)
        candidate = _candidate(
            cid, "SHAFT", "CORE/SHAFT",
            "closed_lw_interior_face+spanning_diagonal_seed", face, page,
            confidence, "opening")
        candidate.update({
            "destructive_allowed": True,
            "verification_status": "verified",
            "wall_intersection_ratio": 0.0,
            "core_containment_ratio": 1.0,
            "boundary_coverage": boundary_coverage,
            "source_element_type": "SINGLE_DIAGONAL_SHAFT_SEED",
            "geometry_audit": {
                "wall_intersection_ratio": 0.0,
                "core_containment_ratio": 1.0,
                "boundary_coverage": boundary_coverage,
                "area_m2": face_area_m2,
                "spanning_diagonals": spanning,
            },
        })
        candidates.append(candidate)
        default_ids.append(cid)
        verified_count += 1
        accepted_polygons.append(face)

    return candidates, default_ids, [
        f"CORE: envelope retained as context only; {verified_count} "
        "wall-bounded interior shaft face(s) verified"]


def _raw_candidates(raw_elements, walls, page, content_rect, slab_union=None,
                    scale: float = 100.0, columns=None, cfg=None):
    warnings: list[str] = []
    candidates: list[dict] = []
    default_ids: list[str] = []
    core_anchors = _word_anchors(page, _CORE_RE, content_rect)
    equipment_anchors = _word_anchors(page, _EQUIPMENT_RE, content_rect)
    wall_polys = [w.polygon for w in walls
                  if str(w.label).upper().startswith("LW")]
    wall_union = unary_union(wall_polys) if wall_polys else None
    structural_polys = [wall.polygon for wall in walls]
    structural_polys.extend(column.polygon for column in (columns or []))
    structural_union = (unary_union(structural_polys).buffer(0)
                        if structural_polys else None)
    page_text = page.get_text("text").upper()
    legend_has_slab_penetration = bool(re.search(
        r"\bSLAB\s+PENETRATION\b", page_text))
    to_mm = PT_TO_MM * float(scale or 100)
    min_penetration_area = float(getattr(
        cfg, "slab_penetration_min_area_m2", 0.05))
    max_penetration_area = float(getattr(
        cfg, "slab_penetration_max_area_m2", 10.0))
    max_structural_ratio = float(getattr(
        cfg, "slab_penetration_max_structural_intersection_ratio", 0.01))

    for i, element in enumerate(raw_elements, 1):
        nearby = _nearby_text(page, element.polygon, radius=40.0)
        steel_label_count = sum(bool(_STEEL_LABEL_RE.match(text))
                                for text in nearby)
        near_equipment = any(element.polygon.distance(pt) < 45
                             for _text, _bbox, pt in equipment_anchors)
        near_core = any(element.polygon.distance(pt) < 140
                        for _text, _bbox, pt in core_anchors)
        near_lw = wall_union is not None and element.polygon.distance(wall_union) < 35
        near_stair = any(re.search(r"\bSTAIR\b", text, re.I)
                         for text in nearby)
        local_opening_text = bool(re.search(
            r"\bSLAB\s+PENETRATION\b|\bSLAB\s+OPENING\b|\bVOID\b|"
            r"\bNO\s+SLAB\b|\bLIFT\b|\bSHAFT\b",
            " ".join(nearby), re.I))
        slab_containment = (element.polygon.intersection(slab_union).area
                            / max(element.polygon.area, 1e-9)
                            if slab_union is not None
                            and not slab_union.is_empty else 0.0)
        structural_ratio = (
            element.polygon.intersection(structural_union).area
            / max(element.polygon.area, 1e-9)
            if structural_union is not None and not structural_union.is_empty
            else 0.0)
        area_m2 = element.polygon.area*to_mm*to_mm/1_000_000.0
        if near_equipment:
            kind, label, action, confidence = (
                "EQUIPMENT_REBATE", "FB/FLOOR BOX", "exclude", 0.95)
            warnings.append("raw X-cross near FB/FLOOR BOX excluded by default")
        elif steel_label_count >= 2:
            kind, label, action, confidence = (
                "STEELWORK_SYMBOL", "STEELWORK X-CROSS", "exclude", 0.95)
            warnings.append(
                "raw X-cross surrounded by steel labels excluded by default")
        elif element.type in {"LIFT", "SHAFT"} or near_core or near_lw:
            kind, label, action, confidence = (
                "SHAFT", "CORE/SHAFT", "opening", 0.90)
        elif (legend_has_slab_penetration
              and (not near_stair or local_opening_text)
              and slab_containment >= 0.98
              and structural_ratio <= max_structural_ratio
              and min_penetration_area <= area_m2 <= max_penetration_area):
            kind, label, action, confidence = (
                "SLAB_PENETRATION", "SLAB PENETRATION", "opening", 0.96)
        elif (not near_stair
              and slab_containment >= 0.90
              and structural_ratio <= 0.05
              and min_penetration_area <= area_m2 <= max_penetration_area):
            kind, label, action, confidence = (
                "SLAB_OPENING", element.label or "SLAB OPENING",
                "opening", 0.88)
        else:
            kind, label, action, confidence = (
                element.type, element.label, "review", 0.55)
        cid = f"raw_{i:02d}_{kind.lower()}"
        candidate = _candidate(
            cid, kind, label, "x_cross_vector", element.polygon,
            page, confidence, action)
        candidate["geometry_audit"] = {
            "legend_has_slab_penetration": legend_has_slab_penetration,
            "slab_containment_ratio": slab_containment,
            "structural_intersection_ratio": structural_ratio,
            "area_m2": area_m2,
            "near_stair": near_stair,
            "local_opening_text": local_opening_text,
            "steel_label_count": steel_label_count,
            "near_equipment": near_equipment,
        }
        if near_stair:
            candidate["object_roles"] = sorted(set(
                candidate.get("object_roles", [])) | {"STAIR"})
            candidate["object_evidence_ids"] = sorted(set(
                candidate.get("object_evidence_ids", [])) |
                {"nearby_stair_context"})
        candidates.append(candidate)
        if action == "opening":
            default_ids.append(cid)
    return candidates, default_ids, warnings


def _dedupe_elements(elements: list[ElementFootprint]) -> list[ElementFootprint]:
    kept: list[ElementFootprint] = []
    for element in sorted(elements, key=lambda x: -x.polygon.area):
        normalized_label = re.sub(r"[^A-Z0-9]+", "", element.label.upper())
        if (element.type == "STAIR" and normalized_label
                and any(other.type == "STAIR" and
                        re.sub(r"[^A-Z0-9]+", "", other.label.upper())
                        == normalized_label for other in kept)):
            continue
        if any(
            element.polygon.intersection(other.polygon).area
            / max(min(element.polygon.area, other.polygon.area), 1e-9) > 0.65
            for other in kept
        ):
            continue
        kept.append(element)
    return kept


def resolve_openings(page: fitz.Page, paths: list, classes: list | None,
                     raw_elements: list[ElementFootprint], walls: list,
                     slabs: list, scale: float, content_rect: fitz.Rect,
                     cfg=None, renderer=None, use_ai: bool = True,
                     columns: list | None = None) -> OpeningResolution:
    slab_union = unary_union([s["polygon_pdf"] for s in slabs]) if slabs else None
    protected_polys = [wall.polygon for wall in walls]
    protected_polys.extend(column.polygon for column in (columns or []))
    protected_solids = (unary_union(protected_polys).buffer(0)
                        if protected_polys else None)
    stair_candidates, stair_defaults, stair_warnings = _stair_candidates(
        page, paths, classes, content_rect, slab_union, scale,
        protected_solids=protected_solids, cfg=cfg)
    xcross_candidates, xcross_defaults, xcross_warnings = (
        _stair_xcross_candidates(
            page, paths, content_rect, slab_union, scale))
    boundary_candidates, boundary_defaults, resolved_penetrations, boundary_warnings = (
        _stairwell_boundary_candidates(
            page, paths, classes, slab_union, scale, xcross_candidates,
            stair_candidates, cfg, protected_solids=protected_solids))
    boundary_labels = {candidate["label"] for candidate in boundary_candidates}
    xcross_labels = {candidate["label"] for candidate in xcross_candidates}
    if xcross_labels:
        stair_defaults = [candidate_id for candidate_id in stair_defaults
                          if next((candidate["label"] for candidate
                                   in stair_candidates
                                   if candidate["id"] == candidate_id), None)
                          not in xcross_labels]
    if boundary_labels:
        # Closed vector enclosure supersedes both the flight-only footprint
        # and the convex hull of diagonal endpoints.
        stair_defaults = [candidate_id for candidate_id in stair_defaults
                          if next((candidate["label"] for candidate
                                   in stair_candidates
                                   if candidate["id"] == candidate_id), None)
                          not in boundary_labels]
        xcross_defaults = [candidate_id for candidate_id in xcross_defaults
                           if next((candidate["label"] for candidate
                                    in xcross_candidates
                                    if candidate["id"] == candidate_id), None)
                           not in boundary_labels]
        for candidate in xcross_candidates:
            if candidate["label"] in boundary_labels:
                candidate["default_action"] = "review"
                candidate["source"] += "+rejected_as_final_hull"
    raw_candidates, raw_defaults, raw_warnings = _raw_candidates(
        raw_elements, walls, page, content_rect, slab_union, scale,
        columns=columns, cfg=cfg)
    core_candidates, core_defaults, core_warnings = (
        _verified_core_wall_opening_candidates(
            walls, raw_elements, page, content_rect, slab_union, scale, cfg,
            paths=paths))
    candidates = (stair_candidates + xcross_candidates
                  + boundary_candidates + raw_candidates + core_candidates)
    policy = _apply_multi_intent_policy(candidates)
    # The old per-detector defaults are audit evidence only. The policy is
    # the single authority for destructive defaults.
    default_ids = list(policy["verified_cut_ids"])

    judgement = {
        "status": "deterministic",
        "opening_ids": default_ids,
        "exclude_ids": [c["id"] for c in candidates
                        if c["default_action"] == "exclude"],
        "confidence": 0.75,
        "reason": "deterministic candidate policy",
    }
    judge_warning = None
    if use_ai and cfg is not None and getattr(cfg, "enable_opening_judge", True):
        try:
            from src.slab_v2.opening_judge import judge_candidates
            judged = judge_candidates(
                page, candidates, slabs, cfg, renderer,
                content_area_pt2=max(content_rect.width * content_rect.height, 1.0))
            if judged.get("status") == "accepted":
                judgement = judged
            else:
                judge_warning = judged.get("reason") or "judge not accepted"
        except Exception as exc:
            judge_warning = f"opening judge failed: {exc}"

    # Geometry guard: once a closed vector enclosure is verified, neither an
    # LLM decision nor the legacy deterministic policy may replace it with
    # the convex hull of X endpoints.
    by_id = {c["id"]: c for c in candidates}
    judged_ids = [
        candidate_id for candidate_id in judgement.get("opening_ids", [])
        if by_id.get(candidate_id, {}).get("cut_eligible", False)]
    for boundary in boundary_candidates:
        if not boundary.get("cut_eligible", False):
            continue
        blocked = set(boundary.get("contained_seed_ids", []))
        blocked.add(boundary.get("rejected_hull_id"))
        judged_ids = [candidate_id for candidate_id in judged_ids
                      if candidate_id not in blocked]
        if boundary["id"] not in judged_ids:
            judged_ids.append(boundary["id"])
    for core in core_candidates:
        if (core.get("cut_eligible", False)
                and core["id"] not in judged_ids):
            judged_ids.append(core["id"])
    for raw in raw_candidates:
        if (raw.get("cut_eligible", False)
                and raw["id"] not in judged_ids):
            judged_ids.append(raw["id"])
    judgement["opening_ids"] = judged_ids

    selected = []
    min_confidence = float(getattr(
        cfg, "penetration_min_confidence", 0.85))
    unresolved_ids = []
    for cid in judgement.get("opening_ids", []):
        candidate = by_id.get(cid)
        if (not candidate or not candidate.get("cut_eligible", False)
                or float(candidate.get("confidence", 0.0)) < min_confidence):
            unresolved_ids.append(cid)
            continue
        polygon = candidate["polygon"]
        if slab_union is not None and not polygon.intersects(slab_union):
            continue
        intent = candidate.get("opening_intent", OpeningIntent.NONE.value)
        if intent == OpeningIntent.LIFT_SHAFT.value:
            etype = "SHAFT"
        elif intent == OpeningIntent.SLAB_PENETRATION.value:
            etype = "SLAB_PENETRATION"
        else:
            etype = "VOID"
        selected.append(ElementFootprint(
            type=etype, polygon=polygon, label=candidate["label"],
            anchor_bbox=tuple(candidate["bbox"]), area_pt2=polygon.area,
            opening_intent=intent,
            object_roles=list(candidate.get("object_roles", [])),
            evidence_ids=list(candidate.get("opening_evidence_ids", [])),
            candidate_id=cid))
    resolved = _dedupe_elements(selected)

    context_objects = []
    for candidate in candidates:
        if "STAIR" not in candidate.get("object_roles", []):
            continue
        context_objects.append(ElementFootprint(
            type="STAIR", polygon=candidate["polygon"],
            label=candidate.get("label", "STAIR"),
            anchor_bbox=tuple(candidate["bbox"]),
            area_pt2=candidate["polygon"].area,
            opening_intent=candidate.get(
                "opening_intent", OpeningIntent.NONE.value),
            object_roles=list(candidate.get("object_roles", [])),
            evidence_ids=list(candidate.get("object_evidence_ids", [])),
            candidate_id=candidate["id"]))
    context_objects = _dedupe_elements(context_objects)

    # A precursor is not an unresolved destructive decision once a verified
    # replacement covers it. This keeps the audit honest without making the
    # model fail merely because both raw and resolved candidates are shown.
    verified_polygons = [
        by_id[cid]["polygon"] for cid in judgement.get("opening_ids", [])
        if cid in by_id and by_id[cid].get("destructive_allowed", False)
    ]
    high_impact_review_ids = []
    for candidate in candidates:
        if candidate.get("destructive_allowed", False):
            continue
        if candidate.get("verification_status") == "context_only":
            continue
        if candidate.get("kind_hint") in {
                "CORE_CONTEXT", "EQUIPMENT_REBATE", "STEELWORK_SYMBOL"}:
            continue
        if candidate.get("source", "").endswith("rejected_as_final_hull"):
            continue
        polygon = candidate.get("polygon")
        superseded = polygon is not None and any(
            polygon.intersection(verified).area / max(polygon.area, 1e-9)
            >= 0.90 for verified in verified_polygons)
        if superseded:
            continue
        if (candidate.get("default_action") == "review"
                and candidate.get("kind_hint") in {
                    "STAIRWELL", "STAIR_PENETRATION", "SHAFT", "VOID",
                    "LIFT", "CORE", "STAIR_OPENING",
                    "STAIR_EDGE_INTERFACE"}):
            high_impact_review_ids.append(candidate["id"])
    stair_review_ids = [
        candidate["id"] for candidate in stair_candidates
        if candidate.get("kind_hint") in {
            "STAIR_OPENING", "STAIR_EDGE_INTERFACE"}
        and candidate.get("verification_status") != "verified"
    ]

    warnings = (stair_warnings + xcross_warnings + boundary_warnings
                + raw_warnings + core_warnings)
    if judge_warning:
        warnings.append(judge_warning + "; deterministic opening policy used")
    report = {
        "raw_elements": len(raw_elements),
        "candidate_count": len(candidates),
        "resolved_openings": len(resolved),
        "stairs": 0,
        "stair_context_count": len(context_objects),
        "shafts": sum(e.type in {"SHAFT", "LIFT"} for e in resolved),
        "voids": sum(e.type == "VOID" for e in resolved),
        "judge_status": judgement.get("status"),
        "judge_confidence": judgement.get("confidence", 0.0),
        "resolved_penetrations": len(resolved_penetrations),
        "verified_cuts": len(resolved),
        "unresolved_candidate_ids": unresolved_ids,
        "high_impact_review_ids": high_impact_review_ids,
        "opening_policy_version": getattr(
            cfg, "opening_policy_version", "penetration_only_v2"),
        "verified_cut_ids": policy["verified_cut_ids"],
        "prevented_stair_cut_ids": policy["prevented_stair_cut_ids"],
        "stair_context_blocked_ids": policy["stair_context_blocked_ids"],
        "penetration_boundary_restored_ids": policy[
            "penetration_boundary_restored_ids"],
        "x_hull_rejected_ids": policy["x_hull_rejected_ids"],
        "mixed_stair_penetration_ids": policy[
            "mixed_stair_penetration_ids"],
        "unresolved_mixed_ids": policy["unresolved_mixed_ids"],
        "judge_exclusions_overridden": [],
        "stair_review_ids": stair_review_ids,
        "prevented_overcuts": sum(
            not candidate.get("destructive_allowed", False)
            for candidate in candidates),
        "boundary_snaps": sum(
            candidate.get("geometry_audit", {}).get(
                "boundary_snap", {}).get("status") == "verified_snap"
            for candidate in candidates),
    }
    return OpeningResolution(
        resolved_openings=resolved,
        verified_cut_openings=resolved,
        context_objects=context_objects,
        review_candidates=[c for c in candidates
                           if not c.get("cut_eligible", False)
                           and c.get("verification_status") != "context_only"],
        stair_footprints=context_objects,
        core_shaft_footprints=[e for e in resolved if e.type in {"SHAFT", "LIFT"}],
        resolved_penetrations=resolved_penetrations,
        candidates=candidates,
        judgement=judgement,
        warnings=warnings,
        report=report,
    )
