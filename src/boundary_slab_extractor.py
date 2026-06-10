"""
Boundary-first slab extraction for line-only / no-fill structural PDFs.

This module does not replace fill-based slab extraction. It recovers slab
regions when fill evidence is weak by polygonizing vector boundary linework and
selecting plausible enclosed floor regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import fitz
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

from src.structural_boundary_detector import StructuralBoundaryResult, detect_structural_boundary_objects


@dataclass
class BoundaryExtractionResult:
    gross_regions: list[Polygon] = field(default_factory=list)
    final_regions: list[Polygon] = field(default_factory=list)
    wall_core_candidates: list[Polygon] = field(default_factory=list)
    boundary_evidence: list[Polygon] = field(default_factory=list)
    boundary_signatures: list[dict] = field(default_factory=list)
    grid_column_anchors: list[Polygon] = field(default_factory=list)
    structural_objects: StructuralBoundaryResult | None = None
    uncertain_candidates: list[Polygon] = field(default_factory=list)
    ignored_regions: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    mode_reason: str = "not_run"
    debug: dict = field(default_factory=dict)


def _drawing_area(page: fitz.Page) -> Polygon:
    """Conservative drawing area excluding common legend/title block zones."""
    w, h = page.rect.width, page.rect.height
    return box(0, 0, w * 0.84, h * 0.90)


@dataclass
class StyledSegment:
    line: LineString
    style_key: str
    width: float
    color: tuple | None = None
    dashed: bool = False


def _segment_from_item(item) -> Optional[LineString]:
    if item[0] != "l":
        return None
    p1 = (float(item[1].x), float(item[1].y))
    p2 = (float(item[2].x), float(item[2].y))
    if math.dist(p1, p2) < 3:
        return None
    return LineString([p1, p2])


def _is_axis_aligned(seg: LineString, tol: float = 1.5) -> bool:
    (x1, y1), (x2, y2) = list(seg.coords)
    return abs(x1 - x2) <= tol or abs(y1 - y2) <= tol


def _color_bucket(color) -> str:
    if color is None or len(color) < 3:
        return "none"
    r, g, b = [float(v) for v in color[:3]]
    # Keep this generic: the bucket is descriptive, not a hardcoded semantic rule.
    return f"{round(r, 1):.1f},{round(g, 1):.1f},{round(b, 1):.1f}"


def _style_key(d: dict) -> str:
    width = float(d.get("width") or 0)
    width_bucket = round(width * 2) / 2.0
    dashes = d.get("dashes")
    dashed = bool(dashes and str(dashes).strip() not in ("[]", "()", "None", ""))
    return f"c={_color_bucket(d.get('color'))}|w={width_bucket:.1f}|dash={int(dashed)}"


def _style_score(length: float, long_count: int, axis_count: int, width: float,
                 color, page_area: float) -> float:
    page_diag = math.sqrt(max(page_area, 1.0))
    score = 0.0
    score += min(0.40, length / max(page_diag * 2.5, 1.0))
    score += min(0.20, long_count * 0.025)
    score += min(0.20, axis_count * 0.012)
    score += 0.10 if width >= 0.6 else 0.0
    if color is not None and len(color) >= 3:
        r, g, b = [float(v) for v in color[:3]]
        brightness = (r + g + b) / 3.0
        chroma = max(r, g, b) - min(r, g, b)
        if brightness < 0.82 or chroma > 0.08:
            score += 0.10
        if brightness > 0.86 and width < 0.5:
            score -= 0.20
    return max(0.0, min(0.95, score))


def _rect_lines(rect: fitz.Rect) -> list[LineString]:
    if rect.is_empty or rect.is_infinite:
        return []
    pts = [
        (float(rect.x0), float(rect.y0)),
        (float(rect.x1), float(rect.y0)),
        (float(rect.x1), float(rect.y1)),
        (float(rect.x0), float(rect.y1)),
    ]
    return [LineString([pts[i], pts[(i + 1) % 4]]) for i in range(4)]


def _extract_linework(drawings: list[dict]) -> tuple[list[LineString], list[Polygon], list[Polygon], list[StyledSegment]]:
    lines: list[LineString] = []
    wall_like: list[Polygon] = []
    anchors: list[Polygon] = []
    styled: list[StyledSegment] = []
    for d in drawings:
        width = float(d.get("width") or 0)
        fill = d.get("fill")
        rect = d.get("rect")
        style_key = _style_key(d)
        color = d.get("color")
        dashes = d.get("dashes")
        dashed = bool(dashes and str(dashes).strip() not in ("[]", "()", "None", ""))
        if rect and (width >= 0.2 or fill is not None or not d.get("items")):
            rect_lines = _rect_lines(rect)
            lines.extend(rect_lines)
            styled.extend(StyledSegment(seg, style_key, width, color, dashed) for seg in rect_lines)
            try:
                poly = box(rect.x0, rect.y0, rect.x1, rect.y1)
                rw = abs(rect.x1 - rect.x0)
                rh = abs(rect.y1 - rect.y0)
                if min(rw, rh) >= 4 and max(rw, rh) / max(min(rw, rh), 1) <= 8:
                    anchors.append(poly)
                elif min(rw, rh) >= 2:
                    wall_like.append(poly)
            except Exception:
                pass

        for item in d.get("items", []):
            seg = _segment_from_item(item)
            if seg is not None and _is_axis_aligned(seg):
                lines.append(seg)
                styled.append(StyledSegment(seg, style_key, width, color, dashed))

        # Thick stroked paths often represent walls/edges; keep their envelopes for debug.
        if width >= 1.2 and d.get("items"):
            pts = []
            for item in d.get("items", []):
                if item[0] == "l":
                    pts.extend([(float(item[1].x), float(item[1].y)), (float(item[2].x), float(item[2].y))])
            if len(pts) >= 2:
                try:
                    wall_like.append(LineString(pts).buffer(max(width, 1.2), cap_style=2, join_style=2))
                except Exception:
                    pass
    return lines, wall_like, anchors, styled


def _learn_boundary_signatures(page: fitz.Page, styled: list[StyledSegment]) -> tuple[list[dict], list[LineString]]:
    """Learn likely boundary styles from this page without hardcoding colors."""
    page_area = page.rect.width * page.rect.height
    page_diag = math.sqrt(max(page_area, 1.0))
    groups: dict[str, dict] = {}
    for item in styled:
        line = item.line
        if line.length < max(8.0, page_diag * 0.006):
            continue
        if not _is_axis_aligned(line, tol=2.0):
            continue
        g = groups.setdefault(item.style_key, {
            "style_key": item.style_key,
            "total_length": 0.0,
            "segment_count": 0,
            "long_count": 0,
            "axis_count": 0,
            "width": item.width,
            "color": item.color,
            "dashed": item.dashed,
            "score": 0.0,
        })
        g["total_length"] += float(line.length)
        g["segment_count"] += 1
        g["axis_count"] += 1
        if line.length >= page_diag * 0.045:
            g["long_count"] += 1

    signatures: list[dict] = []
    for g in groups.values():
        g["score"] = _style_score(
            g["total_length"],
            g["long_count"],
            g["axis_count"],
            float(g["width"] or 0),
            g.get("color"),
            page_area,
        )
        if g["score"] >= 0.38 and g["total_length"] >= page_diag * 0.20:
            signatures.append(g)
    signatures.sort(key=lambda x: (x["score"], x["total_length"]), reverse=True)
    signatures = signatures[:8]
    keys = {s["style_key"] for s in signatures}
    evidence_lines = [
        item.line for item in styled
        if item.style_key in keys
        and item.line.length >= max(8.0, page_diag * 0.006)
        and _is_axis_aligned(item.line, tol=2.0)
    ]
    return signatures, evidence_lines


def _score_region(poly: Polygon, page: fitz.Page, drawing_area: Polygon, anchors: list[Polygon],
                  wall_evidence: list[Polygon] | None = None,
                  boundary_evidence: list[Polygon] | None = None) -> tuple[float, list[str]]:
    reasons = []
    page_area = page.rect.width * page.rect.height
    area_frac = poly.area / max(page_area, 1.0)
    if area_frac < 0.004:
        return 0.0, ["too_small"]
    if area_frac > 0.80:
        return 0.0, ["too_large"]

    centroid = poly.centroid
    if not drawing_area.buffer(2).contains(centroid):
        return 0.0, ["outside_drawing_area"]

    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny
    if width < page.rect.width * 0.08 or height < page.rect.height * 0.08:
        return 0.0, ["too_narrow"]

    score = 0.35
    if 0.03 <= area_frac <= 0.55:
        score += 0.25
        reasons.append("area_plausible")
    if drawing_area.contains(poly.centroid):
        score += 0.15
        reasons.append("inside_drawing_area")
    if anchors:
        nearby = sum(1 for a in anchors if poly.buffer(3).intersects(a))
        if nearby:
            score += min(0.20, nearby * 0.025)
            reasons.append(f"anchors={nearby}")
    if wall_evidence:
        hits = sum(1 for w in wall_evidence if poly.boundary.buffer(4).intersects(w))
        if hits:
            score += min(0.22, hits * 0.018)
            reasons.append(f"walls={hits}")
    if boundary_evidence:
        hits = sum(1 for e in boundary_evidence if poly.boundary.buffer(3).intersects(e))
        if hits:
            score += min(0.30, hits * 0.025)
            reasons.append(f"boundary_signature={hits}")
    try:
        rectangularity = poly.area / max(poly.envelope.area, 1.0)
        if rectangularity > 0.30:
            score += 0.05
            reasons.append("rectangularity_ok")
    except Exception:
        pass
    return min(score, 0.95), reasons


def extract_boundary_first_slabs(page: fitz.Page, drawings: list[dict],
                                 text_blocks: list[dict] | None = None,
                                 legend_semantics: dict | None = None) -> BoundaryExtractionResult:
    """Build slab candidates from closed vector boundary regions."""
    drawing_area = _drawing_area(page)
    lines, wall_like, anchors, styled = _extract_linework(drawings)
    boundary_signatures, boundary_lines = _learn_boundary_signatures(page, styled)
    boundary_evidence = [line.buffer(2.0, cap_style=2, join_style=2) for line in boundary_lines]
    structural_objects = detect_structural_boundary_objects(
        page, drawings, text_blocks=text_blocks, auto_cut_voids=True,
        legend_semantics=legend_semantics,
    )
    wall_evidence = wall_like + [w.polygon for w in structural_objects.walls]
    semantic_boundary_evidence = [
        obj.polygon for obj in structural_objects.uncertain_regions
        if obj.kind == "slab_boundary_evidence"
    ]
    all_boundary_evidence = boundary_evidence + semantic_boundary_evidence
    if not lines:
        return BoundaryExtractionResult(
            boundary_evidence=all_boundary_evidence,
            boundary_signatures=boundary_signatures,
            structural_objects=structural_objects,
            mode_reason="no_vector_linework",
            debug=structural_objects.debug,
        )
    raw_line_count = len(lines)
    polygonize_lines = boundary_lines if len(boundary_lines) >= 4 else lines
    if boundary_lines:
        # Signature-first is both more general and much faster on dense PDFs:
        # polygonize the page-learned boundary graph instead of every text/grid tick.
        min_len = max(6.0, math.sqrt(page.rect.width * page.rect.height) * 0.005)
        polygonize_lines = [line for line in boundary_lines if line.length >= min_len]
    if len(polygonize_lines) > 4500:
        # Dense drawings can make polygonize explode. Keep the longest structural
        # linework first; short ticks/text fragments rarely define slab extent.
        polygonize_lines = sorted(polygonize_lines, key=lambda s: s.length, reverse=True)[:4500]

    try:
        merged = unary_union(polygonize_lines)
        polygons = [p.buffer(0) for p in polygonize(merged) if p.area > 1]
    except Exception as exc:
        return BoundaryExtractionResult(
            structural_objects=structural_objects,
            mode_reason=f"polygonize_failed:{type(exc).__name__}",
            debug=structural_objects.debug,
        )

    scored = []
    ignored = []
    for poly in polygons:
        if poly.is_empty or not poly.is_valid:
            continue
        score, reasons = _score_region(
            poly, page, drawing_area, anchors, wall_evidence, all_boundary_evidence
        )
        has_evidence = any(r.startswith(("walls=", "boundary_signature=", "anchors=")) for r in reasons)
        if not has_evidence:
            score = 0.0
            reasons.append("no_boundary_evidence")
        if score <= 0:
            ignored.append({"polygon": poly, "reason": ",".join(reasons)})
            continue
        scored.append((score, poly, reasons))

    if not scored:
        return BoundaryExtractionResult(
            wall_core_candidates=wall_like,
            boundary_evidence=all_boundary_evidence,
            boundary_signatures=boundary_signatures,
            grid_column_anchors=anchors,
            structural_objects=structural_objects,
            ignored_regions=ignored,
            mode_reason="no_plausible_closed_regions",
            debug={
                "polygonized_regions": len(polygons),
                "boundary_signature_count": len(boundary_signatures),
                "boundary_evidence_count": len(all_boundary_evidence),
                **structural_objects.debug,
            },
        )

    scored.sort(key=lambda x: (x[0], x[1].area), reverse=True)
    top_score = scored[0][0]
    top_area = scored[0][1].area
    kept = [
        poly for score, poly, _ in scored
        if score >= max(0.55, top_score - 0.14) and poly.area >= top_area * 0.08
    ]
    uncertain = [poly for score, poly, _ in scored if 0.35 <= score < max(0.55, top_score - 0.18)]

    try:
        gross = [g for g in getattr(unary_union(kept), "geoms", [unary_union(kept)]) if not g.is_empty]
    except Exception:
        gross = kept
    final = gross
    cut_polys = [
        obj.polygon for obj in structural_objects.cut_candidates
        if obj.confidence >= 0.75 and obj.polygon is not None and not obj.polygon.is_empty
    ]
    if cut_polys and gross:
        try:
            cut_union = unary_union(cut_polys)
            diff = unary_union(gross).difference(cut_union)
            final = [g for g in getattr(diff, "geoms", [diff]) if not g.is_empty and g.area > 1]
        except Exception:
            final = gross

    return BoundaryExtractionResult(
        gross_regions=gross,
        final_regions=final,
        wall_core_candidates=wall_like,
        boundary_evidence=all_boundary_evidence,
        boundary_signatures=boundary_signatures,
        grid_column_anchors=anchors,
        structural_objects=structural_objects,
        uncertain_candidates=uncertain,
        ignored_regions=ignored,
        confidence=top_score,
        mode_reason=(
            "evidence_guided_no_fill_boundary"
            if boundary_signatures or structural_objects.walls else "linework_polygonize"
        ),
        debug={
            "line_segments": len(polygonize_lines),
            "raw_line_segments": raw_line_count,
            "boundary_signature_count": len(boundary_signatures),
            "boundary_evidence_count": len(all_boundary_evidence),
            "boundary_signatures": boundary_signatures,
            "polygonized_regions": len(polygons),
            "scored_regions": len(scored),
            "kept_regions": len(gross),
            "net_regions": len(final),
            "top_reasons": scored[0][2],
            **structural_objects.debug,
        },
    )
