"""Viewport and no-fill slab diagnostics for plan sheets.

This layer is deliberately conservative.  It does not invent slab geometry;
it finds the drawing viewport used for area sanity checks, then explains why
a no-fill GA sheet failed to produce a closed slab boundary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import quantiles

import fitz
from PIL import ImageDraw, ImageFont
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import polygonize_full

from src.slab_v2 import trace as trace_mod


_PLAN_TITLE_RE = (
    "GENERAL ARRANGEMENT",
    "OUTLINE PLAN",
    "FLOOR PLAN",
    "FRAMING PLAN",
    "STEELWORK PLAN",
    "MARKING PLAN",
)


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _rect_payload(rect: fitz.Rect) -> dict:
    return trace_mod.content_rect_snapshot(rect)


def _safe_rect(page: fitz.Page, x0: float, y0: float,
               x1: float, y1: float) -> fitz.Rect:
    r = fitz.Rect(
        max(0.0, min(x0, page.rect.width)),
        max(0.0, min(y0, page.rect.height)),
        max(0.0, min(x1, page.rect.width)),
        max(0.0, min(y1, page.rect.height)),
    )
    if r.width <= 1 or r.height <= 1:
        return fitz.Rect(page.rect)
    return r


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) < 4:
        return sorted(values)[min(len(values) - 1, max(0, int(p * len(values))))]
    qs = quantiles(values, n=100, method="inclusive")
    idx = min(98, max(0, int(round(p * 100)) - 1))
    return qs[idx]


def _word_text(page: fitz.Page) -> tuple[list[tuple], str]:
    words = page.get_text("words") or []
    text = " ".join(str(w[4]) for w in words)
    return words, text.upper()


def _estimate_exclusion_zones(page: fitz.Page, role_audit: dict) -> tuple[float, float, list[dict]]:
    """Return right and bottom cutoffs for the drawing viewport search."""
    words, upper = _word_text(page)
    w, h = float(page.rect.width), float(page.rect.height)
    right_cut = w * 0.94
    bottom_cut = h * 0.82
    zones: list[dict] = []

    # Title blocks and tender notes usually occupy the bottom band.  Prefer a
    # title-block crop only when title-like text is actually found there.
    title_hits = [
        wd for wd in words
        if wd[1] > h * 0.62
        and any(tok in str(wd[4]).upper() for tok in {
            "GENERAL", "ARRANGEMENT", "LEVEL", "PLAN", "TENDER", "REVISION",
            "DRAWING", "SCALE", "PROJECT", "CLIENT", "ARCHITECT",
        })
    ]
    if title_hits or role_audit.get("role") == "geometry_plan":
        bottom_cut = h * 0.78
        zones.append({
            "type": "bottom_title_or_notes",
            "cutoff_y": round(bottom_cut, 3),
            "reason": "title/block/notes text found in lower sheet band",
        })

    # Legends/schedules often live in a right strip.  Do not crop aggressively
    # unless there are many words there.
    right_words = [wd for wd in words if wd[0] > w * 0.82]
    if len(right_words) >= max(8, int(len(words) * 0.08)):
        right_cut = w * 0.86
        zones.append({
            "type": "right_legend_or_schedule",
            "cutoff_x": round(right_cut, 3),
            "word_count": len(right_words),
            "reason": "dense right-side text suggests legend/schedule/title area",
        })

    if not any(tok in upper for tok in _PLAN_TITLE_RE):
        zones.append({
            "type": "weak_plan_title",
            "reason": "no strong plan-title phrase found; viewport is lower confidence",
        })
    return right_cut, bottom_cut, zones


def detect_plan_viewport(page: fitz.Page, paths: list, content_rect: fitz.Rect,
                         role_audit: dict | None = None) -> tuple[fitz.Rect, dict]:
    """Detect the main plan viewport and return an auditable decision."""
    role_audit = role_audit or {}
    page_area = float(page.rect.width * page.rect.height) or 1.0
    content_area = float(content_rect.width * content_rect.height) or page_area
    right_cut, bottom_cut, zones = _estimate_exclusion_zones(page, role_audit)

    if role_audit.get("role") == "evidence_only":
        audit = {
            "schema": "plan_viewport_v1",
            "status": "not_geometry_plan",
            "method": "role_gate",
            "viewport_rect": _rect_payload(content_rect),
            "content_rect": _rect_payload(content_rect),
            "excluded_zones": zones,
            "warnings": ["page role is evidence_only; viewport defaults to content rect"],
        }
        return fitz.Rect(content_rect), audit

    xs: list[float] = []
    ys: list[float] = []
    segment_count = 0
    for path in paths or []:
        if getattr(path, "outside_content", False):
            continue
        for a, b in getattr(path, "segments", []) or []:
            mx = (float(a[0]) + float(b[0])) / 2.0
            my = (float(a[1]) + float(b[1])) / 2.0
            if mx > right_cut or my > bottom_cut:
                continue
            xs.extend([float(a[0]), float(b[0])])
            ys.extend([float(a[1]), float(b[1])])
            segment_count += 1

    warnings: list[str] = []
    if segment_count < 20 or not xs or not ys:
        viewport = fitz.Rect(content_rect)
        status = "fallback"
        method = "content_rect"
        warnings.append("not enough vector segments inside candidate viewport")
    else:
        margin = 24.0
        x0 = _percentile(xs, 0.01) - margin
        y0 = _percentile(ys, 0.01) - margin
        x1 = _percentile(xs, 0.99) + margin
        y1 = _percentile(ys, 0.99) + margin
        viewport = _safe_rect(page, x0, y0, x1, y1)
        # Intersect with existing content rect to preserve the older legend crop.
        viewport = viewport & content_rect
        status = "detected"
        method = "vector_extent_excluding_sheet_margins"
        if viewport.width * viewport.height < page_area * 0.10:
            viewport = fitz.Rect(content_rect)
            status = "fallback"
            method = "content_rect_after_small_candidate"
            warnings.append("candidate viewport too small; using content rect")

    viewport_area = float(viewport.width * viewport.height) or content_area
    audit = {
        "schema": "plan_viewport_v1",
        "status": status,
        "method": method,
        "viewport_rect": _rect_payload(viewport),
        "content_rect": _rect_payload(content_rect),
        "page_area_pt2": round(page_area, 3),
        "viewport_area_pt2": round(viewport_area, 3),
        "content_area_pt2": round(content_area, 3),
        "viewport_area_fraction_of_content": round(viewport_area / max(content_area, 1.0), 6),
        "segment_count_used": segment_count,
        "excluded_zones": zones,
        "role": role_audit.get("role"),
        "title": role_audit.get("title") or role_audit.get("matched_title"),
        "warnings": warnings,
    }
    return viewport, audit


def write_plan_viewport_artifacts(page: fitz.Page, renderer, out_dir: Path,
                                  page_number: int, viewport: fitz.Rect,
                                  content_rect: fitz.Rect, audit: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"plan_viewport_p{page_number:02d}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        img = renderer.faded.copy()
        dr = ImageDraw.Draw(img)
        s = renderer.scale

        def rr(rect: fitz.Rect):
            return [rect.x0 * s, rect.y0 * s, rect.x1 * s, rect.y1 * s]

        dr.rectangle(rr(content_rect), outline=(255, 165, 0), width=3)
        dr.rectangle(rr(viewport), outline=(0, 180, 80), width=5)
        font = _font(18)
        dr.text((max(8, viewport.x0 * s), max(8, viewport.y0 * s - 26)),
                f"PLAN VIEWPORT: {audit.get('status')} / {audit.get('method')}",
                fill=(0, 120, 50), font=font)
        dr.text((max(8, content_rect.x0 * s), max(34, content_rect.y0 * s + 4)),
                "orange=content rect, green=plan viewport",
                fill=(80, 80, 80), font=font)
        img.save(out_dir / f"plan_viewport_p{page_number:02d}.png")
    except Exception:
        return


def analyze_slab_boundary_failure(gross, viewport: fitz.Rect, paths: list,
                                  area_ref: float) -> dict:
    """Explain which viewport sides the assembled slab did not reach."""
    viewport_w = float(viewport.width) or 1.0
    viewport_h = float(viewport.height) or 1.0
    tol_x = max(12.0, viewport_w * 0.04)
    tol_y = max(12.0, viewport_h * 0.04)
    if gross is None:
        bounds = None
        missing = ["left", "right", "top", "bottom"]
        frac = 0.0
    else:
        bx = gross.bounds
        bounds = [round(float(v), 3) for v in bx]
        missing = []
        if abs(bx[0] - viewport.x0) > tol_x:
            missing.append("left")
        if abs(viewport.x1 - bx[2]) > tol_x:
            missing.append("right")
        if abs(bx[1] - viewport.y0) > tol_y:
            missing.append("top")
        if abs(viewport.y1 - bx[3]) > tol_y:
            missing.append("bottom")
        frac = float(getattr(gross, "area", 0.0)) / max(float(area_ref), 1.0)

    long_segments = 0
    for path in paths or []:
        if getattr(path, "outside_content", False):
            continue
        for a, b in getattr(path, "segments", []) or []:
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            if not (viewport.x0 - tol_x <= ax <= viewport.x1 + tol_x
                    and viewport.y0 - tol_y <= ay <= viewport.y1 + tol_y
                    and viewport.x0 - tol_x <= bx <= viewport.x1 + tol_x
                    and viewport.y0 - tol_y <= by <= viewport.y1 + tol_y):
                continue
            if ((abs(ax - bx) > viewport_w * 0.25 and abs(ay - by) < 2.0)
                    or (abs(ay - by) > viewport_h * 0.25 and abs(ax - bx) < 2.0)):
                long_segments += 1

    return {
        "schema": "slab_boundary_failure_v1",
        "status": "unresolved" if missing else "candidate_reaches_viewport",
        "slab_fraction_of_viewport": round(frac, 6),
        "gross_bounds": bounds,
        "viewport_rect": _rect_payload(viewport),
        "missing_boundary_sides": missing,
        "long_axis_segments_in_viewport": long_segments,
        "reason": (
            "assembled slab does not form a closed main slab boundary inside "
            "the plan viewport; no-fill GA resolver stayed fail-closed"
            if missing else
            "assembled slab reaches viewport envelope but failed area threshold"
        ),
    }


def _merge_intervals(intervals: list[tuple[float, float]],
                     gap_tol: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        pa, pb = merged[-1]
        if a <= pb + gap_tol:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def _interval_coverage(intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in intervals)


def _seg_len(a, b) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _pt_payload(pt) -> list[float]:
    return [round(float(pt[0]), 3), round(float(pt[1]), 3)]


def _seg_payload(a, b, **extra) -> dict:
    out = {
        "a": _pt_payload(a),
        "b": _pt_payload(b),
        "length_pt": round(_seg_len(a, b), 3),
    }
    out.update(extra)
    return out


def _inside_rect(rect: fitz.Rect, x: float, y: float, tol: float = 0.0) -> bool:
    return (
        rect.x0 - tol <= x <= rect.x1 + tol
        and rect.y0 - tol <= y <= rect.y1 + tol
    )


def _collect_irregular_edge_candidates(paths: list, viewport: fitz.Rect,
                                       min_seg: float,
                                       area_ref: float) -> tuple[list, list, dict, dict]:
    """Collect solid vector edges usable for no-fill slab polygonization.

    The filter is intentionally style-agnostic for consultant robustness: it
    rejects only known non-geometry sources (outside viewport, fill-only,
    dashed/reference-like paths, tiny segments). Later scoring decides whether
    the resulting closed polygons are slab-like.
    """
    tol = max(8.0, min(float(viewport.width), float(viewport.height)) * 0.01)
    viewport_poly = Polygon([
        (viewport.x0, viewport.y0), (viewport.x1, viewport.y0),
        (viewport.x1, viewport.y1), (viewport.x0, viewport.y1),
    ])
    segments: list[tuple] = []
    segments_by_style: dict[str, list[tuple]] = {}
    accepted_payload: list[dict] = []
    rejected_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    seen = set()

    def add_segment(path, a, b, source: str,
                    path_outside_content: bool) -> bool:
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
        if not _inside_rect(viewport, mx, my, tol):
            reject("outside_viewport")
            return False
        length = _seg_len((ax, ay), (bx, by))
        if length < min_seg:
            reject("too_short")
            return False
        ka = (round(ax, 3), round(ay, 3))
        kb = (round(bx, 3), round(by, 3))
        key_seg = (ka, kb) if ka <= kb else (kb, ka)
        if key_seg in seen:
            reject("duplicate")
            return False
        seen.add(key_seg)
        seg = ((ax, ay), (bx, by))
        segments.append(seg)
        style_key = str(getattr(path, "style_id", "unknown"))
        segments_by_style.setdefault(style_key, []).append(seg)
        if len(accepted_payload) < 500:
            accepted_payload.append(
                _seg_payload(
                    seg[0], seg[1],
                    path_id=getattr(path, "id", None),
                    style_id=getattr(path, "style_id", None),
                    source=source,
                    legacy_outside_content=path_outside_content,
                )
            )
        return True

    def add_fill_boundary(path, path_outside_content: bool) -> bool:
        poly = getattr(path, "fill_polygon", None)
        if poly is None or getattr(poly, "is_empty", True):
            reject("fill_only_without_polygon")
            return False
        try:
            inter_area = float(poly.intersection(viewport_poly).area)
        except Exception:
            reject("fill_boundary_invalid")
            return False
        area_frac = inter_area / max(float(area_ref), 1.0)
        if area_frac < 0.002:
            reject("fill_boundary_too_small")
            return False
        if area_frac > 1.08:
            reject("fill_boundary_too_large")
            return False
        try:
            coords = list(poly.exterior.coords)
        except Exception:
            reject("fill_boundary_no_exterior")
            return False
        added = 0
        for i in range(len(coords) - 1):
            if add_segment(path, coords[i], coords[i + 1], "fill_boundary",
                           path_outside_content):
                added += 1
        return added > 0

    for path in paths or []:
        # Do not reject the whole path just because the legacy content crop
        # marked it outside.  GA/no-fill sheets often have a bad content rect;
        # the detected plan viewport is the stronger boundary for this resolver.
        path_outside_content = bool(getattr(path, "outside_content", False))
        key = getattr(path, "key", None)
        if key is not None and getattr(key, "dashes", ""):
            reject("dashed_or_reference")
            continue
        if not getattr(path, "has_stroke", True):
            if add_fill_boundary(path, path_outside_content):
                continue
            reject("fill_only")
            continue
        if key is not None and getattr(key, "stroke", None) is None:
            if add_fill_boundary(path, path_outside_content):
                continue
            reject("no_stroke")
            continue
        for a, b in getattr(path, "segments", []) or []:
            add_segment(path, a, b, "stroke_segment", path_outside_content)
    return segments, accepted_payload, rejected_counts, segments_by_style


def _build_gap_closures(segments: list, viewport: fitz.Rect,
                        gap_tol: float, axis_tol: float) -> tuple[list, list]:
    """Add auditable synthetic closures only across small plausible gaps."""
    endpoints: list[tuple[float, float]] = []
    for a, b in segments:
        endpoints.append((round(float(a[0]), 3), round(float(a[1]), 3)))
        endpoints.append((round(float(b[0]), 3), round(float(b[1]), 3)))
    unique = sorted(set(endpoints))
    closures: list[tuple] = []
    payload: list[dict] = []
    used: set[tuple] = set()
    # O(n^2) is acceptable here because endpoint counts on plan edges are
    # modest after filtering. Cap work defensively for very dense sheets.
    max_points = 2500
    if len(unique) > max_points:
        return [], [{
            "status": "skipped",
            "reason": "too_many_endpoints_for_gap_closure",
            "endpoint_count": len(unique),
        }]
    for i, p in enumerate(unique):
        for q in unique[i + 1:]:
            dx = abs(q[0] - p[0])
            dy = abs(q[1] - p[1])
            if dx > gap_tol and dy > gap_tol:
                continue
            dist = math.hypot(dx, dy)
            if dist <= 1e-6 or dist > gap_tol:
                continue
            axis_aligned = dx <= axis_tol or dy <= axis_tol
            tiny_corner = dist <= gap_tol * 0.45
            if not (axis_aligned or tiny_corner):
                continue
            mx, my = (p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5
            if not _inside_rect(viewport, mx, my, 1.0):
                continue
            pair_key = (p, q)
            if pair_key in used:
                continue
            used.add(pair_key)
            closures.append((p, q))
            if len(payload) < 300:
                payload.append(_seg_payload(
                    p, q,
                    source="gap_closure",
                    closure_type="axis" if axis_aligned else "tiny_corner",
                    distance_pt=round(dist, 3)))
    return closures, payload


def _score_no_fill_polygon(poly: Polygon, viewport: fitz.Rect,
                           area_ref: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    area_frac = float(poly.area) / max(float(area_ref), 1.0)
    if area_frac < 0.08:
        return 0.0, ["area_too_small"]
    if area_frac > 1.05:
        return 0.0, ["area_exceeds_viewport"]

    min_dim = max(1.0, min(float(viewport.width), float(viewport.height)))
    bounds = poly.bounds
    reach_tol_x = max(20.0, float(viewport.width) * 0.08)
    reach_tol_y = max(20.0, float(viewport.height) * 0.08)
    reached = 0
    if abs(bounds[0] - viewport.x0) <= reach_tol_x:
        reached += 1
        reasons.append("reaches_left")
    if abs(viewport.x1 - bounds[2]) <= reach_tol_x:
        reached += 1
        reasons.append("reaches_right")
    if abs(bounds[1] - viewport.y0) <= reach_tol_y:
        reached += 1
        reasons.append("reaches_top")
    if abs(viewport.y1 - bounds[3]) <= reach_tol_y:
        reached += 1
        reasons.append("reaches_bottom")

    area_score = min(1.0, area_frac / 0.35)
    if area_frac > 0.90:
        area_score *= max(0.25, 1.0 - (area_frac - 0.90) * 4.0)
        reasons.append("very_large_area")
    reach_score = min(1.0, reached / 3.0)
    compactness = 0.0
    if poly.length > 0:
        compactness = min(1.0, (4.0 * math.sqrt(max(poly.area, 1.0))) / poly.length)
    hole_penalty = min(0.25, len(poly.interiors) * 0.05)
    complexity_penalty = 0.0
    if len(poly.exterior.coords) > 240:
        complexity_penalty = 0.15
        reasons.append("high_vertex_count")

    score = 0.52 * area_score + 0.33 * reach_score + 0.15 * compactness
    score = max(0.0, score - hole_penalty - complexity_penalty)
    if area_frac >= 0.10:
        reasons.append("area_above_min")
    if compactness >= 0.25:
        reasons.append("reasonable_compactness")
    if min(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) < min_dim * 0.12:
        score *= 0.5
        reasons.append("thin_fragment_penalty")
    return round(score, 4), reasons


def _polygonize_no_fill_segments(
    graph_segments: list,
    viewport: fitz.Rect,
    area_ref: float,
    snap_grid: float,
    *,
    id_prefix: str = "nofill_poly",
) -> tuple[list[dict], dict[str, Polygon], dict]:
    """Polygonize a candidate line graph and score slab-like faces.

    Keeping this as a helper lets the resolver try isolated vector style
    families before falling back to the full, noisy sheet graph.  Dense GA
    drawings often contain many valid closed annotation loops that make the
    combined graph unstable; style isolation keeps real perimeter evidence
    intact while reducing topology noise.
    """
    stats: dict = {
        "status": "ok",
        "dangle_count": 0,
        "cut_edge_count": 0,
        "invalid_count": 0,
    }
    candidate_rows: list[dict] = []
    polygons_by_id: dict[str, Polygon] = {}
    if len(graph_segments) < 4:
        stats.update({"status": "skipped", "reason": "not_enough_segments"})
        return candidate_rows, polygons_by_id, stats
    try:
        mls = MultiLineString(graph_segments)
        noded = shapely.node(shapely.set_precision(mls, grid_size=snap_grid))
        faces, dangles, cuts, invalids = polygonize_full([noded])
    except Exception as exc:
        stats.update({"status": "error", "reason": f"polygonize_failed:{exc}"})
        return candidate_rows, polygons_by_id, stats

    viewport_poly = Polygon([
        (viewport.x0, viewport.y0), (viewport.x1, viewport.y0),
        (viewport.x1, viewport.y1), (viewport.x0, viewport.y1),
    ])
    stats["dangle_count"] = len(getattr(dangles, "geoms", []))
    stats["cut_edge_count"] = len(getattr(cuts, "geoms", []))
    stats["invalid_count"] = len(getattr(invalids, "geoms", []))

    for idx, geom in enumerate(faces.geoms):
        if geom.is_empty:
            continue
        poly = geom if isinstance(geom, Polygon) else Polygon(geom)
        if not poly.is_valid or poly.area <= 0:
            continue
        inter_area = poly.intersection(viewport_poly).area
        containment = inter_area / max(poly.area, 1.0)
        if containment < 0.92:
            continue
        score, reasons = _score_no_fill_polygon(poly, viewport, area_ref)
        cid = f"{id_prefix}_{idx:03d}"
        row = {
            "id": cid,
            "area_pt2": round(float(poly.area), 3),
            "area_fraction_of_viewport": round(float(poly.area) / max(float(area_ref), 1.0), 6),
            "bounds": [round(float(v), 3) for v in poly.bounds],
            "vertex_count": len(poly.exterior.coords),
            "hole_count": len(poly.interiors),
            "viewport_containment_ratio": round(containment, 4),
            "score": score,
            "score_reasons": reasons,
        }
        candidate_rows.append(row)
        polygons_by_id[cid] = poly
    candidate_rows.sort(key=lambda r: r["score"], reverse=True)
    return candidate_rows, polygons_by_id, stats


def _cluster_axis_segments(rows: list[tuple[float, float, float]], tol: float) -> list[dict]:
    """Cluster axis-aligned segment intervals by their constant coordinate."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r[0])
    clusters: list[dict] = []
    for coord, a, b in rows:
        lo, hi = sorted((float(a), float(b)))
        if not clusters or abs(coord - clusters[-1]["coord"]) > tol:
            clusters.append({"coord": float(coord), "intervals": [(lo, hi)]})
            continue
        cur = clusters[-1]
        n = len(cur["intervals"])
        cur["coord"] = (cur["coord"] * n + float(coord)) / (n + 1)
        cur["intervals"].append((lo, hi))

    for cur in clusters:
        intervals = sorted(cur["intervals"])
        merged: list[tuple[float, float]] = []
        for lo, hi in intervals:
            if not merged or lo > merged[-1][1] + tol:
                merged.append((lo, hi))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        cur["merged_intervals"] = merged
        cur["coverage"] = _interval_coverage(merged)
        cur["min"] = min((i[0] for i in merged), default=0.0)
        cur["max"] = max((i[1] for i in merged), default=0.0)
    return clusters


def _assemble_supported_envelope_candidate(
    segments: list,
    viewport: fitz.Rect,
    area_ref: float,
    axis_tol: float,
) -> tuple[Polygon | None, dict]:
    """Fallback candidate from real long structural edge support.

    This is deliberately narrower than "take the viewport rectangle": it only
    returns a polygon when vector linework contains substantial horizontal and
    vertical side coverage forming a bounded drawing envelope.  It helps GA
    pages where the main outline is not a closed face because intersections are
    not split cleanly.
    """
    vw = float(viewport.width) or 1.0
    vh = float(viewport.height) or 1.0
    min_dim = min(vw, vh)
    h_rows: list[tuple[float, float, float]] = []
    v_rows: list[tuple[float, float, float]] = []
    min_long = max(60.0, min_dim * 0.07)
    for a, b in segments:
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = abs(bx - ax), abs(by - ay)
        length = math.hypot(dx, dy)
        if length < min_long:
            continue
        if dy <= axis_tol:
            h_rows.append(((ay + by) * 0.5, ax, bx))
        elif dx <= axis_tol:
            v_rows.append(((ax + bx) * 0.5, ay, by))

    # GA slabs often draw perimeter as paired/offset lines.  Cluster a little
    # wider than endpoint snapping so those double-lines contribute to one
    # supported side, while the max-gap guard below still blocks large guesses.
    cluster_tol = max(12.0, min_dim * 0.02)
    h_clusters = _cluster_axis_segments(h_rows, cluster_tol)
    v_clusters = _cluster_axis_segments(v_rows, cluster_tol)
    audit = {
        "method": "supported_axis_envelope",
        "horizontal_cluster_count": len(h_clusters),
        "vertical_cluster_count": len(v_clusters),
        "status": "unresolved",
        "reason": "",
    }
    if len(h_clusters) < 2 or len(v_clusters) < 2:
        audit["reason"] = "not_enough_axis_clusters"
        return None, audit

    def span_stats(cluster: dict, span0: float, span1: float) -> tuple[float, float, dict]:
        """Return coverage ratio, largest uncovered gap, and endpoint support."""
        span0, span1 = sorted((float(span0), float(span1)))
        clipped: list[tuple[float, float]] = []
        for lo, hi in cluster.get("merged_intervals", []):
            a, b = max(float(lo), span0), min(float(hi), span1)
            if b > a:
                clipped.append((a, b))
        if not clipped:
            return 0.0, span1 - span0, {
                "start_gap": span1 - span0,
                "end_gap": span1 - span0,
                "internal_gap": span1 - span0,
                "endpoint_supported": False,
            }
        clipped.sort()
        coverage = _interval_coverage(clipped)
        start_gap = max(0.0, clipped[0][0] - span0)
        end_gap = max(0.0, span1 - clipped[-1][1])
        internal_gaps: list[float] = []
        for prev, cur in zip(clipped, clipped[1:]):
            internal_gaps.append(max(0.0, cur[0] - prev[1]))
        max_gap = max([start_gap, end_gap, *internal_gaps] or [0.0])
        endpoint_tol = max(axis_tol * 2.0, (span1 - span0) * 0.08, 24.0)
        return coverage / max(span1 - span0, 1.0), max_gap, {
            "start_gap": start_gap,
            "end_gap": end_gap,
            "internal_gap": max(internal_gaps or [0.0]),
            "endpoint_supported": start_gap <= endpoint_tol and end_gap <= endpoint_tol,
        }

    best: tuple[float, dict, dict, dict, dict] | None = None
    min_width = vw * 0.35
    min_height = vh * 0.18
    for top in h_clusters:
        for bottom in h_clusters:
            if float(bottom["coord"]) <= float(top["coord"]):
                continue
            y0, y1 = float(top["coord"]), float(bottom["coord"])
            height = y1 - y0
            if height < min_height:
                continue
            x_min = max(min(float(top["min"]), float(bottom["min"])), viewport.x0)
            x_max = min(max(float(top["max"]), float(bottom["max"])), viewport.x1)
            width = x_max - x_min
            if width < min_width:
                continue
            top_cov, top_gap, top_span = span_stats(top, x_min, x_max)
            bottom_cov, bottom_gap, bottom_span = span_stats(bottom, x_min, x_max)
            if min(top_cov, bottom_cov) < 0.42:
                continue
            max_h_gap = max(60.0, width * 0.08)
            dense_ga_edgework = len(segments) >= 80
            top_gap_ok = top_gap <= max_h_gap or (
                dense_ga_edgework
                and top_span["endpoint_supported"]
                and top_span["internal_gap"] <= max(120.0, width * 0.20)
                # Irregular GA/no-fill boundaries often have perimeter lines
                # interrupted by columns, core edges, or reference callouts.
                # Endpoint support plus bounded internal gaps is stronger than
                # raw interval coverage, so allow a lower coverage gate only in
                # dense edgework sheets.
                and top_cov >= 0.62
            )
            bottom_gap_ok = bottom_gap <= max_h_gap or (
                dense_ga_edgework
                and bottom_span["endpoint_supported"]
                and bottom_span["internal_gap"] <= max(120.0, width * 0.20)
                and bottom_cov >= 0.62
            )
            if not (top_gap_ok and bottom_gap_ok):
                continue
            for left in v_clusters:
                x_left = float(left["coord"])
                if abs(x_left - x_min) > max(60.0, width * 0.08):
                    continue
                left_cov, left_gap, left_span = span_stats(left, y0, y1)
                if left_cov < 0.28:
                    continue
                for right in v_clusters:
                    x_right = float(right["coord"])
                    if x_right <= x_left:
                        continue
                    if abs(x_right - x_max) > max(60.0, width * 0.08):
                        continue
                    right_cov, right_gap, right_span = span_stats(right, y0, y1)
                    if right_cov < 0.28:
                        continue
                    max_v_gap = max(60.0, height * 0.18)
                    left_gap_ok = left_gap <= max_v_gap or (
                        left_span["endpoint_supported"]
                        and left_span["internal_gap"] <= max(90.0, height * 0.30)
                        and left_cov >= 0.45
                    )
                    right_gap_ok = right_gap <= max_v_gap or (
                        right_span["endpoint_supported"]
                        and right_span["internal_gap"] <= max(90.0, height * 0.30)
                        and right_cov >= 0.45
                    )
                    if not (left_gap_ok and right_gap_ok):
                        continue
                    side_score = min(1.0, (top_cov + bottom_cov + left_cov + right_cov) / 2.8)
                    area_frac = (width * height) / max(float(area_ref), 1.0)
                    if area_frac < 0.08 or area_frac > 0.90:
                        continue
                    score = side_score + min(0.35, area_frac)
                    if best is None or score > best[0]:
                        best = (score, top, bottom, left, right)

    if best is None:
        audit["reason"] = "no_supported_outer_envelope"
        audit["horizontal_clusters"] = [
            {"coord": round(c["coord"], 3), "coverage": round(c["coverage"], 3),
             "min": round(c["min"], 3), "max": round(c["max"], 3)}
            for c in sorted(h_clusters, key=lambda c: c["coverage"], reverse=True)[:8]
        ]
        audit["vertical_clusters"] = [
            {"coord": round(c["coord"], 3), "coverage": round(c["coverage"], 3),
             "min": round(c["min"], 3), "max": round(c["max"], 3)}
            for c in sorted(v_clusters, key=lambda c: c["coverage"], reverse=True)[:8]
        ]
        return None, audit

    score, top, bottom, left, right = best
    y0, y1 = sorted((float(top["coord"]), float(bottom["coord"])))
    x0, x1 = sorted((float(left["coord"]), float(right["coord"])))
    poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    poly = poly.intersection(Polygon([
        (viewport.x0, viewport.y0), (viewport.x1, viewport.y0),
        (viewport.x1, viewport.y1), (viewport.x0, viewport.y1),
    ])).buffer(0)
    if poly.is_empty or not isinstance(poly, Polygon):
        audit["reason"] = "supported_envelope_invalid_after_clip"
        return None, audit
    area_frac = float(poly.area) / max(float(area_ref), 1.0)
    audit.update({
        "status": "verified",
        "reason": "four supported structural sides form a no-fill envelope",
        "score": round(float(score), 4),
        "bounds": [round(float(v), 3) for v in poly.bounds],
        "area_pt2": round(float(poly.area), 3),
        "area_fraction_of_viewport": round(area_frac, 6),
        "top": {"coord": round(top["coord"], 3), "coverage": round(top["coverage"], 3)},
        "bottom": {"coord": round(bottom["coord"], 3), "coverage": round(bottom["coverage"], 3)},
        "left": {"coord": round(left["coord"], 3), "coverage": round(left["coverage"], 3)},
        "right": {"coord": round(right["coord"], 3), "coverage": round(right["coverage"], 3)},
    })
    return poly, audit


def _assemble_relaxed_supported_envelope_candidate(
    segments: list,
    viewport: fitz.Rect,
    area_ref: float,
    axis_tol: float,
) -> tuple[Polygon | None, dict]:
    """Fallback for GA/no-fill plans with fragmented perimeter linework.

    The stricter supported envelope derives x extents from the selected top and
    bottom edges.  That is deliberately conservative, but it rejects common GA
    sheets where the top/bottom slab edges are interrupted by ramps, columns,
    or local setdown callouts while the true left/right perimeter is still
    visible.  This pass chooses all four sides independently, then verifies the
    candidate rectangle against real side coverage.  It still refuses viewport
    rectangles and reference-only drawings by requiring structural line support
    on every side.
    """
    vw = float(viewport.width) or 1.0
    vh = float(viewport.height) or 1.0
    min_dim = min(vw, vh)
    min_long = max(60.0, min_dim * 0.07)
    h_rows: list[tuple[float, float, float]] = []
    v_rows: list[tuple[float, float, float]] = []
    for a, b in segments:
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dx, dy = abs(bx - ax), abs(by - ay)
        length = math.hypot(dx, dy)
        if length < min_long:
            continue
        if dy <= axis_tol:
            coord = (ay + by) * 0.5
            if viewport.y0 - axis_tol <= coord <= viewport.y1 + axis_tol:
                h_rows.append((coord, ax, bx))
        elif dx <= axis_tol:
            coord = (ax + bx) * 0.5
            if viewport.x0 - axis_tol <= coord <= viewport.x1 + axis_tol:
                v_rows.append((coord, ay, by))

    cluster_tol = max(12.0, min_dim * 0.02)
    h_clusters = _cluster_axis_segments(h_rows, cluster_tol)
    v_clusters = _cluster_axis_segments(v_rows, cluster_tol)
    audit = {
        "method": "relaxed_supported_axis_envelope",
        "horizontal_cluster_count": len(h_clusters),
        "vertical_cluster_count": len(v_clusters),
        "status": "unresolved",
        "reason": "",
    }
    if len(segments) < 7:
        audit["reason"] = "relaxed_envelope_requires_dense_fragmented_edgework"
        return None, audit
    if len(h_clusters) < 2 or len(v_clusters) < 2:
        audit["reason"] = "not_enough_axis_clusters"
        return None, audit

    def clipped_coverage(cluster: dict, span0: float, span1: float) -> tuple[float, float, dict]:
        span0, span1 = sorted((float(span0), float(span1)))
        clipped: list[tuple[float, float]] = []
        for lo, hi in cluster.get("merged_intervals", []):
            a, b = max(float(lo), span0), min(float(hi), span1)
            if b > a:
                clipped.append((a, b))
        if not clipped:
            return 0.0, span1 - span0, {
                "start_gap": span1 - span0,
                "end_gap": span1 - span0,
                "internal_gap": span1 - span0,
            }
        clipped.sort()
        coverage = _interval_coverage(clipped)
        start_gap = max(0.0, clipped[0][0] - span0)
        end_gap = max(0.0, span1 - clipped[-1][1])
        internal_gap = max(
            [max(0.0, cur[0] - prev[1]) for prev, cur in zip(clipped, clipped[1:])]
            or [0.0]
        )
        return coverage / max(span1 - span0, 1.0), max(start_gap, end_gap, internal_gap), {
            "start_gap": start_gap,
            "end_gap": end_gap,
            "internal_gap": internal_gap,
        }

    candidates: list[tuple[float, dict, dict, dict, dict, dict]] = []
    min_width = vw * 0.35
    min_height = vh * 0.22
    for top in h_clusters:
        for bottom in h_clusters:
            if float(bottom["coord"]) <= float(top["coord"]):
                continue
            y0, y1 = float(top["coord"]), float(bottom["coord"])
            height = y1 - y0
            if height < min_height:
                continue
            # Avoid choosing the detected viewport border as the slab.  A real
            # slab side can be near the viewport edge, but using all viewport
            # edges is nearly always a sheet/frame artefact.
            near_top = abs(y0 - float(viewport.y0)) <= max(16.0, vh * 0.015)
            near_bottom = abs(y1 - float(viewport.y1)) <= max(16.0, vh * 0.015)
            for left in v_clusters:
                for right in v_clusters:
                    if float(right["coord"]) <= float(left["coord"]):
                        continue
                    x0, x1 = float(left["coord"]), float(right["coord"])
                    width = x1 - x0
                    if width < min_width:
                        continue
                    near_left = abs(x0 - float(viewport.x0)) <= max(16.0, vw * 0.015)
                    near_right = abs(x1 - float(viewport.x1)) <= max(16.0, vw * 0.015)
                    if near_top and near_bottom and near_left and near_right:
                        continue
                    area_frac = (width * height) / max(float(area_ref), 1.0)
                    if area_frac < 0.10 or area_frac > 0.86:
                        continue
                    top_cov, top_gap, top_span = clipped_coverage(top, x0, x1)
                    bottom_cov, bottom_gap, bottom_span = clipped_coverage(bottom, x0, x1)
                    left_cov, left_gap, left_span = clipped_coverage(left, y0, y1)
                    right_cov, right_gap, right_span = clipped_coverage(right, y0, y1)
                    covs = [top_cov, bottom_cov, left_cov, right_cov]
                    if min(top_cov, bottom_cov) < 0.30:
                        continue
                    if min(left_cov, right_cov) < 0.18:
                        continue
                    if sum(covs) < 1.65:
                        continue
                    max_h_gap = max(160.0, width * 0.30)
                    max_v_gap = max(120.0, height * 0.38)
                    if max(top_gap, bottom_gap) > max_h_gap:
                        continue
                    if max(left_gap, right_gap) > max_v_gap:
                        continue
                    score = (
                        min(0.75, sum(covs) / 4.0)
                        + min(0.20, area_frac * 0.25)
                        + (0.05 if min(covs) >= 0.30 else 0.0)
                    )
                    candidates.append((score, top, bottom, left, right, {
                        "area_fraction_of_viewport": area_frac,
                        "coverage": {
                            "top": top_cov,
                            "bottom": bottom_cov,
                            "left": left_cov,
                            "right": right_cov,
                        },
                        "max_gaps": {
                            "top": top_gap,
                            "bottom": bottom_gap,
                            "left": left_gap,
                            "right": right_gap,
                        },
                        "spans": {
                            "top": top_span,
                            "bottom": bottom_span,
                            "left": left_span,
                            "right": right_span,
                        },
                    }))

    if not candidates:
        audit["reason"] = "no_relaxed_supported_envelope"
        audit["horizontal_clusters"] = [
            {"coord": round(c["coord"], 3), "coverage": round(c["coverage"], 3),
             "min": round(c["min"], 3), "max": round(c["max"], 3)}
            for c in sorted(h_clusters, key=lambda c: c["coverage"], reverse=True)[:10]
        ]
        audit["vertical_clusters"] = [
            {"coord": round(c["coord"], 3), "coverage": round(c["coverage"], 3),
             "min": round(c["min"], 3), "max": round(c["max"], 3)}
            for c in sorted(v_clusters, key=lambda c: c["coverage"], reverse=True)[:10]
        ]
        return None, audit

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, top, bottom, left, right, details = candidates[0]

    def _bounds_of(candidate: tuple) -> tuple[float, float, float, float]:
        return (
            float(candidate[3]["coord"]),
            float(candidate[1]["coord"]),
            float(candidate[4]["coord"]),
            float(candidate[2]["coord"]),
        )

    def _is_contained_alternative(best_bounds: tuple[float, float, float, float],
                                  alt_bounds: tuple[float, float, float, float]) -> bool:
        bx0, by0, bx1, by1 = best_bounds
        ax0, ay0, ax1, ay1 = alt_bounds
        tol = max(axis_tol * 2.0, 20.0)
        best_area = max((bx1 - bx0) * (by1 - by0), 1.0)
        alt_area = max((ax1 - ax0) * (ay1 - ay0), 0.0)
        return (
            ax0 >= bx0 - tol and ay0 >= by0 - tol
            and ax1 <= bx1 + tol and ay1 <= by1 + tol
            and alt_area <= best_area * 0.88
        )

    def _has_strong_side_support(candidate: tuple) -> bool:
        coverage = candidate[5].get("coverage", {})
        if not coverage:
            return False
        return min(float(v) for v in coverage.values()) >= 0.55

    # If two very different envelopes are equally plausible, stay fail-closed.
    if (
        len(candidates) > 1
        and candidates[1][0] >= score - 0.035
        and _has_strong_side_support(candidates[1])
        and not _is_contained_alternative(_bounds_of(candidates[0]), _bounds_of(candidates[1]))
    ):
        audit["reason"] = "ambiguous_relaxed_envelopes"
        audit["best"] = {
            "score": round(score, 4),
            "bounds": [
                round(float(left["coord"]), 3), round(float(top["coord"]), 3),
                round(float(right["coord"]), 3), round(float(bottom["coord"]), 3),
            ],
            **details,
        }
        audit["runner_up"] = {
            "score": round(candidates[1][0], 4),
            "bounds": [
                round(float(candidates[1][3]["coord"]), 3),
                round(float(candidates[1][1]["coord"]), 3),
                round(float(candidates[1][4]["coord"]), 3),
                round(float(candidates[1][2]["coord"]), 3),
            ],
            **candidates[1][5],
        }
        return None, audit

    y0, y1 = sorted((float(top["coord"]), float(bottom["coord"])))
    x0, x1 = sorted((float(left["coord"]), float(right["coord"])))
    poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).intersection(Polygon([
        (viewport.x0, viewport.y0), (viewport.x1, viewport.y0),
        (viewport.x1, viewport.y1), (viewport.x0, viewport.y1),
    ])).buffer(0)
    if poly.is_empty or not isinstance(poly, Polygon):
        audit["reason"] = "relaxed_envelope_invalid_after_clip"
        return None, audit
    area_frac = float(poly.area) / max(float(area_ref), 1.0)
    audit.update({
        "status": "verified",
        "reason": "independently supported structural sides form a no-fill envelope",
        "score": round(float(score), 4),
        "bounds": [round(float(v), 3) for v in poly.bounds],
        "area_pt2": round(float(poly.area), 3),
        "area_fraction_of_viewport": round(area_frac, 6),
        **details,
    })
    return poly, audit


def assemble_irregular_no_fill_slab_boundary(
    paths: list,
    viewport: fitz.Rect,
    area_ref: float,
    *,
    snap_grid: float = 0.05,
) -> tuple[Polygon | None, dict]:
    """Assemble irregular no-fill GA slab boundaries from solid vector edges.

    This resolver is more capable than the four-side v1 fallback, but still
    fail-closed.  It accepts only a closed polygon with slab-like area and
    viewport support; otherwise it returns an audit explaining why no geometry
    was exported.
    """
    vw = float(viewport.width) or 1.0
    vh = float(viewport.height) or 1.0
    min_dim = min(vw, vh)
    min_seg = max(6.0, min_dim * 0.006)
    gap_tol = max(8.0, min(36.0, min_dim * 0.018))
    axis_tol = max(2.0, min_dim * 0.004)

    segments, accepted_payload, rejected_counts, segments_by_style = (
        _collect_irregular_edge_candidates(paths, viewport, min_seg, area_ref))
    closures, closure_payload = _build_gap_closures(
        segments, viewport, gap_tol, axis_tol)
    graph_segments = segments + closures

    audit = {
        "schema": "no_fill_slab_boundary_v2",
        "method": "irregular_structural_outline_assembly",
        "viewport_rect": _rect_payload(viewport),
        "viewport_area_pt2": round(float(area_ref), 3),
        "edge_candidate_count": len(segments),
        "edge_candidates_sample": accepted_payload,
        "rejected_edge_counts": rejected_counts,
        "gap_closure_count": len(closures),
        "gap_closures": closure_payload,
        "style_family_attempts": [],
        "polygon_candidates": [],
        "status": "unresolved",
        "reason": "",
    }

    style_attempt_candidates: list[tuple[dict, Polygon, dict]] = []
    style_summaries: list[dict] = []
    for style_id, style_segments in sorted(
        segments_by_style.items(),
        key=lambda item: sum(_seg_len(a, b) for a, b in item[1]),
        reverse=True,
    )[:10]:
        if len(style_segments) < 4:
            continue
        style_closures, style_closure_payload = _build_gap_closures(
            style_segments, viewport, gap_tol, axis_tol)
        style_graph = style_segments + style_closures
        rows, polys, stats = _polygonize_no_fill_segments(
            style_graph, viewport, area_ref, snap_grid,
            id_prefix=f"style_{style_id}_poly")
        best = rows[0] if rows else None
        total_len = sum(_seg_len(a, b) for a, b in style_segments)
        summary = {
            "style_id": style_id,
            "segment_count": len(style_segments),
            "total_length_pt": round(total_len, 3),
            "gap_closure_count": len(style_closures),
            "polygonize": stats,
            "best_candidate": best,
            "gap_closures_sample": style_closure_payload[:20],
        }
        style_summaries.append(summary)
        for row in rows[:4]:
            poly = polys.get(row["id"])
            if poly is not None:
                style_attempt_candidates.append((row, poly, summary))
    audit["style_family_attempts"] = style_summaries

    style_attempt_candidates.sort(key=lambda item: item[0]["score"], reverse=True)
    if style_attempt_candidates:
        best_row, best_poly, best_summary = style_attempt_candidates[0]
        runner = style_attempt_candidates[1][0] if len(style_attempt_candidates) > 1 else None
        if best_row["score"] >= 0.58 and not (
            runner is not None and runner["score"] >= best_row["score"] - 0.04
        ):
            selected = best_poly.buffer(0)
            if selected.is_valid and not selected.is_empty:
                audit["status"] = "verified"
                audit["reason"] = "style-isolated no-fill slab outline passed score gate"
                audit["selected_candidate"] = {
                    **best_row,
                    "source": "style_isolated_polygonize",
                    "style_id": best_summary.get("style_id"),
                }
                audit["candidate_area_fraction_of_viewport"] = best_row["area_fraction_of_viewport"]
                audit["candidate_bounds"] = best_row["bounds"]
                return selected, audit
        elif best_row["score"] >= 0.58:
            audit["style_family_ambiguity"] = {
                "best_candidate": best_row,
                "runner_up_candidate": runner,
                "reason": "multiple style-isolated candidates scored too close",
            }

    envelope_poly, envelope_audit = _assemble_supported_envelope_candidate(
        segments, viewport, area_ref, axis_tol)
    audit["supported_envelope_candidate"] = envelope_audit
    relaxed_envelope_poly: Polygon | None = None
    relaxed_envelope_audit: dict = {}
    if envelope_poly is None or envelope_audit.get("status") != "verified":
        relaxed_envelope_poly, relaxed_envelope_audit = (
            _assemble_relaxed_supported_envelope_candidate(
                segments, viewport, area_ref, axis_tol))
        audit["relaxed_supported_envelope_candidate"] = relaxed_envelope_audit
        if (
            relaxed_envelope_poly is not None
            and relaxed_envelope_audit.get("status") == "verified"
        ):
            envelope_poly = relaxed_envelope_poly
            envelope_audit = relaxed_envelope_audit
    if (
        envelope_poly is not None
        and envelope_audit.get("status") == "verified"
        and len(graph_segments) > 1200
    ):
        audit["status"] = "verified"
        audit["reason"] = "supported envelope accepted before expensive polygonize"
        audit["selected_candidate"] = {
            "id": "supported_envelope",
            "area_pt2": envelope_audit.get("area_pt2"),
            "area_fraction_of_viewport": envelope_audit.get(
                "area_fraction_of_viewport"),
            "bounds": envelope_audit.get("bounds"),
            "score": envelope_audit.get("score"),
            "score_reasons": [envelope_audit.get("reason"),
                              "dense_graph_fast_path"],
        }
        return envelope_poly, audit
    if len(graph_segments) < 4:
        if envelope_poly is not None and envelope_audit.get("status") == "verified":
            audit["status"] = "verified"
            audit["reason"] = "supported envelope accepted with sparse graph"
            audit["selected_candidate"] = {
                "id": "supported_envelope",
                "area_pt2": envelope_audit.get("area_pt2"),
                "area_fraction_of_viewport": envelope_audit.get("area_fraction_of_viewport"),
                "bounds": envelope_audit.get("bounds"),
                "score": envelope_audit.get("score"),
                "score_reasons": [envelope_audit.get("reason")],
            }
            return envelope_poly, audit
        audit["reason"] = "not_enough_solid_edges_in_viewport"
        return None, audit

    candidate_rows, polygons_by_id, poly_stats = _polygonize_no_fill_segments(
        graph_segments, viewport, area_ref, snap_grid)
    audit["dangle_count"] = poly_stats.get("dangle_count")
    audit["cut_edge_count"] = poly_stats.get("cut_edge_count")
    audit["invalid_count"] = poly_stats.get("invalid_count")
    if poly_stats.get("status") == "error":
        exc_reason = str(poly_stats.get("reason", "polygonize_failed"))
        if envelope_poly is not None and envelope_audit.get("status") == "verified":
            audit["status"] = "verified"
            audit["reason"] = "supported envelope accepted after polygonize failed"
            audit["polygonize_error"] = exc_reason
            audit["selected_candidate"] = {
                "id": "supported_envelope",
                "area_pt2": envelope_audit.get("area_pt2"),
                "area_fraction_of_viewport": envelope_audit.get(
                    "area_fraction_of_viewport"),
                "bounds": envelope_audit.get("bounds"),
                "score": envelope_audit.get("score"),
                "score_reasons": [envelope_audit.get("reason"),
                                  "polygonize_failed_fallback"],
            }
            return envelope_poly, audit
        audit["status"] = "error"
        audit["reason"] = exc_reason
        return None, audit

    audit["polygon_candidates"] = candidate_rows[:30]

    if not candidate_rows:
        if envelope_poly is not None and envelope_audit.get("status") == "verified":
            audit["status"] = "verified"
            audit["reason"] = "supported envelope accepted after polygonize produced no slab-like candidates"
            audit["selected_candidate"] = {
                "id": "supported_envelope",
                "area_pt2": envelope_audit.get("area_pt2"),
                "area_fraction_of_viewport": envelope_audit.get("area_fraction_of_viewport"),
                "bounds": envelope_audit.get("bounds"),
                "score": envelope_audit.get("score"),
                "score_reasons": [envelope_audit.get("reason")],
            }
            return envelope_poly, audit
        audit["reason"] = "polygonize_produced_no_slab_like_candidates"
        return None, audit
    best = candidate_rows[0]
    if best["score"] < 0.58:
        if envelope_poly is not None and envelope_audit.get("status") == "verified":
            audit["status"] = "verified"
            audit["reason"] = "supported envelope accepted after polygonize candidates were below gate"
            audit["best_polygonize_candidate"] = best
            audit["selected_candidate"] = {
                "id": "supported_envelope",
                "area_pt2": envelope_audit.get("area_pt2"),
                "area_fraction_of_viewport": envelope_audit.get("area_fraction_of_viewport"),
                "bounds": envelope_audit.get("bounds"),
                "score": envelope_audit.get("score"),
                "score_reasons": [envelope_audit.get("reason")],
            }
            return envelope_poly, audit
        audit["reason"] = "best_candidate_below_confidence_gate"
        audit["best_candidate"] = best
        return None, audit
    if len(candidate_rows) > 1 and candidate_rows[1]["score"] >= best["score"] - 0.04:
        audit["reason"] = "ambiguous_multiple_outline_candidates"
        audit["best_candidate"] = best
        audit["runner_up_candidate"] = candidate_rows[1]
        return None, audit

    selected = polygons_by_id.get(best["id"])
    if selected is None:
        audit["reason"] = "selected_candidate_geometry_missing"
        return None, audit
    selected = selected.buffer(0)
    if selected.is_empty or not selected.is_valid:
        audit["reason"] = "selected_candidate_invalid_after_clean"
        return None, audit
    audit["status"] = "verified"
    audit["reason"] = "closed irregular no-fill slab outline passed score gate"
    audit["selected_candidate"] = best
    audit["candidate_area_fraction_of_viewport"] = best["area_fraction_of_viewport"]
    audit["candidate_bounds"] = best["bounds"]
    return selected, audit


def assemble_no_fill_slab_boundary(paths: list, viewport: fitz.Rect,
                                   area_ref: float) -> tuple[Polygon | None, dict]:
    """Conservatively assemble a rectangular no-fill slab boundary.

    This is intentionally narrow: it only accepts a boundary when the vector
    evidence contains four strong, mostly continuous outer sides.  Irregular or
    incomplete outlines stay fail-closed for review instead of being guessed.
    """
    vw = float(viewport.width) or 1.0
    vh = float(viewport.height) or 1.0
    axis_tol = max(2.0, min(vw, vh) * 0.004)
    cluster_tol = max(6.0, min(vw, vh) * 0.01)
    gap_tol = max(24.0, min(vw, vh) * 0.025)
    min_seg = max(8.0, min(vw, vh) * 0.008)

    h_clusters: dict[int, list[tuple[float, float]]] = {}
    v_clusters: dict[int, list[tuple[float, float]]] = {}
    rejected = 0
    segment_count = 0

    def in_view(x: float, y: float) -> bool:
        return (
            viewport.x0 - cluster_tol <= x <= viewport.x1 + cluster_tol
            and viewport.y0 - cluster_tol <= y <= viewport.y1 + cluster_tol
        )

    for path in paths or []:
        if getattr(path, "outside_content", False):
            continue
        key = getattr(path, "key", None)
        if key is not None and getattr(key, "dashes", ""):
            rejected += 1
            continue
        if key is not None and getattr(key, "stroke", None) is None:
            rejected += 1
            continue
        for a, b in getattr(path, "segments", []) or []:
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            if not (in_view(ax, ay) and in_view(bx, by)):
                continue
            dx = abs(bx - ax)
            dy = abs(by - ay)
            if max(dx, dy) < min_seg:
                continue
            if dy <= axis_tol and dx >= min_seg:
                y = (ay + by) * 0.5
                bucket = int(round(y / cluster_tol))
                h_clusters.setdefault(bucket, []).append((ax, bx))
                segment_count += 1
            elif dx <= axis_tol and dy >= min_seg:
                x = (ax + bx) * 0.5
                bucket = int(round(x / cluster_tol))
                v_clusters.setdefault(bucket, []).append((ay, by))
                segment_count += 1

    def best_sides(clusters: dict[int, list[tuple[float, float]]],
                   span: float, low: bool) -> dict | None:
        candidates = []
        for bucket, intervals in clusters.items():
            merged = _merge_intervals(intervals, gap_tol)
            coverage = _interval_coverage(merged)
            ratio = coverage / max(span, 1.0)
            coord = bucket * cluster_tol
            candidates.append({
                "bucket": bucket,
                "coord": coord,
                "coverage": round(coverage, 3),
                "coverage_ratio": round(ratio, 4),
                "segments": len(intervals),
                "merged_intervals": [
                    [round(a, 3), round(b, 3)] for a, b in merged
                ],
            })
        candidates = [c for c in candidates if c["coverage_ratio"] >= 0.45]
        if not candidates:
            return None
        return sorted(candidates, key=lambda c: c["coord"], reverse=not low)[0]

    top = best_sides(h_clusters, vw, low=True)
    bottom = best_sides(h_clusters, vw, low=False)
    left = best_sides(v_clusters, vh, low=True)
    right = best_sides(v_clusters, vh, low=False)
    sides = {"top": top, "bottom": bottom, "left": left, "right": right}
    missing = [name for name, side in sides.items() if not side]

    audit = {
        "schema": "no_fill_slab_boundary_v1",
        "method": "axis_aligned_outer_side_coverage",
        "viewport_rect": _rect_payload(viewport),
        "segment_count_used": segment_count,
        "rejected_dashed_or_fill_only_paths": rejected,
        "sides": sides,
        "missing_boundary_sides": missing,
        "status": "unresolved",
        "reason": "",
    }
    if missing:
        audit["reason"] = "not enough vector coverage to verify all four outer slab sides"
        return None, audit

    x0 = float(left["coord"])
    x1 = float(right["coord"])
    y0 = float(top["coord"])
    y1 = float(bottom["coord"])
    if x1 <= x0 or y1 <= y0:
        audit["reason"] = "invalid side ordering"
        return None, audit
    poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    frac = poly.area / max(float(area_ref), 1.0)
    audit["candidate_bounds"] = [round(v, 3) for v in poly.bounds]
    audit["candidate_area_fraction_of_viewport"] = round(frac, 6)
    if frac < 0.10:
        audit["reason"] = "verified side rectangle is still below slab area threshold"
        return None, audit
    if any((side or {}).get("coverage_ratio", 0.0) < 0.55 for side in sides.values()):
        audit["reason"] = "one or more sides below conservative 55% coverage threshold"
        return None, audit
    audit["status"] = "verified"
    audit["reason"] = "four outer sides have enough non-dashed vector coverage"
    return poly, audit


def write_slab_boundary_failure_artifacts(page: fitz.Page, renderer,
                                          out_dir: Path, page_number: int,
                                          gross, viewport: fitz.Rect,
                                          audit: dict) -> None:
    """Write a visual overlay for a no-fill slab boundary decision."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"slab_boundary_assembly_p{page_number:02d}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        img = renderer.faded.copy()
        dr = ImageDraw.Draw(img)
        s = renderer.scale

        def rect_pts(rect: fitz.Rect):
            return [rect.x0 * s, rect.y0 * s, rect.x1 * s, rect.y1 * s]

        dr.rectangle(rect_pts(viewport), outline=(0, 180, 80), width=5)
        geoms = getattr(gross, "geoms", [gross]) if gross is not None else []
        for geom in geoms:
            if geom is None or getattr(geom, "is_empty", False):
                continue
            coords = [(x * s, y * s) for x, y in geom.exterior.coords]
            if len(coords) >= 3:
                dr.polygon(coords, outline=(220, 30, 30), fill=(255, 0, 0, 45))
                dr.line(coords, fill=(220, 30, 30), width=4)
        font = _font(18)
        if audit.get("status") == "verified":
            msg = "NO-FILL SLAB BOUNDARY VERIFIED"
            help_text = "green=plan viewport, red=accepted no-fill slab boundary"
        else:
            msg = (
                "SLAB FAIL-CLOSED: missing boundary "
                + ",".join(audit.get("missing_boundary_sides") or ["unknown"])
            )
            help_text = "green=plan viewport, red=tiny assembled slab rejected"
        dr.text((max(8, viewport.x0 * s), max(8, viewport.y0 * s - 28)),
                msg, fill=(180, 0, 0), font=font)
        dr.text((max(8, viewport.x0 * s), max(34, viewport.y0 * s + 4)),
                help_text, fill=(80, 80, 80), font=font)
        img.save(out_dir / f"slab_boundary_assembly_p{page_number:02d}.png")
    except Exception:
        return


def _no_fill_actionable_summary(audit: dict) -> dict:
    """Condense a no-fill resolver audit into a field engineer friendly cause."""
    reason = str(audit.get("reason") or "")
    envelope = audit.get("supported_envelope_candidate") or {}
    style_attempts = audit.get("style_family_attempts") or []
    best_style_fraction = 0.0
    best_style_id = None
    for attempt in style_attempts:
        best = attempt.get("best_candidate") or {}
        frac = float(best.get("area_fraction_of_viewport") or 0.0)
        if frac > best_style_fraction:
            best_style_fraction = frac
            best_style_id = attempt.get("style_key")

    primary = reason or "unresolved"
    if envelope.get("status") == "unresolved" and envelope.get("reason"):
        primary = str(envelope.get("reason"))
    if "polygonize_failed" in reason:
        primary = "polygonize_failed_topology"
    if audit.get("status") == "verified":
        primary = "verified_no_fill_boundary"

    suggestions = []
    if primary in {"no_supported_outer_envelope", "missing_perimeter_edges"}:
        suggestions.append("use another geometry authority page in the same level if available")
        suggestions.append("inspect missing perimeter edges before exporting slab")
    if primary == "polygonize_failed_topology":
        suggestions.append("line graph is too noisy/self-crossing; review rejected edge categories and viewport")
    if best_style_fraction < 0.08 and audit.get("status") != "verified":
        suggestions.append("no line-style family produced a large enough slab candidate")
    if not suggestions and audit.get("status") != "verified":
        suggestions.append("keep slab unexported and mark contract_unfulfilled")

    return {
        "primary_failure": primary,
        "status": audit.get("status") or "unresolved",
        "edge_candidate_count": audit.get("edge_candidate_count", 0),
        "gap_closure_count": audit.get("gap_closure_count", 0),
        "best_style_key": best_style_id,
        "best_style_area_fraction": round(best_style_fraction, 6),
        "supported_envelope_status": envelope.get("status"),
        "supported_envelope_reason": envelope.get("reason"),
        "export_policy": (
            "export_verified_no_fill_slab"
            if audit.get("status") == "verified"
            else "fail_closed_do_not_export_tiny_or_ambiguous_slab"
        ),
        "suggested_next_actions": suggestions,
    }


def write_no_fill_irregular_artifacts(page: fitz.Page, renderer,
                                      out_dir: Path, page_number: int,
                                      gross, viewport: fitz.Rect,
                                      audit: dict) -> None:
    """Write v2 no-fill resolver audit files and visual overlays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if "actionable_summary" not in audit:
        audit["actionable_summary"] = _no_fill_actionable_summary(audit)
    suffix = f"p{page_number:02d}"
    (out_dir / f"no_fill_slab_resolution_{suffix}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"no_fill_edge_candidates_{suffix}.json").write_text(
        json.dumps({
            "schema": "no_fill_edge_candidates_v1",
            "viewport_rect": audit.get("viewport_rect"),
            "edge_candidate_count": audit.get("edge_candidate_count", 0),
            "edge_candidates_sample": audit.get("edge_candidates_sample", []),
            "rejected_edge_counts": audit.get("rejected_edge_counts", {}),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"no_fill_gap_closures_{suffix}.json").write_text(
        json.dumps({
            "schema": "no_fill_gap_closures_v1",
            "viewport_rect": audit.get("viewport_rect"),
            "gap_closure_count": audit.get("gap_closure_count", 0),
            "gap_closures": audit.get("gap_closures", []),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / f"no_fill_polygon_candidates_{suffix}.json").write_text(
        json.dumps({
            "schema": "no_fill_polygon_candidates_v1",
            "viewport_rect": audit.get("viewport_rect"),
            "polygon_candidates": audit.get("polygon_candidates", []),
            "selected_candidate": audit.get("selected_candidate"),
            "status": audit.get("status"),
            "reason": audit.get("reason"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        img = renderer.faded.copy()
        dr = ImageDraw.Draw(img)
        s = renderer.scale
        font = _font(16)

        def xy(pt):
            return (float(pt[0]) * s, float(pt[1]) * s)

        def rect_pts(rect: fitz.Rect):
            return [rect.x0 * s, rect.y0 * s, rect.x1 * s, rect.y1 * s]

        # Edge candidate sample overlay.
        edge_img = img.copy()
        edge_dr = ImageDraw.Draw(edge_img)
        edge_dr.rectangle(rect_pts(viewport), outline=(0, 180, 80), width=5)
        for row in audit.get("edge_candidates_sample", [])[:450]:
            a = row.get("a")
            b = row.get("b")
            if a and b:
                edge_dr.line([xy(a), xy(b)], fill=(0, 160, 220), width=2)
        edge_dr.text((max(8, viewport.x0 * s), max(8, viewport.y0 * s - 24)),
                     f"NO-FILL EDGE CANDIDATES: {audit.get('edge_candidate_count', 0)}",
                     fill=(0, 120, 180), font=font)
        edge_img.save(out_dir / f"no_fill_edge_candidates_{suffix}.png")

        # Gap closure overlay.
        gap_img = img.copy()
        gap_dr = ImageDraw.Draw(gap_img)
        gap_dr.rectangle(rect_pts(viewport), outline=(0, 180, 80), width=5)
        for row in audit.get("gap_closures", [])[:300]:
            a = row.get("a")
            b = row.get("b")
            if a and b:
                gap_dr.line([xy(a), xy(b)], fill=(255, 0, 200), width=4)
        gap_dr.text((max(8, viewport.x0 * s), max(8, viewport.y0 * s - 24)),
                    f"GAP CLOSURES: {audit.get('gap_closure_count', 0)}",
                    fill=(180, 0, 160), font=font)
        gap_img.save(out_dir / f"no_fill_gap_closures_{suffix}.png")

        # Polygon decision overlay.
        poly_img = img.copy()
        poly_dr = ImageDraw.Draw(poly_img)
        poly_dr.rectangle(rect_pts(viewport), outline=(0, 180, 80), width=5)
        for row in audit.get("polygon_candidates", [])[:12]:
            b = row.get("bounds")
            if not b:
                continue
            outline = (255, 165, 0)
            width = 2
            if row.get("id") == (audit.get("selected_candidate") or {}).get("id"):
                outline = (0, 220, 80)
                width = 5
            poly_dr.rectangle(
                [b[0] * s, b[1] * s, b[2] * s, b[3] * s],
                outline=outline,
                width=width,
            )
            poly_dr.text((b[0] * s + 4, b[1] * s + 4),
                         f"{row.get('id')} s={row.get('score')}",
                         fill=outline, font=font)
        geoms = getattr(gross, "geoms", [gross]) if gross is not None else []
        for geom in geoms:
            if geom is None or getattr(geom, "is_empty", False):
                continue
            coords = [(x * s, y * s) for x, y in geom.exterior.coords]
            if len(coords) >= 3:
                poly_dr.line(coords, fill=(0, 160, 60), width=5)
        poly_dr.text((max(8, viewport.x0 * s), max(8, viewport.y0 * s - 24)),
                     f"NO-FILL POLYGONS: {audit.get('status')} / {audit.get('reason')}",
                     fill=(0, 120, 50) if audit.get("status") == "verified" else (180, 80, 0),
                     font=font)
        poly_img.save(out_dir / f"no_fill_polygon_candidates_{suffix}.png")

        # Graph overlay currently mirrors edge+closure evidence.
        graph_img = edge_img.copy()
        graph_dr = ImageDraw.Draw(graph_img)
        for row in audit.get("gap_closures", [])[:300]:
            a = row.get("a")
            b = row.get("b")
            if a and b:
                graph_dr.line([xy(a), xy(b)], fill=(255, 0, 200), width=4)
        graph_dr.text((max(8, viewport.x0 * s), max(32, viewport.y0 * s + 4)),
                      "cyan=PDF solid edge candidates, magenta=audited gap closures",
                      fill=(60, 60, 60), font=font)
        graph_img.save(out_dir / f"no_fill_graph_{suffix}.png")
        (out_dir / f"no_fill_graph_{suffix}.json").write_text(
            json.dumps({
                "schema": "no_fill_graph_v1",
                "edge_candidate_count": audit.get("edge_candidate_count", 0),
                "gap_closure_count": audit.get("gap_closure_count", 0),
                "dangle_count": audit.get("dangle_count"),
                "cut_edge_count": audit.get("cut_edge_count"),
                "status": audit.get("status"),
                "reason": audit.get("reason"),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        return
