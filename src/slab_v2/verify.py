"""
Stage D — verification against the drawing's own dimension annotations.

Dimension texts are associated to vector segments by projection (text center
must project onto the segment, perpendicular distance bounded, parallel
within tolerance). Implied scales (value_mm / segment_length_mm) then vote:
the largest consensus group defines the dimension-implied scale. Numbers in
schedule tables don't converge to one scale and are thereby ignored.

Verification is SOFT: it can fail a wrong selection (scale disagreement),
boost confidence (edge/extent matches), or be inconclusive when the page has
no usable dimensions — inconclusive does not block the result, because the
hard guarantee (vector-exact coordinates) comes from Stages A-B.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import fitz
from shapely.geometry import LineString, Point

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import (StyleClass, DimensionAnnotation,
                                VerificationReport)

# standalone mm values: "3600", "12 500", "12,500" — 3 to 5 significant digits
_DIM_RE = re.compile(r"^\s*(\d{1,3}(?:[ ,]\d{3})+|\d{3,5})\s*$")

PT_TO_MM = 25.4 / 72.0
MIN_CONSENSUS = 3       # dims that must agree before the vote counts


def _parse_value_mm(text: str) -> float | None:
    m = _DIM_RE.match(text)
    if not m:
        return None
    return float(m.group(1).replace(",", "").replace(" ", ""))


def _all_segments(page: fitz.Page, min_len: float = 15.0) -> list:
    out = []
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":
                a = (item[1].x, item[1].y)
                b = (item[2].x, item[2].y)
                if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_len:
                    out.append((a, b))
    return out


def parse_dimensions(
    page: fitz.Page,
    classes: list[StyleClass],
    cfg: SlabV2Config,
    content_rect: fitz.Rect,
) -> list[DimensionAnnotation]:
    """Find dimension texts page-wide; associate each with its segment.

    Dimension chains sit just OUTSIDE the drawing content rect (along the
    sheet border), so the scan covers the whole page and only excludes the
    legend area and the title-block strips. Schedule-table numbers that
    slip through are neutralized later by the distinct-value consensus.

    Association: segment parallel to the text within tolerance, text center
    projects onto the segment span, perpendicular distance minimal and below
    cfg.dim_assoc_radius_pt.
    """
    from src.pdf_processor import FFL_PATTERN, SCALE_PATTERN
    from src.vision_refiner import find_legend_rect

    legend = find_legend_rect(page)
    pw, ph = page.rect.width, page.rect.height

    def excluded(cx: float, cy: float) -> bool:
        if legend.contains(fitz.Point(cx, cy)):
            return True
        return cx > pw * 0.78 and cy > ph * 0.88   # title block corner

    dims: list[DimensionAnnotation] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            rot = math.degrees(math.atan2(-dy, dx)) % 180.0
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                val = _parse_value_mm(txt)
                if val is None:
                    continue
                if FFL_PATTERN.search(txt) or SCALE_PATTERN.search(txt):
                    continue
                bbox = span.get("bbox")
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                if excluded(cx, cy):
                    continue
                dims.append(DimensionAnnotation(
                    text=txt, value_mm=val, bbox=tuple(bbox),
                    rotation_deg=rot))

    if not dims:
        return dims

    segments = _all_segments(page)

    for d in dims:
        cx = (d.bbox[0] + d.bbox[2]) / 2
        cy = (d.bbox[1] + d.bbox[3]) / 2
        best, best_perp = None, cfg.dim_assoc_radius_pt
        for (a, b) in segments:
            vx, vy = b[0] - a[0], b[1] - a[1]
            seg_len = math.hypot(vx, vy)
            ang = math.degrees(math.atan2(-vy, vx)) % 180.0
            diff = min(abs(ang - d.rotation_deg),
                       180.0 - abs(ang - d.rotation_deg))
            if diff > cfg.dim_parallel_tol_deg:
                continue
            # projection of text center onto the segment
            t = ((cx - a[0]) * vx + (cy - a[1]) * vy) / (seg_len * seg_len)
            if not (-0.05 <= t <= 1.05):
                continue
            px, py = a[0] + t * vx, a[1] + t * vy
            perp = math.hypot(cx - px, cy - py)
            if perp < best_perp:
                best, best_perp = (a, b), perp
        if best is not None:
            d.dim_line = best
            d.measured_pt = math.hypot(best[1][0] - best[0][0],
                                       best[1][1] - best[0][1])
    return dims


def _consensus_scale(dims: list[DimensionAnnotation],
                     cfg: SlabV2Config) -> tuple:
    """Largest group of dims agreeing on one implied scale.

    Returns (modal_scale | None, precise_scale, consensus_dims,
    agreement_fraction). precise_scale is the continuous median of the
    consensus group's implied scales — 0.0 unless the group is strong
    (>= precise_scale_min_dims DISTINCT values, relative spread within
    precise_scale_max_spread). The bucket modal alone snaps to multiples
    of 5 and silently absorbs non-integer viewport scales (A1 plotted at
    A3 = 1:141.42), a systematic ~1% size error the customer notices.
    """
    pairs = []
    for d in dims:
        if d.measured_pt and d.measured_pt > 5.0:
            implied = d.value_mm / (d.measured_pt * PT_TO_MM)
            pairs.append((d, implied))
    if not pairs:
        return None, 0.0, [], 0.0

    buckets = Counter()
    for _d, s in pairs:
        buckets[int(round(s / 5.0) * 5)] += 1
    # a valid consensus needs >= MIN_CONSENSUS dims with DISTINCT values —
    # repeated values in schedule tables (uniform cell widths) otherwise
    # fabricate a fake scale
    for modal, count in buckets.most_common():
        if count < MIN_CONSENSUS:
            break
        members = [(d, s) for d, s in pairs
                   if abs(s - modal) / max(modal, 1) <= 0.05]
        if len({d.value_mm for d, _s in members}) < MIN_CONSENSUS:
            continue
        # precise scale only from LONG dim lines: short ones (a 350 detail
        # dim is ~10pt of ink) carry tick noise and 5mm label rounding that
        # would inject the very error we are trying to remove
        strong = [(d, s) for d, s in members
                  if d.measured_pt >= cfg.precise_scale_min_len_pt]
        precise = 0.0
        if len({d.value_mm for d, _s in strong}) \
                >= cfg.precise_scale_min_dims:
            scales = sorted(s for _d, s in strong)
            spread = (scales[-1] - scales[0]) / max(scales[0], 1e-9)
            if spread <= cfg.precise_scale_max_spread:
                # length-weighted median: order by scale, take the scale at
                # half the cumulative dim-line length
                ws = sorted((s, d.measured_pt) for d, s in strong)
                half = sum(w for _s, w in ws) / 2.0
                acc = 0.0
                for s, w in ws:
                    acc += w
                    if acc >= half:
                        precise = s
                        break
        dims_only = [d for d, _s in members]
        return modal, precise, dims_only, len(members) / len(pairs)
    return None, 0.0, [], 0.0


def verify_selection(
    slabs: list[dict],
    dims: list[DimensionAnnotation],
    scale: int | None,
    content_rect: fitz.Rect,
    cfg: SlabV2Config,
) -> VerificationReport:
    report = VerificationReport()
    associated = [d for d in dims if d.measured_pt and d.measured_pt > 5.0]
    report.n_dims_associated = len(associated)

    modal, precise, consensus, agreement = _consensus_scale(associated, cfg)
    if modal is not None and not (
            cfg.scale_sanity_min <= modal <= cfg.scale_sanity_max):
        # a coincidental consensus on a page with few real dims can imply
        # absurd scales (seen: 1:5840 -> 45,000 m2 "slab"); never let it
        # override the text scale
        report.failures.append(
            f"measured scale 1:{modal} outside the sane range "
            f"1:{cfg.scale_sanity_min}..1:{cfg.scale_sanity_max} — ignored")
        modal, precise, consensus, agreement = None, 0.0, [], 0.0
    report.scale_consistency = agreement
    report.scale_precise = precise

    # ── 1. scale check (only meaningful with a consensus) ─────────────────
    # scale_used reports the MEASURED scale when a consensus exists — the
    # caller treats it as authoritative over the text-detected scale.
    scale_ok = True
    if modal is not None:
        if scale is None:
            scale = modal
        elif abs(modal - scale) / scale > 0.05:
            scale_ok = False
            report.failures.append(
                f"{len(consensus)} dimensions agree on scale 1:{modal}, "
                f"which contradicts the page text scale 1:{scale}")
            scale = modal          # measured evidence wins downstream
        report.scale_used = modal
    else:
        report.scale_used = scale or 0

    # edge/extent comparisons use the most accurate scale available
    if precise:
        scale = precise

    # ── 2. edge matching (confidence booster, consensus dims only) ────────
    # Slab edges rarely equal a single grid dimension — they span CHAINS
    # (9020 + 8925 + ...), and the polygon breaks straight runs at every
    # wall node. So: simplify collinear vertices for measurement only, and
    # match against single values plus consecutive chain sums.
    if scale and consensus:
        candidates = _dim_value_candidates(consensus)
        for s in slabs:
            geom = s["polygon_pdf"]
            geoms = getattr(geom, "geoms", [geom])
            for g in geoms:
                coords = list(g.simplify(0.5).exterior.coords)
                for i in range(len(coords) - 1):
                    a, b = coords[i], coords[i + 1]
                    edge_pt = math.hypot(b[0] - a[0], b[1] - a[1])
                    if edge_pt < 5.0:
                        continue
                    edge_mm = edge_pt * PT_TO_MM * scale
                    edge_ang = math.degrees(math.atan2(
                        -(b[1] - a[1]), b[0] - a[0])) % 180.0
                    best = None
                    for value, n_dims, ang in candidates:
                        diff = min(abs(ang - edge_ang),
                                   180.0 - abs(ang - edge_ang))
                        if diff > 2 * cfg.dim_parallel_tol_deg:
                            continue
                        rel = abs(edge_mm - value) / value
                        if rel <= cfg.dim_rel_tol and (
                                best is None or rel < best[2]):
                            best = (value, n_dims, rel)
                    if best:
                        report.edge_matches.append({
                            "edge": (a, b),
                            "edge_mm": round(edge_mm, 0),
                            "dim_value_mm": best[0],
                            "n_dims_summed": best[1],
                            "rel_err": round(best[2], 4),
                        })

    # ── 3. extent check (bbox vs the largest dims on the page) ────────────
    if scale and slabs and consensus:
        geom = slabs[0]["polygon_pdf"]
        minx, miny, maxx, maxy = geom.bounds
        big_dims = sorted({v for v, _n, _a in
                           _dim_value_candidates(consensus)},
                          reverse=True)[:20]
        for target, label in (((maxx - minx) * PT_TO_MM * scale, "width"),
                              ((maxy - miny) * PT_TO_MM * scale, "height")):
            hit = next((v for v in big_dims
                        if abs(v - target) / target <= 0.025), None)
            report.extent_check[label] = {
                "bbox_mm": round(target), "matched_dim": hit}

    # ── verdict ───────────────────────────────────────────────────────────
    if modal is None:
        report.failures.append(
            "verification inconclusive: no dimension consensus on this page "
            f"({len(associated)} associated, {MIN_CONSENSUS} needed)")
        report.passed = True          # soft check — kernel guarantee stands
    else:
        report.passed = scale_ok
    return report


def _dim_value_candidates(consensus: list[DimensionAnnotation]) -> list:
    """Lengths a slab edge may legitimately measure: every consensus dim
    value, plus sums of CONSECUTIVE dims of one collinear chain (grid
    dimension rows: 9020 + 8925 + ... = overall edge). Returns
    [(value_mm, n_dims_summed, angle_deg), ...]."""
    out = []
    chains: dict[tuple, list] = {}
    for d in consensus:
        if not d.dim_line:
            continue
        (x1, y1), (x2, y2) = d.dim_line
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy)
        if ln < 1.0:
            continue
        ang = math.degrees(math.atan2(-dy, dx)) % 180.0
        out.append((d.value_mm, 1, ang))
        ux, uy = dx / ln, dy / ln
        if uy < 0 or (uy == 0 and ux < 0):       # canonical direction
            ux, uy = -ux, -uy
        nx, ny = -uy, ux
        offset = x1 * nx + y1 * ny
        key = (int(round(ang / 2.0)) % 90, int(round(offset / 8.0)))
        t0 = x1 * ux + y1 * uy
        t1 = x2 * ux + y2 * uy
        chains.setdefault(key, []).append(
            (min(t0, t1), max(t0, t1), d.value_mm, ang))
    for members in chains.values():
        if len(members) < 2:
            continue
        members.sort()
        for i in range(len(members)):
            total, prev_end = members[i][2], members[i][1]
            for j in range(i + 1, len(members)):
                gap = members[j][0] - prev_end
                if not -5.0 <= gap <= 25.0:      # chain must stay contiguous
                    break
                total += members[j][2]
                prev_end = members[j][1]
                out.append((total, j - i + 1, members[i][3]))
    return out


def _near_parallel(a, b, d: DimensionAnnotation, cfg: SlabV2Config) -> bool:
    """Dimension text lies near edge (a,b) and is parallel to it."""
    ang_e = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0])) % 180.0
    diff = min(abs(ang_e - d.rotation_deg),
               180.0 - abs(ang_e - d.rotation_deg))
    if diff > 2 * cfg.dim_parallel_tol_deg:
        return False
    cx = (d.bbox[0] + d.bbox[2]) / 2
    cy = (d.bbox[1] + d.bbox[3]) / 2
    return LineString([a, b]).distance(Point(cx, cy)) <= 60.0
