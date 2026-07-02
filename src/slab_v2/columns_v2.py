"""
Column detection v2 — text-anchor-then-shape.

Pass 1: find text labels matching column symbols on the page, search for
rectangular shapes near each label, assign symbol precisely.
Pass 2: shape-first fallback (columns.py logic) for unlabeled columns,
skipping polygons already claimed by Pass 1.

Solves the C1/C2/C3-all-600x600 problem: when multiple schedule types share
dimensions, the text label disambiguates which symbol belongs where.
"""

from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from pathlib import Path

import fitz
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ColumnFootprint, ColumnType
from src.slab_v2.columns import (
    _path_polygon, _rect_sides_pt, _size_match, PT_TO_MM,
    _collect_steel_exclusion_zones, _in_steel_exclusion, _normalize_label,
)


def _recover_segmented_column_box(anchor: Point, paths: list, classes: list,
                                  to_mm: float, radius: float = 75.0):
    """Reconstruct an under/over-only dashed square from its dash segments.

    Some PDFs encode every dash as a separate solid line, so no closed path
    exists for the normal rectangle detector.  We cluster nearby heavy black
    axis-aligned segments and use their exact endpoint extent as the box.
    """
    segments = []
    for path in paths:
        # Under-only columns are intentionally drawn just outside the gross
        # slab/content rectangle, so text-anchored recovery must inspect them.
        if not (0 <= path.style_id < len(classes)):
            continue
        key = classes[path.style_id].key
        stroke = key.stroke
        if not stroke or max(stroke) > 0.15 or key.width < 0.9:
            continue
        for a, b in path.segments:
            line = LineString([a, b])
            if line.distance(anchor) <= radius:
                dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
                if dx < 0.1 and dy < 0.1:
                    continue
                if min(dx, dy) > 0.25 * max(dx, dy):
                    continue
                segments.append((line, a, b, dx >= dy))
    if len(segments) < 6:
        return None

    seen = set()
    candidates = []
    for seed in range(len(segments)):
        if seed in seen:
            continue
        group = {seed}
        queue = [seed]
        seen.add(seed)
        while queue:
            i = queue.pop()
            for j in range(len(segments)):
                if j in seen:
                    continue
                if segments[i][0].distance(segments[j][0]) <= 2.6:
                    seen.add(j)
                    group.add(j)
                    queue.append(j)
        if len(group) < 6:
            continue
        pts = [pt for i in group for pt in (segments[i][1], segments[i][2])]
        minx = min(p[0] for p in pts); maxx = max(p[0] for p in pts)
        miny = min(p[1] for p in pts); maxy = max(p[1] for p in pts)
        w_pt, h_pt = maxx - minx, maxy - miny
        if min(w_pt, h_pt) <= 0:
            continue
        w_mm, h_mm = w_pt * to_mm, h_pt * to_mm
        if not (150 <= min(w_mm, h_mm) <= 1500
                and max(w_mm, h_mm) <= 1500
                and 0.55 <= w_pt / h_pt <= 1.8):
            continue
        horizontals = sum(segments[i][3] for i in group)
        verticals = len(group) - horizontals
        if horizontals < 2 or verticals < 2:
            continue
        poly = box(minx, miny, maxx, maxy)
        candidates.append((poly.distance(anchor), poly, w_mm, h_mm))
    if not candidates:
        return None
    _dist, poly, w_mm, h_mm = min(candidates, key=lambda item: item[0])
    return poly, max(w_mm, h_mm), min(w_mm, h_mm)


def extract_columns_v2(
    page: fitz.Page,
    paths: list,
    slab_union,
    scale: float,
    column_types: dict[str, ColumnType],
    cfg: SlabV2Config,
    elements: list | None = None,
    columns_per_floor_census: dict[str, int] | None = None,
    classes: list | None = None,
    audit_out_dir: Path | None = None,
) -> tuple[list[ColumnFootprint], list[str], dict]:
    """Text-anchor-then-shape column detection with shape-first fallback."""
    warnings: list[str] = []
    if not scale:
        return [], ["column detection skipped: no scale"], {
            "status": "review", "candidates": [], "assignments": []}
    steel_symbols = {
        _normalize_label(sym) for sym, ct in (column_types or {}).items()
        if str(getattr(ct, "material", "") or "").upper() == "STEEL"
    }
    steel_symbols.discard("")
    steel_skipped = sorted(
        sym for sym, ct in (column_types or {}).items()
        if str(getattr(ct, "material", "") or "").upper() == "STEEL"
    )
    if steel_skipped:
        warnings.append(
            "steel column types skipped in RC-only phase: "
            + ", ".join(steel_skipped)
        )
    column_types = {
        sym: ct for sym, ct in (column_types or {}).items()
        if str(getattr(ct, "material", "") or "").upper() != "STEEL"
    }
    to_mm = PT_TO_MM * scale

    # Grid bubbles describe the structural drawing envelope more faithfully
    # than the slab fill: over/under-only perimeter columns may sit outside it.
    try:
        from src.slab_v2.wall_profile_resolver import _grid_anchors
        grid_anchors = _grid_anchors(page)
    except Exception:
        grid_anchors = {}
    if len(grid_anchors) >= 3:
        xs = [point[0] for point in grid_anchors.values()]
        ys = [point[1] for point in grid_anchors.values()]
        search_envelope = box(min(xs), min(ys), max(xs), max(ys)).buffer(60)
    elif slab_union is not None:
        search_envelope = slab_union.buffer(80)
    else:
        search_envelope = box(*page.rect)

    openings = unary_union([e.polygon for e in elements]) \
        if elements else None
    steel_exclusion_radius = max(cfg.steel_exclusion_radius_pt, 40.0)
    steel_exclusion_zones = _collect_steel_exclusion_zones(
        page, steel_symbols, steel_exclusion_radius, slab_union)
    if steel_exclusion_zones:
        warnings.append(
            f"steel exclusion zones active: {len(steel_exclusion_zones)} label(s)")

    # ── build all rectangular candidates from vector paths ───────────────
    all_cands = []  # (poly, w_mm, d_mm, is_dashed, outside_content, id, style_role)
    for p in paths:
        is_dashed = bool(
            classes and 0 <= p.style_id < len(classes)
            and classes[p.style_id].key.dashes)
        style_role = (classes[p.style_id].role
                      if classes and 0 <= p.style_id < len(classes)
                      else "UNKNOWN")
        if style_role == "ANNOTATION":
            continue
        poly = _path_polygon(p)
        if poly is None:
            continue
        sides = _rect_sides_pt(poly)
        if sides is None:
            continue
        w_mm, d_mm = sides[0] * to_mm, sides[1] * to_mm
        if not (100.0 <= d_mm and w_mm <= cfg.column_max_side_mm):
            continue
        if openings is not None:
            overlap = poly.intersection(openings).area
            if overlap / max(poly.area, 1e-9) > 0.80:
                continue
        if p.outside_content and not poly.intersects(search_envelope):
            continue
        if slab_union is not None and not p.outside_content:
            dist_mm = poly.distance(slab_union) * to_mm
            if not poly.intersects(slab_union) and dist_mm > w_mm:
                continue
        all_cands.append((poly, w_mm, d_mm, is_dashed,
                          bool(p.outside_content), f"colcand_{len(all_cands)+1:03d}",
                          style_role))

    if not all_cands:
        return [], warnings, {"status": "review", "candidates": [],
                              "assignments": [], "grid_anchors": grid_anchors}

    # build STRtree for spatial queries
    cand_geoms = [c[0] for c in all_cands]
    cand_tree = STRtree(cand_geoms)

    # ── PASS 1: text-anchor-then-shape ───────────────────────────────────
    symbols_norm = {_normalize_label(s): s for s in column_types}
    text_anchors = []  # (symbol, cx, cy)
    if symbols_norm:
        all_words = page.get_text("words")
        for w in all_words:
            txt = _normalize_label(w[4])
            if txt in symbols_norm:
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                if not search_envelope.contains(Point(cx, cy)):
                    continue
                text_anchors.append((symbols_norm[txt], cx, cy))

        # merge split labels: "C" + "9" → "C9" (PDF splits multi-line text)
        _SPLIT_MERGE_DIST = 20.0  # pt
        letter_words = []
        digit_words = []
        for w in all_words:
            raw = w[4].strip()
            norm = _normalize_label(raw)
            if not norm:
                continue
            if norm.isalpha() and len(norm) <= 3:
                letter_words.append((norm, (w[0]+w[2])/2, (w[1]+w[3])/2))
            elif norm.isdigit() and len(norm) <= 2:
                digit_words.append((norm, (w[0]+w[2])/2, (w[1]+w[3])/2))
        for ltxt, lx, ly in letter_words:
            for dtxt, dx, dy in digit_words:
                if ((lx - dx)**2 + (ly - dy)**2)**0.5 > _SPLIT_MERGE_DIST:
                    continue
                combined = ltxt + dtxt
                if combined in symbols_norm:
                    mx, my = (lx + dx) / 2, (ly + dy) / 2
                    if not search_envelope.contains(Point(mx, my)):
                        continue
                    text_anchors.append((symbols_norm[combined], mx, my))

        # dedupe: same symbol within 30pt → keep first occurrence only
        _seen: dict[str, tuple[float, float]] = {}
        _deduped: list[tuple[str, float, float]] = []
        for sym, cx, cy in text_anchors:
            if sym in _seen:
                ox, oy = _seen[sym]
                if ((cx - ox)**2 + (cy - oy)**2)**0.5 < 30:
                    continue
            _seen[sym] = (cx, cy)
            _deduped.append((sym, cx, cy))
        text_anchors = _deduped

    claimed: set[int] = set()  # indices into all_cands
    pass1_columns: list[ColumnFootprint] = []
    search_r = cfg.column_text_search_radius_pt
    edge_rows = []
    for anchor_index, (sym, cx, cy) in enumerate(text_anchors):
        ct = column_types.get(sym)
        if ct is None:
            continue
        has_size = bool(ct.width_mm and ct.depth_mm)
        radius = search_r if has_size else max(search_r, 60.0)
        anchor = Point(cx, cy)
        for raw_index in cand_tree.query(anchor.buffer(radius)):
            candidate_index = int(raw_index)
            poly, w, d, _dashed, outside, candidate_id, _role = all_cands[candidate_index]
            if has_size and not _size_match(w, d, ct, cfg.column_size_tol_mm):
                continue
            if not has_size and (min(w, d) < 150 or
                                 max(w, d) > cfg.column_max_side_mm):
                continue
            if _in_steel_exclusion(poly, steel_exclusion_zones):
                continue
            distance = poly.distance(anchor)
            size_error = 0.0 if not has_size else min(
                abs(w-ct.width_mm)+abs(d-ct.depth_mm),
                abs(w-ct.depth_mm)+abs(d-ct.width_mm))
            cost = distance + 0.02*size_error + (2.0 if outside else 0.0)
            edge_rows.append((cost, anchor_index, candidate_index, distance,
                              radius, sym, candidate_id))

    # Solve one-to-one globally, so an early ambiguous label cannot steal the
    # only valid rectangle from a later exact label.
    assignments = []
    if edge_rows:
        anchor_ids = sorted({row[1] for row in edge_rows})
        candidate_ids = sorted({row[2] for row in edge_rows})
        ai = {value: i for i, value in enumerate(anchor_ids)}
        ci = {value: i for i, value in enumerate(candidate_ids)}
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
            matrix = np.full((len(anchor_ids), len(candidate_ids)), 1e6)
            edge_lookup = {}
            for row in edge_rows:
                key = (ai[row[1]], ci[row[2]])
                if row[0] < matrix[key]:
                    matrix[key] = row[0]
                    edge_lookup[key] = row
            rr, cc = linear_sum_assignment(matrix)
            assignments = [edge_lookup[(r, c)] for r, c in zip(rr, cc)
                           if matrix[r, c] < 1e5 and (r, c) in edge_lookup]
        except Exception:
            used_anchors, used_candidates = set(), set()
            for row in sorted(edge_rows):
                if row[1] in used_anchors or row[2] in used_candidates:
                    continue
                assignments.append(row)
                used_anchors.add(row[1]); used_candidates.add(row[2])

    assigned_anchor_ids = set()
    assignment_report = []
    for _cost, anchor_index, candidate_index, distance, radius, sym, candidate_id in assignments:
        poly, w, d, _dashed, outside, _cid, _role = all_cands[candidate_index]
        claimed.add(candidate_index)
        assigned_anchor_ids.add(anchor_index)
        confidence = max(0.75, min(0.99, 1.0-0.25*distance/max(radius, 1)))
        pass1_columns.append(ColumnFootprint(
            symbol=sym, polygon=poly, w_mm=round(w, 0), d_mm=round(d, 0),
            labeled=True, candidate_id=candidate_id,
            source="global_text_assignment", confidence=round(confidence, 3)))
        assignment_report.append({"symbol": sym, "candidate_id": candidate_id,
                                  "distance_pt": round(distance, 3),
                                  "outside_content": outside,
                                  "confidence": round(confidence, 3),
                                  "source": "global_text_assignment"})
        ct = column_types.get(sym)
        if ct and not (ct.width_mm and ct.depth_mm):
            warnings.append(
                f"{sym}: dimensions absent; measured {w:.0f}x{d:.0f}mm "
                "from globally assigned text-anchored rectangle")

    for anchor_index, (sym, cx, cy) in enumerate(text_anchors):
        if anchor_index in assigned_anchor_ids or not classes:
            continue
        ct = column_types.get(sym)
        if ct is None:
            continue
        recovered = _recover_segmented_column_box(
            Point(cx, cy), paths, classes, to_mm, radius=75.0)
        if recovered is not None:
            poly, w, d = recovered
            if (ct.width_mm and ct.depth_mm
                    and not _size_match(w, d, ct, cfg.column_size_tol_mm)):
                continue
            # Segmented boxes often coexist with a closed path on another
            # layer. Do not introduce a second column at the same location.
            if any(poly.intersection(existing.polygon).area /
                   max(poly.union(existing.polygon).area, 1e-9) > 0.5
                   for existing in pass1_columns):
                continue
            candidate_id = f"segmented_{sym}_{anchor_index+1}"
            pass1_columns.append(ColumnFootprint(
                symbol=sym, polygon=poly, w_mm=round(w, 0), d_mm=round(d, 0),
                labeled=True, candidate_id=candidate_id,
                source="segmented_anchor_recovery", confidence=0.90))
            assignment_report.append({"symbol": sym,
                                      "candidate_id": candidate_id,
                                      "confidence": 0.90,
                                      "source": "segmented_anchor_recovery"})

    # dedupe pass1 (same polygon claimed by multiple nearby text anchors)
    if len(pass1_columns) > 1:
        p1_geoms = [c.polygon for c in pass1_columns]
        p1_tree = STRtree(p1_geoms)
        p1_keep = set()
        for i, col in enumerate(pass1_columns):
            dup = False
            for j in p1_tree.query(col.polygon):
                j = int(j)
                if j == i or j not in p1_keep:
                    continue
                inter = col.polygon.intersection(p1_geoms[j]).area
                union = col.polygon.area + p1_geoms[j].area - inter
                if union > 0 and inter / union > 0.5:
                    dup = True
                    break
            if not dup:
                p1_keep.add(i)
        pass1_columns = [pass1_columns[i] for i in sorted(p1_keep)]

    # ── PASS 2: shape-first fallback for unclaimed candidates ────────────
    unclaimed = [(i, poly, w, d, dashed, style_role)
                 for i, (poly, w, d, dashed, outside, _candidate_id, style_role)
                 in enumerate(all_cands)
                 if i not in claimed and not dashed and not outside]

    matched = []  # (poly, w_mm, d_mm, [symbols], style_role)
    if column_types:
        for _i, poly, w, d, _dashed, srole in unclaimed:
            if _in_steel_exclusion(poly, steel_exclusion_zones):
                continue
            syms = [t.symbol for t in column_types.values()
                    if _size_match(w, d, t, cfg.column_size_tol_mm)]
            if syms:
                matched.append((poly, w, d, syms, srole))
    else:
        # no census: anonymous repeated-size types
        groups = defaultdict(list)
        for _i, poly, w, d, _dashed, srole in unclaimed:
            key = (int(round(w / 25.0) * 25), int(round(d / 25.0) * 25))
            groups[key].append((poly, w, d, srole))
        for (kw, kd), members in sorted(groups.items()):
            if len(members) >= cfg.column_min_repeat:
                sym = f"COL{kw}x{kd}"
                matched.extend((poly, w, d, [sym], srole)
                               for poly, w, d, srole in members)
        if matched:
            warnings.append(
                "no column schedule — using "
                f"{len({m[3][0] for m in matched})} anonymous type(s)")

    # dedupe pass2
    if matched:
        matched.sort(key=lambda m: m[0].area)
        m_geoms = [m[0] for m in matched]
        m_tree = STRtree(m_geoms)
        kept_idx = []
        kept_set: set[int] = set()
        for i, (poly, _w, _d, _s, _sr) in enumerate(matched):
            dup = False
            for j in m_tree.query(poly):
                j = int(j)
                if j == i or j not in kept_set:
                    continue
                inter = poly.intersection(m_geoms[j]).area
                union = poly.area + m_geoms[j].area - inter
                if union > 0 and inter / union > 0.5:
                    dup = True
                    break
            if not dup:
                # also check against pass1 claimed polygons
                if pass1_columns:
                    for p1col in pass1_columns:
                        inter = poly.intersection(p1col.polygon).area
                        union = poly.area + p1col.polygon.area - inter
                        if union > 0 and inter / union > 0.5:
                            dup = True
                            break
                if not dup:
                    kept_set.add(i)
                    kept_idx.append(i)

        ambiguous = 0
        for i in kept_idx:
            poly, w, d, syms, srole = matched[i]
            symbol = syms[0] if len(syms) == 1 else "C?"
            if symbol == "C?":
                ambiguous += 1
            conf = 0.75 if srole == "COLUMN" else 0.60
            pass1_columns.append(ColumnFootprint(
                symbol=symbol, polygon=poly,
                w_mm=round(w, 0), d_mm=round(d, 0), labeled=False,
                source="shape_fallback", confidence=conf))
        if ambiguous:
            warnings.append(
                f"{ambiguous} unlabeled column(s) match multiple schedule "
                f"sizes — exported as 'C?'")

    columns = pass1_columns

    # ── census count cap: Gemini count is the hard limit per symbol ──────
    if columns_per_floor_census:
        by_sym: dict[str, list[tuple[int, ColumnFootprint]]] = defaultdict(list)
        for i, c in enumerate(columns):
            if c.symbol == "C?":
                continue
            by_sym[c.symbol].append((i, c))
        drop = set()
        for sym, group in by_sym.items():
            cap = columns_per_floor_census.get(sym)
            if cap is None or len(group) <= cap:
                continue
            group.sort(key=lambda t: (-int(t[1].labeled), t[0]))
            for idx, _ in group[cap:]:
                drop.add(idx)
        if drop:
            columns = [c for i, c in enumerate(columns) if i not in drop]

    # ── census cross-check ───────────────────────────────────────────────
    if columns_per_floor_census:
        detected: dict[str, int] = defaultdict(int)
        for c in columns:
            detected[c.symbol] += 1
        for sym, expected in columns_per_floor_census.items():
            got = detected.get(sym, 0)
            if got != expected:
                warnings.append(
                    f"census expects {expected}× {sym}, detected {got}")
        for sym, got in detected.items():
            if sym not in columns_per_floor_census and sym != "C?":
                warnings.append(
                    f"detected {got}× {sym} not in census for this floor")

    labeled_count = sum(1 for c in columns if c.labeled)
    if text_anchors and labeled_count:
        warnings.insert(0,
            f"text-anchor pass: {labeled_count} column(s) identified by "
            f"label, {len(columns) - labeled_count} by shape fallback")

    candidate_report = [{
        "id": candidate_id,
        "bounds": [round(value, 3) for value in poly.bounds],
        "exterior": [[round(float(x), 3), round(float(y), 3)]
                     for x, y in poly.exterior.coords],
        "w_mm": round(w, 1), "d_mm": round(d, 1),
        "dashed": dashed, "outside_content": outside,
        "claimed": index in claimed,
    } for index, (poly, w, d, dashed, outside, candidate_id, _srole)
       in enumerate(all_cands)]
    detected_report = defaultdict(int)
    for column in columns:
        detected_report[column.symbol] += 1
    expected_report = {
        str(symbol).rstrip("*"): int(count or 0)
        for symbol, count in (columns_per_floor_census or {}).items()
        if str(symbol).upper().startswith("C")
        and not str(symbol).upper().startswith(("CH", "CHS"))}
    missing = {symbol: count-detected_report.get(symbol, 0)
               for symbol, count in expected_report.items()
               if detected_report.get(symbol, 0) < count}
    extra = {symbol: count for symbol, count in detected_report.items()
             if symbol not in expected_report and symbol != "C?"}
    ambiguous = detected_report.get("C?", 0)
    status = "verified" if expected_report and not missing and not extra \
        and not ambiguous else "review"
    audit = {"status": status, "grid_anchors": grid_anchors,
             "candidates": candidate_report, "assignments": assignment_report,
             "expected": expected_report, "detected": dict(detected_report),
             "missing": missing, "extra": extra,
             "ambiguous_count": ambiguous}
    if audit_out_dir:
        audit_out_dir = Path(audit_out_dir)
        audit_out_dir.mkdir(parents=True, exist_ok=True)
        pnum = page.number+1
        (audit_out_dir / f"column_candidates_p{pnum:02d}.json").write_text(
            json.dumps(candidate_report, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (audit_out_dir / f"column_assignment_p{pnum:02d}.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        (audit_out_dir / "column_expected_vs_detected.json").write_text(
            json.dumps({key: audit[key] for key in
                       ("status", "expected", "detected", "missing", "extra",
                        "ambiguous_count")}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        _render_column_audit(page, all_cands, columns,
                             audit_out_dir / f"column_candidates_p{pnum:02d}.png",
                             audit_out_dir / f"column_assignment_p{pnum:02d}.png")
    return columns, warnings, audit


def _render_column_audit(page: fitz.Page, candidates: list, columns: list,
                         candidates_path: Path, assignment_path: Path) -> None:
    from PIL import Image, ImageDraw
    factor = 120/72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
    base = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    candidate_image = base.copy()
    draw = ImageDraw.Draw(candidate_image)
    for _poly, _w, _d, _dashed, outside, candidate_id, *_rest in candidates:
        minx, miny, maxx, maxy = _poly.bounds
        color = (240, 150, 20) if outside else (40, 150, 220)
        draw.rectangle((minx*factor, miny*factor, maxx*factor, maxy*factor),
                       outline=color, width=3)
        draw.text((minx*factor, miny*factor), candidate_id, fill=color)
    candidate_image.save(candidates_path)
    assigned = base.copy()
    draw = ImageDraw.Draw(assigned)
    for column in columns:
        minx, miny, maxx, maxy = column.polygon.bounds
        color = (30, 180, 70) if column.symbol != "C?" else (230, 130, 20)
        draw.rectangle((minx*factor, miny*factor, maxx*factor, maxy*factor),
                       outline=color, width=5)
        draw.text((minx*factor, max(0, miny*factor-14)), column.symbol,
                  fill=color)
    assigned.save(assignment_path)
