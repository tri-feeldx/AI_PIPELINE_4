"""
Stage B — planarization and face enumeration.

Segments from the selected style classes are noded (GEOS snap-rounding at
cfg.snap_grid_pt) and polygonized into atomic closed faces. The true slab
boundary, whatever line style it uses, is guaranteed to appear as one face
or a union of adjacent faces — with exact vector coordinates.

Gap handling moves dangling endpoints ONTO existing nodes (never invented
midpoints), bounded by cfg.gap_ladder_pt.
"""

from __future__ import annotations

import shapely
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import polygonize_full, unary_union
from shapely.strtree import STRtree

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import VectorPath, Face, FaceGraph


def _boundary_edges_to_inject(
    segments: list,
    content_rect,
    proximity_pt: float = 3.0,
    min_touches: int = 2,
) -> list:
    """Detect which content_rect edges have segments terminating nearby.

    Multi-sheet structural plans often have slab boundaries that extend
    beyond the drawing area.  Injecting the touched content_rect edges
    closes the boundary so polygonization can produce large slab faces.
    """
    if content_rect is None:
        return []
    x0, y0 = content_rect.x0, content_rect.y0
    x1, y1 = content_rect.x1, content_rect.y1

    edges = {
        "top":    ((x0, y0), (x1, y0)),
        "right":  ((x1, y0), (x1, y1)),
        "bottom": ((x1, y1), (x0, y1)),
        "left":   ((x0, y1), (x0, y0)),
    }

    touches = {side: 0 for side in edges}
    for (a, b) in segments:
        for pt in (a, b):
            px, py = pt
            if abs(py - y0) <= proximity_pt and x0 - proximity_pt <= px <= x1 + proximity_pt:
                touches["top"] += 1
            if abs(px - x1) <= proximity_pt and y0 - proximity_pt <= py <= y1 + proximity_pt:
                touches["right"] += 1
            if abs(py - y1) <= proximity_pt and x0 - proximity_pt <= px <= x1 + proximity_pt:
                touches["bottom"] += 1
            if abs(px - x0) <= proximity_pt and y0 - proximity_pt <= py <= y1 + proximity_pt:
                touches["left"] += 1

    return [edges[side] for side, count in touches.items()
            if count >= min_touches]


def _fill_boundary_segments(fill_poly, simplify_tol: float = 0.5) -> list:
    """Extract simplified boundary segments from a fill polygon's exterior ring."""
    if not fill_poly.is_valid or fill_poly.is_empty:
        return []
    simplified = fill_poly.exterior.simplify(simplify_tol, preserve_topology=True)
    coords = list(simplified.coords)
    return [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)
            if coords[i] != coords[i + 1]]


def _collect_segments(paths: list[VectorPath], style_ids: set[int],
                      dash_bridge_tol_pt: float = 0.0) -> list:
    """Deduplicated segments of the requested classes (content area only),
    plus collinear bridges for dashed lines exported as separate dashes."""
    seen = set()
    by_class: dict[int, list] = {}
    for p in paths:
        if p.style_id not in style_ids or p.outside_content:
            continue
        # Fill-only paths: use simplified fill boundary instead of micro-segments
        if (not p.has_stroke and p.fill_polygon is not None
                and p.fill_polygon.is_valid):
            source_segs = _fill_boundary_segments(p.fill_polygon)
        else:
            source_segs = p.segments
        for (a, b) in source_segs:
            # direction-normalized quantized key (0.001 pt) for exact dedup
            ka = (round(a[0], 3), round(a[1], 3))
            kb = (round(b[0], 3), round(b[1], 3))
            if ka == kb:
                continue
            key = (ka, kb) if ka <= kb else (kb, ka)
            if key in seen:
                continue
            seen.add(key)
            by_class.setdefault(p.style_id, []).append((a, b))

    segments = []
    for sid, segs in by_class.items():
        collinear = _bridge_collinear(segs)
        segments.extend(segs)
        segments.extend(collinear)
        if dash_bridge_tol_pt > 0:
            # corner/junction gaps are not collinear; close them by joining
            # mutually-nearest free endpoints of the same class
            segments.extend(
                _bridge_endpoints(segs + collinear, dash_bridge_tol_pt))
    return segments


def _bridge_endpoints(segs: list, tol_pt: float,
                      max_dev_deg: float = 30.0) -> list:
    """Connectors between mutually-nearest FREE endpoints within tol_pt.

    A free endpoint occurs in exactly one segment (a dash end).  Pairing is
    mutual-nearest and one-shot, and a bridge is only allowed when it
    CONTINUES at least one of the two dashes (deviation <= max_dev_deg from
    that dash's outward direction).  Slab edges are pairs of parallel dashed
    lines ~1.4pt apart — without the direction rule the nearer endpoint is
    on the neighbouring line and bridging rungs a ladder across the pair.
    Both bridge vertices are existing PDF coordinates.
    """
    import math
    from collections import Counter

    def key(p):
        return (round(p[0], 3), round(p[1], 3))

    cnt = Counter()
    existing = set()
    outward: dict[tuple, tuple] = {}
    for a, b in segs:
        ka, kb = key(a), key(b)
        cnt[ka] += 1
        cnt[kb] += 1
        existing.add((ka, kb) if ka <= kb else (kb, ka))
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L > 1e-9:
            outward[ka] = (-dx / L, -dy / L)   # pointing away from segment
            outward[kb] = (dx / L, dy / L)
    free = [p for p, c in cnt.items() if c == 1]
    if len(free) < 2:
        return []

    min_cos = math.cos(math.radians(max_dev_deg))

    def continues(p, q):
        """Bridge p->q must roughly follow the outward direction of the
        dash at p or at q (corner gaps align with one side only)."""
        bx, by = q[0] - p[0], q[1] - p[1]
        L = math.hypot(bx, by)
        if L < 1e-9:
            return False
        for end, sign in ((p, 1.0), (q, -1.0)):
            d = outward.get(end)
            if d and (bx * d[0] + by * d[1]) * sign / L >= min_cos:
                return True
        return False

    cell = max(tol_pt, 1e-6)
    grid: dict[tuple, list] = {}
    for p in free:
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(p)

    def nearest(p):
        cx, cy = int(p[0] // cell), int(p[1] // cell)
        best, best_d = None, tol_pt
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for q in grid.get((gx, gy), ()):
                    if q == p:
                        continue
                    if ((p, q) if p <= q else (q, p)) in existing:
                        continue      # already directly connected
                    if not continues(p, q):
                        continue      # would rung across a parallel pair
                    d = math.hypot(p[0] - q[0], p[1] - q[1])
                    if 1e-9 < d <= best_d:
                        best, best_d = q, d
        return best

    bridges, used = [], set()
    for p in free:
        if p in used:
            continue
        q = nearest(p)
        if q is None or q in used:
            continue
        if nearest(q) == p:
            bridges.append((p, q))
            used.add(p)
            used.add(q)
    return bridges


def _bridge_collinear(segs: list, max_gap_pt: float = 30.0) -> list:
    """Connectors across gaps between collinear segments of one style class.

    CAD exporters often emit dashed lines (match lines, boundary markers) as
    one segment per dash — geometrically open. Segments lying on the same
    infinite line (direction and offset quantized) are sorted along it and
    consecutive gaps up to gap_tol are bridged. Bridges lie exactly ON the
    original line, so no coordinate is distorted.
    """
    import math
    lens = sorted(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs)
    if not lens:
        return []
    median = lens[len(lens) // 2]
    # dash gaps scale with dash length; solid long lines only get tiny heals
    gap_tol = min(max(6.0, 3.0 * median), max_gap_pt) if median < 20.0 else 2.0

    groups: dict[tuple, list] = {}
    for (a, b) in segs:
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ux, uy = dx / L, dy / L
        if ux < 0 or (ux == 0 and uy < 0):       # canonical direction
            ux, uy = -ux, -uy
        ang = round(math.degrees(math.atan2(uy, ux)) * 2) / 2  # 0.5° grid
        off = round((a[0] * (-uy) + a[1] * ux) * 2) / 2        # 0.5 pt grid
        t1 = a[0] * ux + a[1] * uy
        t2 = b[0] * ux + b[1] * uy
        groups.setdefault((ang, off), []).append(
            (min(t1, t2), max(t1, t2), a if t1 <= t2 else b,
             b if t1 <= t2 else a))

    bridges = []
    for items in groups.values():
        if len(items) < 2:
            continue
        items.sort()
        cur_end_t, cur_end_pt = items[0][1], items[0][3]
        for (t1, t2, p1, p2) in items[1:]:
            gap = t1 - cur_end_t
            if 1e-6 < gap <= gap_tol:
                bridges.append((cur_end_pt, p1))
            if t2 > cur_end_t:
                cur_end_t, cur_end_pt = t2, p2
    return bridges


def _polygonize(segments: list, snap_grid: float):
    mls = MultiLineString(segments)
    snapped = shapely.set_precision(mls, grid_size=snap_grid)
    try:
        noded = shapely.node(snapped)
    except Exception:
        # GEOS noding can fail on near-degenerate geometry from fill boundaries;
        # retry with a coarser grid before giving up
        snapped = shapely.set_precision(mls, grid_size=snap_grid * 10)
        try:
            noded = shapely.node(snapped)
        except Exception:
            return [], [], []
    faces, dangles, cuts, invalids = polygonize_full([noded])
    return list(faces.geoms), list(dangles.geoms), list(cuts.geoms)


def _snap_dangles_to_nodes(segments: list, dangles: list, eps: float) -> list:
    """Move dangle endpoints onto the nearest existing endpoint within eps.

    Only endpoint coordinates of existing segments are snap targets, so no
    coordinate is invented.
    """
    endpoints = set()
    for (a, b) in segments:
        endpoints.add((round(a[0], 3), round(a[1], 3)))
        endpoints.add((round(b[0], 3), round(b[1], 3)))
    pts = [Point(p) for p in endpoints]
    if not pts:
        return segments
    tree = STRtree(pts)

    dangle_ends = set()
    for d in dangles:
        coords = list(d.coords)
        for p in (coords[0], coords[-1]):
            dangle_ends.add((round(p[0], 3), round(p[1], 3)))

    remap = {}
    for de in dangle_ends:
        p = Point(de)
        idx = tree.query(p.buffer(eps))
        best, best_d = None, eps
        for i in idx:
            cand = pts[i]
            cd = p.distance(cand)
            if 1e-9 < cd <= best_d:
                best, best_d = (cand.x, cand.y), cd
        if best is not None:
            remap[de] = best

    if not remap:
        return segments

    out = []
    for (a, b) in segments:
        ka = (round(a[0], 3), round(a[1], 3))
        kb = (round(b[0], 3), round(b[1], 3))
        na = remap.get(ka, a)
        nb = remap.get(kb, b)
        if na != nb:
            out.append((na, nb))
    return out


def _assign_hierarchy(faces: list[Face]) -> None:
    """parent_id = smallest containing face; depth = chain length."""
    if not faces:
        return
    polys = [f.polygon for f in faces]
    tree = STRtree(polys)
    order = sorted(range(len(faces)), key=lambda i: faces[i].area_pt2)
    for i in order:
        f = faces[i]
        rp = f.polygon.representative_point()
        candidates = tree.query(f.polygon)
        best, best_area = None, None
        for j in candidates:
            if j == i:
                continue
            g = faces[j]
            if g.area_pt2 <= f.area_pt2:
                continue
            if g.polygon.contains(rp):
                if best_area is None or g.area_pt2 < best_area:
                    best, best_area = g.id, g.area_pt2
        f.parent_id = best
    by_id = {f.id: f for f in faces}
    for f in faces:
        d, p = 0, f.parent_id
        while p is not None and d < 50:
            d += 1
            p = by_id[p].parent_id
        f.depth = d


def _bordering_styles(face, seg_index: STRtree, seg_geoms: list,
                      seg_styles: list, tol: float = 0.1) -> frozenset:
    band = face.polygon.exterior.buffer(tol)
    hits = seg_index.query(band)
    out = set()
    for i in hits:
        if seg_geoms[i].intersects(band):
            out.add(seg_styles[i])
    return frozenset(out)


def build_face_graph(
    paths: list[VectorPath],
    style_ids: set[int],
    cfg: SlabV2Config,
    content_area_pt2: float,
    content_rect=None,
) -> FaceGraph:
    """Node + polygonize the segments of the given style classes."""
    segments = _collect_segments(
        paths, style_ids,
        dash_bridge_tol_pt=getattr(cfg, "dash_bridge_tol_pt", 0.0))

    if content_rect is not None:
        boundary_segs = _boundary_edges_to_inject(
            segments, content_rect,
            proximity_pt=cfg.content_boundary_proximity_pt,
            min_touches=cfg.content_boundary_min_touches)
        segments.extend(boundary_segs)

    if len(segments) > cfg.max_polygonize_segments:
        segments.sort(
            key=lambda s: -((s[0][0] - s[1][0]) ** 2 + (s[0][1] - s[1][1]) ** 2))
        segments = segments[: cfg.max_polygonize_segments]

    if not segments:
        return FaceGraph(faces=[], source_style_ids=tuple(sorted(style_ids)))

    min_face_area = cfg.min_face_area_frac * content_area_pt2

    polys, dangles, cuts = _polygonize(segments, cfg.snap_grid_pt)
    snap_used = 0.0

    def significant(ps):
        return sum(1 for p in ps if p.area >= min_face_area)

    if significant(polys) < cfg.min_faces and dangles:
        for eps in cfg.gap_ladder_pt:
            healed = _snap_dangles_to_nodes(segments, dangles, eps)
            polys2, dangles2, cuts2 = _polygonize(healed, cfg.snap_grid_pt)
            if significant(polys2) > significant(polys):
                polys, dangles, cuts = polys2, dangles2, cuts2
                snap_used = eps
                break

    faces = []
    for poly in polys:
        if poly.is_empty or poly.area <= 0:
            continue
        faces.append(Face(
            id=len(faces),
            polygon=poly,
            area_pt2=poly.area,
            label_anchor=tuple(poly.representative_point().coords[0]),
        ))

    # inject fill polygons of the selected classes as extra candidate faces
    for p in paths:
        if (p.style_id in style_ids and not p.outside_content
                and p.fill_polygon is not None
                and p.fill_polygon.area >= min_face_area):
            fp = p.fill_polygon
            if any(abs(f.area_pt2 - fp.area) / max(fp.area, 1.0) < 0.01
                   and f.polygon.intersection(fp).area > 0.95 * fp.area
                   for f in faces):
                continue
            faces.append(Face(
                id=len(faces), polygon=fp, area_pt2=fp.area,
                label_anchor=tuple(fp.representative_point().coords[0]),
                source="fill",
            ))

    # bordering style classes per face
    seg_geoms, seg_styles = [], []
    for p in paths:
        if p.style_id in style_ids and not p.outside_content:
            for (a, b) in p.segments:
                if a != b:
                    seg_geoms.append(LineString([a, b]))
                    seg_styles.append(p.style_id)
    if seg_geoms and faces:
        tree = STRtree(seg_geoms)
        for f in faces:
            f.style_ids = _bordering_styles(f, tree, seg_geoms, seg_styles)

    _assign_hierarchy(faces)

    return FaceGraph(
        faces=faces,
        dangles=dangles,
        cut_edges=cuts,
        snap_used_pt=snap_used,
        source_style_ids=tuple(sorted(style_ids)),
        n_segments_in=len(segments),
    )


def augment_until_closed(
    paths: list[VectorPath],
    base_ids: set[int],
    classes: list,
    cfg: SlabV2Config,
    content_area_pt2: float,
    content_rect=None,
) -> tuple[FaceGraph, set[int], list[int]]:
    """Greedy class augmentation when the elected classes don't close a
    slab-sized face (e.g. the plan is cut by a match line drawn in another
    style). A candidate class is kept only if it GROWS the largest face —
    grid/hatch classes that fragment the region are rejected.
    """
    target = cfg.min_area_frac * content_area_pt2

    def max_face(fg: FaceGraph) -> float:
        return max((f.area_pt2 for f in fg.faces), default=0.0)

    best_fg = build_face_graph(paths, base_ids, cfg, content_area_pt2,
                               content_rect=content_rect)
    best_ids = set(base_ids)
    added: list[int] = []
    if max_face(best_fg) >= target:
        return best_fg, best_ids, added

    candidates = [
        c for c in sorted(classes, key=lambda c: -c.total_length_pt)
        if c.id not in base_ids and c.role != "FRAME"
        and not (c.prefiltered and c.role == "HATCH")
    ]
    for c in candidates[:12]:
        trial = best_ids | {c.id}
        fg2 = build_face_graph(paths, trial, cfg, content_area_pt2,
                               content_rect=content_rect)
        m2, m1 = max_face(fg2), max_face(best_fg)
        if m2 >= target or m2 > 1.5 * m1:
            best_fg, best_ids = fg2, trial
            added.append(c.id)
            if max_face(best_fg) >= target:
                break
    return best_fg, best_ids, added


def assemble_slab_polygon(faces: list[Face], face_ids: list[int],
                          void_ids: list[int],
                          min_component_frac: float = 0.02,
                          sliver_heal_pt: float = 0.5):
    """Tolerant assembly of an AI face selection into slab geometry.

    - union the selected faces; small disconnected strays (mislabeled ids)
      are dropped, large components become separate slab parts
    - buffer/unbuffer heals micro-slivers while preserving genuine holes
    - voids are subtracted only where explicitly marked

    Returns (geometry | None, error str | None).
    """
    by_id = {f.id: f for f in faces}
    try:
        union = unary_union([by_id[i].polygon for i in face_ids])
        union = shapely.make_valid(union)
        comps = list(getattr(union, "geoms", [union]))
        comps = [c for c in comps if not c.is_empty and c.area > 0]
        if not comps:
            return None, "selected faces produced empty geometry"
        max_area = max(c.area for c in comps)
        kept = [c for c in comps if c.area >= min_component_frac * max_area]

        gross = unary_union(kept)
        if sliver_heal_pt > 0:
            gross = gross.buffer(sliver_heal_pt).buffer(-sliver_heal_pt)
        gross = shapely.make_valid(gross)

        if void_ids:
            voids = unary_union([by_id[i].polygon for i in void_ids])
            gross = shapely.make_valid(gross.difference(voids))
        if gross.is_empty:
            return None, "voids removed the entire slab area"
        return gross, None
    except Exception as e:                  # noqa: BLE001
        return None, str(e)
