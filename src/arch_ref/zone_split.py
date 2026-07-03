"""Split-deck slab partition + cross-drawing axis mapping (Bước 3).

split_polygon_by_zones partitions a slab polygon between deck zones by
nearest RL label (Voronoi over the label points, cells unioned per zone,
clipped to the slab).  One zone is the degenerate flat-floor case and
passes through untouched — no special-casing per building type.

fit_axis_map fits x' = a*x + b on shared grid-axis positions of two
sheets (least squares), which absorbs scale differences AND mirroring
(2381 STR sheets are mirrored against the ARCH set).
"""
from __future__ import annotations

import shapely
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union, voronoi_diagram


def split_polygon_by_zones(slab_poly, zones: dict) -> list:
    """[(rl_m, polygon)] — every zone's share of the slab.

    zones: {rl_m: [(x, y), ...]} label positions in the SLAB's own
    coordinate space (map them first when they come from another sheet).
    """
    zones = {rl: pts for rl, pts in zones.items() if pts}
    if not zones:
        return []
    if len(zones) == 1:
        return [(next(iter(zones)), slab_poly)]

    all_pts = [(rl, Point(p)) for rl, pts in zones.items() for p in pts]
    envelope = shapely.box(*slab_poly.buffer(10).bounds)
    cells = voronoi_diagram(MultiPoint([p for _, p in all_pts]),
                            envelope=envelope)

    by_zone: dict[float, list] = {rl: [] for rl in zones}
    for cell in cells.geoms:
        # a Voronoi cell belongs to the label it contains
        owner = min(all_pts, key=lambda ap: cell.distance(ap[1])
                    if not cell.contains(ap[1]) else -1.0)
        by_zone[owner[0]].append(cell)

    out = []
    for rl, cell_list in by_zone.items():
        if not cell_list:
            continue
        region = unary_union(cell_list).intersection(slab_poly)
        region = shapely.make_valid(region)
        if not region.is_empty and region.area > 0:
            out.append((rl, region))
    return out


def fit_axis_map(pairs: list):
    """Least-squares linear map x' = a*x + b from (x, x') samples."""
    n = len(pairs)
    if n == 0:
        raise ValueError("no axis pairs")
    if n == 1:
        x0, y0 = pairs[0]
        return lambda x: x - x0 + y0
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    denom = n * sxx - sx * sx
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return lambda x: a * x + b
