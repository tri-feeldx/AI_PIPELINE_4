"""Close only millimetre-scale gaps at verified orthogonal wall junctions."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import fitz
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from src.slab_v2.models import WallFootprint

PT_TO_MM = 25.4 / 72.0


def _orientation(wall: WallFootprint) -> str | None:
    minx, miny, maxx, maxy = wall.polygon.bounds
    width, height = maxx-minx, maxy-miny
    if width >= 2.5*height:
        return "horizontal"
    if height >= 2.5*width:
        return "vertical"
    return None


def _opening_blocks(extension, openings) -> bool:
    if not openings:
        return False
    union = unary_union([item.polygon for item in openings
                         if getattr(item, "polygon", None) is not None])
    return not union.is_empty and extension.intersects(union)


def resolve_wall_junctions(page: fitz.Page, walls: list[WallFootprint],
                           expected: dict, scale: float, openings: list,
                           cfg, out_dir: Path) -> tuple[list[WallFootprint], dict]:
    """Snap endpoints to the outer face of perpendicular expected LW walls."""
    out_dir.mkdir(parents=True, exist_ok=True)
    to_mm = PT_TO_MM*float(scale or 0)
    if to_mm <= 0:
        return walls, {"status": "review", "junctions": [],
                       "warnings": ["Wall junction snap skipped: no scale."]}
    tolerance_pt = cfg.wall_junction_snap_max_mm/to_mm
    expected_symbols = {str(symbol).upper() for symbol, count in
                        (expected or {}).items() if int(count or 0) > 0}
    resolved = list(walls)
    rows = []

    for index, wall in enumerate(list(resolved)):
        symbol = wall.label.upper()
        orientation = _orientation(wall)
        if not symbol.startswith("LW") or symbol not in expected_symbols or not orientation:
            continue
        minx, miny, maxx, maxy = wall.polygon.bounds
        targets = {"left": minx, "right": maxx} if orientation == "horizontal" \
            else {"top": miny, "bottom": maxy}
        for endpoint, current in list(targets.items()):
            best = None
            for other in resolved:
                if other is wall or other.label.upper() not in expected_symbols:
                    continue
                other_orientation = _orientation(other)
                if other_orientation is None or other_orientation == orientation:
                    continue
                ominx, ominy, omaxx, omaxy = other.polygon.bounds
                if orientation == "horizontal":
                    if maxy < ominy-tolerance_pt or miny > omaxy+tolerance_pt:
                        continue
                    target = ominx if endpoint == "left" else omaxx
                else:
                    if maxx < ominx-tolerance_pt or minx > omaxx+tolerance_pt:
                        continue
                    target = ominy if endpoint == "top" else omaxy
                delta_pt = target-current
                delta_mm = abs(delta_pt)*to_mm
                if delta_mm > cfg.wall_junction_snap_max_mm:
                    continue
                if best is None or delta_mm < best[0]:
                    best = (delta_mm, target, other)
            if best is None or best[0] <= 0.05:
                continue
            delta_mm, target, other = best
            if orientation == "horizontal":
                new_bounds = (target if endpoint == "left" else minx, miny,
                              target if endpoint == "right" else maxx, maxy)
            else:
                new_bounds = (minx, target if endpoint == "top" else miny,
                              maxx, target if endpoint == "bottom" else maxy)
            extension = box(*new_bounds).difference(wall.polygon)
            if _opening_blocks(extension, openings):
                rows.append({"wall": wall.label, "endpoint": endpoint,
                             "target_wall": other.label, "status": "blocked",
                             "before_gap_mm": round(delta_mm, 3),
                             "reason": "extension intersects opening"})
                continue
            minx, miny, maxx, maxy = new_bounds
            new_poly = box(*new_bounds)
            new_length = ((maxx-minx) if orientation == "horizontal"
                          else (maxy-miny))*to_mm
            wall = replace(wall, polygon=new_poly, l_mm=round(new_length, 1),
                           source=f"{wall.source}+junction_snap")
            resolved[index] = wall
            rows.append({"wall": wall.label, "endpoint": endpoint,
                         "target_wall": other.label, "status": "snapped",
                         "before_gap_mm": round(delta_mm, 3),
                         "after_gap_mm": 0.0,
                         "evidence": ["expected_wall", "orthogonal_axis",
                                      "outer_face_within_tolerance"]})

    unresolved = [row for row in rows if row["status"] != "snapped"]
    status = "verified" if not unresolved else "review"
    report = {"status": status, "page": page.number+1,
              "max_snap_mm": cfg.wall_junction_snap_max_mm,
              "verified_gap_mm": cfg.wall_junction_verified_gap_mm,
              "junctions": rows, "warnings": [
                  "One or more small wall junctions could not be snapped."
              ] if unresolved else []}
    (out_dir / "wall_junction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"wall_junction_candidates_p{page.number+1:02d}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"wall_junction_resolved_p{page.number+1:02d}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _render(page, walls, rows,
            out_dir / f"wall_junction_candidates_p{page.number+1:02d}.png",
            resolved=False)
    _render(page, resolved, rows,
            out_dir / f"wall_junction_resolved_p{page.number+1:02d}.png",
            resolved=True)
    return resolved, report


def _render(page: fitz.Page, walls: list, rows: list, path: Path,
            resolved: bool) -> None:
    from PIL import Image, ImageDraw
    factor = 120/72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    changed = {row["wall"] for row in rows if row["status"] == "snapped"}
    for wall in walls:
        color = ((40, 190, 80, 150) if resolved and wall.label in changed
                 else (230, 140, 20, 120))
        for geom in getattr(wall.polygon, "geoms", [wall.polygon]):
            if not isinstance(geom, Polygon):
                continue
            points = [(x*factor, y*factor) for x, y in geom.exterior.coords]
            draw.polygon(points, fill=color, outline=color[:3]+(255,))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    ImageDraw.Draw(image).text((10, 10),
        "green=snapped wall, orange=original/review", fill="black")
    image.save(path)
