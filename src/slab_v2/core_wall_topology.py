"""Assign core-wall identities from topology and target-page vectors."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import fitz
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from src.slab_v2.models import WallFootprint

PT_TO_MM = 25.4 / 72.0


@dataclass
class CoreWallRunCandidate:
    candidate_id: str
    source_index: int
    current_label: str
    orientation: str
    bbox: tuple
    thickness_pt: float
    length_pt: float
    relative_position: str = "unknown"
    nearby_text: list[str] = field(default_factory=list)
    vector_coverage: list[float] = field(default_factory=list)
    topology_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class CoreWallIdentityCorrection:
    candidate_id: str
    from_label: str
    to_label: str
    reason: str
    before_bbox: tuple
    after_bbox: tuple


@dataclass
class CoreWallTopologyAssignment:
    page: int
    status: str
    assignments: dict[str, str] = field(default_factory=dict)
    corrections: list[CoreWallIdentityCorrection] = field(default_factory=list)
    rejected_extension_bboxes: list[tuple] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _orientation(wall: WallFootprint) -> str:
    minx, miny, maxx, maxy = wall.polygon.bounds
    return "horizontal" if maxx-minx >= maxy-miny else "vertical"


def _interval_coverage(intervals, start, end, gap=1.0):
    rows = sorted((max(start, min(a, b)), min(end, max(a, b)))
                  for a, b in intervals if min(end, max(a, b)) >
                  max(start, min(a, b)))
    if not rows or end <= start:
        return 0.0
    total = 0.0
    lo, hi = rows[0]
    for nxt_lo, nxt_hi in rows[1:]:
        if nxt_lo <= hi + gap:
            hi = max(hi, nxt_hi)
        else:
            total += hi-lo
            lo, hi = nxt_lo, nxt_hi
    return min(1.0, (total + hi-lo)/(end-start))


def _rail_coverages(paths, classes, bbox, axis_tol):
    minx, miny, maxx, maxy = bbox
    rails = [[], []]
    for path in paths:
        if path.outside_content:
            continue
        if classes and 0 <= path.style_id < len(classes):
            if classes[path.style_id].key.dashes:
                continue
        for start, end in path.segments:
            dx, dy = end[0]-start[0], end[1]-start[1]
            if abs(dx) < 3.0 or abs(dy) > max(0.35, 0.01*abs(dx)):
                continue
            y = (start[1]+end[1])/2
            for index, rail_y in enumerate((miny, maxy)):
                if abs(y-rail_y) <= axis_tol:
                    rails[index].append((start[0], end[0]))
    return [_interval_coverage(rows, minx, maxx, gap=max(1.0, axis_tol))
            for rows in rails]


def _nearby_text(page, bounds, radius=12.0):
    minx, miny, maxx, maxy = bounds
    region = fitz.Rect(minx-radius, miny-radius, maxx+radius, maxy+radius)
    rows = []
    for word in page.get_text("words"):
        if fitz.Rect(word[:4]).intersects(region):
            value = str(word[4]).strip()
            if value and value not in rows:
                rows.append(value)
    return rows[:12]


def _candidate_registry(page, walls):
    candidates = []
    for index, wall in enumerate(walls):
        if not wall.label.upper().startswith("LW"):
            continue
        minx, miny, maxx, maxy = wall.polygon.bounds
        orientation = _orientation(wall)
        candidates.append(CoreWallRunCandidate(
            candidate_id=f"core_run_{index:02d}", source_index=index,
            current_label=wall.label.upper(), orientation=orientation,
            bbox=(minx, miny, maxx, maxy),
            thickness_pt=min(maxx-minx, maxy-miny),
            length_pt=max(maxx-minx, maxy-miny),
            nearby_text=_nearby_text(page, wall.polygon.bounds)))
    return candidates


def _unique_vertical(candidates, label):
    rows = [row for row in candidates
            if row.current_label == label and row.orientation == "vertical"]
    return rows[0] if len(rows) == 1 else None


def _score_horizontal(candidate, target_bbox):
    minx, miny, maxx, maxy = candidate.bbox
    tx0, ty0, tx1, ty1 = target_bbox
    span = max(tx1-tx0, 1e-6)
    target_thickness = max(ty1-ty0, 1e-6)
    overlap = max(0.0, min(maxx, tx1)-max(minx, tx0))/span
    y_error = abs((miny+maxy)/2-(ty0+ty1)/2)/target_thickness
    thickness_error = abs((maxy-miny)-target_thickness)/target_thickness
    return 3.0*overlap-0.9*y_error-0.4*thickness_error


def _bounds_close(left, right, tolerance=0.05):
    return all(abs(a-b) <= tolerance for a, b in zip(left, right))


def resolve_core_wall_topology(page: fitz.Page, paths: list, classes: list,
                               walls: list[WallFootprint], scale: float,
                               openings: list, cfg, out_dir: Path):
    """Globally assign LW1/LW3 using core topology, then complete LW1.

    Text labels remain evidence only.  A destructive identity correction is
    accepted only when the target page contains both real vector rails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_registry(page, walls)
    assignment = CoreWallTopologyAssignment(
        page=page.number+1, status="review")
    verticals = {name: _unique_vertical(candidates, name)
                 for name in ("LW2", "LW4", "LW5", "LW7")}
    if not all(verticals.values()) or not scale:
        assignment.warnings.append(
            "Core topology requires unique target-page LW2/LW4/LW5/LW7 vectors.")
        return _finish(page, walls, walls, candidates, assignment, out_dir)

    lw2, lw4, lw5, lw7 = (verticals[name]
                          for name in ("LW2", "LW4", "LW5", "LW7"))
    horizontal = [row for row in candidates if row.orientation == "horizontal"]
    if len(horizontal) < 2:
        assignment.warnings.append("Insufficient horizontal core-wall runs.")
        return _finish(page, walls, walls, candidates, assignment, out_dir)

    thicknesses = sorted(row.thickness_pt for row in horizontal
                         if row.thickness_pt > 0)
    thickness = thicknesses[len(thicknesses)//2]
    outer_minx = min(lw2.bbox[0], lw2.bbox[2])
    outer_maxx = max(lw7.bbox[0], lw7.bbox[2])
    terminal_y = max(lw2.bbox[3], lw7.bbox[3])
    bottom_bbox = (outer_minx, terminal_y,
                   outer_maxx, terminal_y+thickness)
    inner_minx = lw4.bbox[2]
    inner_maxx = lw5.bbox[0]
    inner_rows = [row for row in horizontal
                  if row.bbox[1] < terminal_y-0.25*thickness]
    if inner_maxx <= inner_minx or not inner_rows:
        assignment.warnings.append("Degenerate inner LW3 topology span.")
        return _finish(page, walls, walls, candidates, assignment, out_dir)

    # Select wall identities globally.  The score favours geometry and
    # junction position; the current PDF label is only a small tie-breaker.
    inner_target_y = max(row.bbox[1] for row in inner_rows
                         if row.bbox[1] < terminal_y-0.25*thickness)
    inner_bbox = (inner_minx, inner_target_y,
                  inner_maxx, inner_target_y+thickness)
    inner = max(inner_rows, key=lambda row:
                _score_horizontal(row, inner_bbox) +
                (0.15 if row.current_label == "LW3" else 0.0))
    bottom_source = max(horizontal, key=lambda row:
                        _score_horizontal(row, bottom_bbox) +
                        (0.15 if row.current_label == "LW1" else 0.0))

    axis_tol = max(0.8, 0.12*thickness)
    bottom_coverage = _rail_coverages(
        paths, classes, bottom_bbox, axis_tol)
    inner_coverage = _rail_coverages(
        paths, classes, inner.bbox, axis_tol)
    min_coverage = float(getattr(cfg, "lw1_min_vector_coverage", 0.35))
    if min(bottom_coverage) < min_coverage:
        assignment.warnings.append(
            f"LW1 retained: target bottom rail coverage {bottom_coverage} "
            f"below {min_coverage:.2f}.")
        for row in candidates:
            row.vector_coverage = (_rail_coverages(
                paths, classes, row.bbox, axis_tol)
                if row.orientation == "horizontal" else [])
        return _finish(page, walls, walls, candidates, assignment, out_dir)
    inner_identity_correct = (
        inner.current_label == "LW3"
        and abs(inner.bbox[0]-inner_minx) <= axis_tol
        and abs(inner.bbox[2]-inner_maxx) <= axis_tol)
    if not inner_identity_correct and min(inner_coverage) < min_coverage:
        assignment.warnings.append(
            f"LW3 retained: inner rail coverage {inner_coverage} "
            f"below {min_coverage:.2f}.")
        return _finish(page, walls, walls, candidates, assignment, out_dir)

    bottom_polygon = box(*bottom_bbox)
    inner_polygon = (walls[inner.source_index].polygon
                     if inner_identity_correct else
                     box(inner_minx, inner.bbox[1],
                         inner_maxx, inner.bbox[3]))
    opening_union = unary_union([
        item.polygon for item in openings
        if getattr(item, "polygon", None) is not None]) if openings else None
    if opening_union is not None and not opening_union.is_empty:
        extension = bottom_polygon.difference(
            walls[bottom_source.source_index].polygon)
        if extension.intersects(opening_union):
            assignment.warnings.append(
                "LW1 topology correction intersects an opening.")
            return _finish(page, walls, walls, candidates, assignment, out_dir)

    to_mm = PT_TO_MM*float(scale)
    old_inner = walls[inner.source_index]
    old_bottom = walls[bottom_source.source_index]
    bottom_identity_correct = (
        bottom_source.current_label == "LW1"
        and _bounds_close(bottom_source.bbox, bottom_bbox,
                          tolerance=axis_tol))
    resolved_inner = replace(
        old_inner, label="LW3", polygon=inner_polygon,
        l_mm=round((inner_maxx-inner_minx)*to_mm, 1),
        source=f"{old_inner.source}+global_core_topology",
        confidence=max(old_inner.confidence,
                       min(0.99, 0.78+0.18*min(inner_coverage))),
        mapping_status="verified")
    resolved_bottom = replace(
        old_bottom, label="LW1",
        polygon=(old_bottom.polygon if bottom_identity_correct
                 else bottom_polygon),
        l_mm=round((outer_maxx-outer_minx)*to_mm, 1),
        source=f"{old_bottom.source}+global_core_topology",
        confidence=max(old_bottom.confidence,
                       min(0.99, 0.78+0.18*min(bottom_coverage))),
        mapping_status="verified")

    consumed = {inner.source_index, bottom_source.source_index}
    resolved = [wall for index, wall in enumerate(walls)
                if index not in consumed]
    resolved.extend([resolved_inner, resolved_bottom])
    assignment.assignments = {
        "LW3": inner.candidate_id, "LW1": bottom_source.candidate_id}
    for row, new_wall in ((inner, resolved_inner),
                          (bottom_source, resolved_bottom)):
        row.vector_coverage = (inner_coverage if row is inner
                               else bottom_coverage)
        row.topology_scores = {
            "LW3": round(_score_horizontal(row, inner_bbox), 4),
            "LW1": round(_score_horizontal(row, bottom_bbox), 4)}
        if (row.current_label != new_wall.label or
                not _bounds_close(row.bbox, new_wall.polygon.bounds)):
            assignment.corrections.append(CoreWallIdentityCorrection(
                candidate_id=row.candidate_id,
                from_label=row.current_label, to_label=new_wall.label,
                reason=("inner run between LW4/LW5" if new_wall.label == "LW3"
                        else "bottom chord from LW2 outer face to LW7 outer face"),
                before_bbox=tuple(row.bbox),
                after_bbox=tuple(new_wall.polygon.bounds)))

    # This is exactly the geometry the old label-driven extension invented.
    # Keep it only as audit evidence so it can never enter the Ruby model.
    if not inner_identity_correct:
        old_inner_full = box(outer_minx, inner.bbox[1],
                             outer_maxx, inner.bbox[3])
        rejected = old_inner_full.difference(inner_polygon)
        assignment.rejected_extension_bboxes = [tuple(g.bounds) for g in
            getattr(rejected, "geoms", [rejected]) if not g.is_empty]
    assignment.status = "verified"
    return _finish(page, walls, resolved, candidates, assignment, out_dir)


def resolve_lw1_topology(page: fitz.Page, paths: list, classes: list,
                         walls: list[WallFootprint], scale: float,
                         openings: list, cfg, out_dir: Path):
    """Backward-compatible wrapper for the global core topology resolver."""
    return resolve_core_wall_topology(
        page, paths, classes, walls, scale, openings, cfg, out_dir)


def _finish(page, before, after, candidates, assignment, out_dir):
    pnum = page.number+1
    candidate_rows = [asdict(row) for row in candidates]
    report = {
        "page": pnum, "topology": "global LW1..LW7",
        "status": assignment.status,
        "assignments": assignment.assignments,
        "corrections": [asdict(row) for row in assignment.corrections],
        "rejected_extension_bboxes": assignment.rejected_extension_bboxes,
        "warnings": assignment.warnings,
        "recoveries": [asdict(row) for row in assignment.corrections
                       if row.to_label == "LW1"],
    }
    _write_json(out_dir / f"core_wall_run_candidates_p{pnum:02d}.json",
                candidate_rows)
    _write_json(out_dir / f"core_wall_global_assignment_p{pnum:02d}.json",
                report)
    _write_json(out_dir / f"core_wall_identity_corrections_p{pnum:02d}.json",
                report["corrections"])
    _write_json(out_dir / f"core_wall_topology_p{pnum:02d}.json", report)
    _write_json(out_dir / f"lw1_recovery_p{pnum:02d}.json", report)
    _write_json(out_dir / "core_wall_topology_readiness.json", report)
    _render_candidates(page, candidates,
                       out_dir / f"core_wall_run_candidates_p{pnum:02d}.png")
    _render_assignment(page, after, assignment,
                       out_dir / f"core_wall_global_assignment_p{pnum:02d}.png")
    _render_assignment(page, after, assignment,
                       out_dir / f"lw1_recovery_p{pnum:02d}.png")
    _render_assignment(page, after, assignment,
                       out_dir / f"core_wall_topology_p{pnum:02d}.png")
    return after, report


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _base_image(page):
    from PIL import Image
    factor = 120/72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA"), factor


def _render_candidates(page, candidates, path):
    from PIL import Image, ImageDraw
    image, factor = _base_image(page)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for row in candidates:
        minx, miny, maxx, maxy = row.bbox
        coords = [(minx*factor, miny*factor), (maxx*factor, maxy*factor)]
        draw.rectangle(coords, outline=(0, 190, 220, 255), width=3)
        draw.text((minx*factor, miny*factor-12),
                  f"{row.candidate_id}:{row.current_label}",
                  fill=(20, 120, 140, 255))
    Image.alpha_composite(image, overlay).convert("RGB").save(path)


def _render_assignment(page, walls, assignment, path):
    from PIL import Image, ImageDraw
    image, factor = _base_image(page)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for bounds in assignment.rejected_extension_bboxes:
        minx, miny, maxx, maxy = bounds
        draw.rectangle([(minx*factor, miny*factor),
                        (maxx*factor, maxy*factor)],
                       fill=(220, 30, 40, 80), outline=(220, 30, 40, 255),
                       width=3)
    for wall in walls:
        if not wall.label.upper().startswith("LW"):
            continue
        for geom in getattr(wall.polygon, "geoms", [wall.polygon]):
            if not isinstance(geom, Polygon):
                continue
            points = [(x*factor, y*factor) for x, y in geom.exterior.coords]
            draw.polygon(points, fill=(145, 30, 180, 145),
                         outline=(110, 15, 145, 255))
            minx, miny, _maxx, _maxy = geom.bounds
            draw.text((minx*factor, miny*factor-12), wall.label,
                      fill=(90, 10, 130, 255))
    Image.alpha_composite(image, overlay).convert("RGB").save(path)


def reconcile_core_wall_topologies(storeys: list[dict], out_dir: Path) -> dict:
    """Choose a topology reference without copying its page coordinates."""
    rows = []
    for storey in storeys:
        result = storey["result"]
        labels = {wall.label.upper() for wall in result.walls}
        complete = len(labels & {f"LW{i}" for i in range(1, 8)})
        topology = result.wall_readiness.get("core_topology_report", {})
        junction = result.wall_readiness.get("junction_report", {})
        derived = sum(
            "topology" in str(wall.source).lower()
            for wall in result.walls
            if wall.label.upper() in {f"LW{i}" for i in range(1, 8)})
        score = complete*10
        score += 5 if topology.get("status") == "verified" else 0
        score += 3 if junction.get("status") == "verified" else 0
        score -= 2*derived
        rows.append({
            "page": result.page_index+1,
            "level_id": storey.get("level_id", ""),
            "lw_symbols": sorted(labels & {f"LW{i}" for i in range(1, 8)}),
            "completeness": complete, "score": score,
            "topology_derived_walls": derived,
            "topology_status": topology.get("status", "review"),
            "junction_status": junction.get("status", "review"),
        })
    reference = max(rows, key=lambda row: row["score"], default=None)
    report = {
        "status": "verified" if reference and reference["completeness"] == 7
        else "review",
        "reference": reference,
        "topology_contract": {
            "LW1": ["LW2_outer_face", "LW7_outer_face"],
            "LW3": ["LW4_inner_face", "LW5_inner_face"]},
        "coordinate_policy": "target-page vectors only",
        "floors": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "core_wall_topology_reference.json", report)
    for storey in storeys:
        result = storey["result"]
        readiness = {
            "page": result.page_index+1,
            "opening_status": result.opening_report.get("judge_status", "review"),
            "core_topology_status": result.wall_readiness.get(
                "core_topology_status", "review"),
            "junction_status": result.wall_readiness.get(
                "junction_status", "review"),
            "stair_solids": sum(element.type == "STAIR"
                                for element in result.render_elements),
            "shaft_solids": sum(element.type in {"SHAFT", "LIFT", "CORE"}
                                for element in result.render_elements),
        }
        page_dir = Path(result.debug_dir)
        _write_json(page_dir / "opening_wall_readiness.json", readiness)
    return report
