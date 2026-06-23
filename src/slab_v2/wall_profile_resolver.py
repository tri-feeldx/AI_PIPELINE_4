"""Wall topology recovery and elevation-profile extraction.

Key-plan/elevation pages provide semantic topology and vertical geometry.
The floor plan remains the coordinate authority. Gemini selects only IDs;
all geometry is generated from PDF vectors.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import fitz
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from src.slab_v2 import gemini_client
from src.slab_v2.models import WallElevationProfile, WallFootprint

PT_TO_MM = 25.4 / 72.0
_PERIMETER_RE = re.compile(r"^W\d+[A-Z]?$", re.I)
_SCALE_RE = re.compile(r"SCALE\s*[:=]?\s*1\s*:\s*(\d{1,4})", re.I)

_JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "profiles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "wall_symbol": {"type": "STRING"},
                    "candidate_ids": {
                        "type": "ARRAY", "items": {"type": "STRING"}},
                    "confidence": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["wall_symbol", "candidate_ids",
                             "confidence", "reason"],
            },
        },
        "warnings": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["profiles", "warnings"],
}


def _words(page: fitz.Page) -> list:
    return page.get_text("words")


def _title_anchors(page: fitz.Page) -> list[dict]:
    words = _words(page)
    rows = []
    for i, word in enumerate(words):
        if str(word[4]).upper() != "WALL":
            continue
        tail = words[i + 1:i + 5]
        elev = next((w for w in tail if str(w[4]).upper() == "ELEVATION"), None)
        symbol = next((w for w in tail
                       if re.fullmatch(r"(?:LW|W|SW|RW)\d+[A-Z]?",
                                       str(w[4]), re.I)), None)
        if not elev or not symbol:
            continue
        rows.append({
            "symbol": str(symbol[4]).upper(),
            "bbox": (word[0], min(word[1], elev[1], symbol[1]),
                     symbol[2], max(word[3], elev[3], symbol[3])),
            "cx": (word[0] + symbol[2]) / 2,
            "cy": (word[1] + symbol[3]) / 2,
        })
    return rows


def _view_boxes(page: fitz.Page, anchors: list[dict]) -> dict[str, tuple]:
    if not anchors:
        return {}
    rows = []
    for anchor in sorted(anchors, key=lambda a: a["cy"]):
        group = next((g for g in rows
                      if abs(g[0]["cy"] - anchor["cy"]) < 60), None)
        if group is None:
            rows.append([anchor])
        else:
            group.append(anchor)
    result = {}
    for ri, group in enumerate(rows):
        group.sort(key=lambda a: a["cx"])
        previous_title_y = max((a["cy"] for g in rows[:ri] for a in g),
                               default=page.rect.y0)
        top = page.rect.y0 + 20 if ri == 0 else previous_title_y + 30
        for i, anchor in enumerate(group):
            left = (page.rect.x0 + 20 if i == 0 else
                    (group[i - 1]["cx"] + anchor["cx"]) / 2)
            right = (min(page.rect.x1 * 0.80, page.rect.x1 - 20)
                     if i == len(group) - 1 else
                     (anchor["cx"] + group[i + 1]["cx"]) / 2)
            result[anchor["symbol"]] = (
                left, top, right, max(top + 40, anchor["bbox"][3] + 35))
    return result


def _drawing_rect_candidate(drawing: dict, view: fitz.Rect) -> tuple | None:
    fill = drawing.get("fill")
    rect = drawing.get("rect")
    if not fill or not rect or not view.intersects(rect):
        return None
    luminance = sum(fill) / 3.0
    if not 0.55 <= luminance <= 0.96:
        return None
    if rect.width < 30 or rect.height < 18 or rect.width / rect.height < 1.5:
        return None
    if not view.contains(rect):
        return None
    return tuple(rect)


def _black_profile_segments(page: fitz.Page, rect: tuple) -> list[tuple]:
    """Return horizontal/diagonal black edges that can define a wall top.

    Some structural PDFs paint the wall body as a rectangular gray fill and
    draw the actual sloped/stepped top edge over it. PyMuPDF consequently
    reports only the fill rectangle. Reconstructing the visible upper envelope
    preserves the elevation profile without raster tracing.
    """
    clip = fitz.Rect(rect) + (-2, -2, 2, 2)
    segments = []
    for drawing in page.get_drawings():
        color = drawing.get("color")
        if color is None or max(color) > 0.20:
            continue
        if float(drawing.get("width") or 0) < 0.70:
            continue
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            if not clip.contains(p1) or not clip.contains(p2):
                continue
            if abs(p2.x-p1.x) < 4:
                continue
            x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            segments.append((x1, y1, x2, y2))
    return segments


def _visible_profile_polygon(page: fitz.Page, rect: tuple) -> list[list[float]]:
    """Rebuild the filled wall polygon from its vector upper envelope."""
    x0, y0, x1, y1 = rect
    segments = _black_profile_segments(page, rect)
    if not segments:
        return [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]

    xs = {x0, x1}
    for sx0, _sy0, sx1, _sy1 in segments:
        xs.add(max(x0, min(x1, sx0)))
        xs.add(max(x0, min(x1, sx1)))
    top = []
    for x in sorted(xs):
        values = []
        for sx0, sy0, sx1, sy1 in segments:
            if sx0-0.25 <= x <= sx1+0.25:
                t = 0.0 if abs(sx1-sx0) < 1e-6 else (x-sx0)/(sx1-sx0)
                values.append(sy0 + max(0.0, min(1.0, t))*(sy1-sy0))
        # The geometric top has the smallest PDF y coordinate. Restrict the
        # search to the fill's vertical extent so dimensions above are ignored.
        values = [y for y in values if y0-2 <= y <= y1+1]
        top.append((x, min(values) if values else y0))

    # Small offsets are normally reinforcement/dimension strokes laid over a
    # rectangular wall body. Preserve only material slope/step changes.
    if top and max(y for _x, y in top) - min(y for _x, y in top) <= 8.0:
        top = [(x, y0) for x, _y in top]

    # Drop redundant collinear samples but retain slope/step vertices.
    cleaned = []
    for point in top:
        if len(cleaned) >= 2:
            a, b = cleaned[-2], cleaned[-1]
            cross = ((b[0]-a[0])*(point[1]-b[1]) -
                     (b[1]-a[1])*(point[0]-b[0]))
            if abs(cross) < 0.05:
                cleaned[-1] = point
                continue
        cleaned.append(point)
    polygon = [(x0, y1), (x1, y1)] + list(reversed(cleaned))
    return [[round(x, 3), round(y, 3)] for x, y in polygon]


def _profile_candidates(page: fitz.Page, boxes: dict[str, tuple]) -> list[dict]:
    candidates = []
    drawings = page.get_drawings()
    for symbol, box in boxes.items():
        view = fitz.Rect(box)
        index = 0
        for drawing in drawings:
            rect = _drawing_rect_candidate(drawing, view)
            if rect is None:
                continue
            index += 1
            candidates.append({
                "id": f"{symbol}_panel_{index:02d}",
                "wall_symbol": symbol,
                "bbox": [round(v, 3) for v in rect],
                "polygon_pdf": _visible_profile_polygon(page, rect),
                "area_pt2": round((rect[2]-rect[0])*(rect[3]-rect[1]), 1),
                "source": "vector_fill",
            })
    return candidates


def _render_candidates(page: fitz.Page, candidates: list[dict], path: Path) -> bytes:
    from PIL import Image, ImageDraw
    factor = 120 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in candidates:
        x0, y0, x1, y1 = row["bbox"]
        box = (x0*factor, y0*factor, x1*factor, y1*factor)
        points = [(x*factor, y*factor) for x, y in row.get(
            "polygon_pdf", [])]
        if points:
            draw.polygon(points, outline=(220, 40, 40), fill=(220, 40, 40, 45))
        else:
            draw.rectangle(box, outline=(220, 40, 40), width=4)
        draw.text((box[0], max(0, box[1]-15)), row["id"], fill=(180, 0, 0))
    image.save(path)
    return path.read_bytes()


def _page_scale(page: fitz.Page, box: tuple) -> tuple[float, str]:
    text = page.get_text("text", clip=fitz.Rect(box))
    values = {int(x) for x in _SCALE_RE.findall(text)}
    if len(values) == 1:
        return float(next(iter(values))), "verified_view_text"
    all_values = {int(x) for x in _SCALE_RE.findall(page.get_text("text"))}
    if len(all_values) == 1:
        return float(next(iter(all_values))), "verified_unique_page_text"
    return 0.0, "missing_or_ambiguous"


def _profiles_from_decision(page: fitz.Page, boxes: dict[str, tuple],
                            candidates: list[dict], decision: dict) -> dict:
    valid = {row["id"]: row for row in candidates}
    decisions = {str(row.get("wall_symbol", "")).upper(): row
                 for row in decision.get("profiles", [])}
    profiles = {}
    for symbol, box in boxes.items():
        available = [row for row in candidates if row["wall_symbol"] == symbol]
        picked = [valid[cid] for cid in decisions.get(symbol, {}).get(
            "candidate_ids", []) if cid in valid and valid[cid]["wall_symbol"] == symbol]
        accepted = bool(picked) and float(decisions.get(symbol, {}).get(
            "confidence") or 0) >= 0.80
        selected = picked if accepted else available
        if not selected:
            continue
        x_min = min(row["bbox"][0] for row in selected)
        x_max = max(row["bbox"][2] for row in selected)
        baseline = max(row["bbox"][3] for row in selected)
        view_grids = _grid_anchors(page, fitz.Rect(box))
        relevant_grids = {
            label: point for label, point in view_grids.items()
            if x_min-50 <= point[0] <= x_max+50}
        grid_sequence = sorted(relevant_grids, key=lambda label:
                               relevant_grids[label][0])
        if len(grid_sequence) >= 2:
            station_x0 = relevant_grids[grid_sequence[0]][0]
            station_x1 = relevant_grids[grid_sequence[-1]][0]
        else:
            station_x0, station_x1 = x_min, x_max
        span = max(station_x1 - station_x0, 1e-6)
        grid_stations = {label: round(
            (relevant_grids[label][0]-station_x0)/span, 6)
            for label in grid_sequence}
        scale, scale_status = _page_scale(page, box)
        panels = []
        for row in sorted(selected, key=lambda r: r["bbox"][0]):
            x0, y0, x1, y1 = row["bbox"]
            polygon_pdf = row.get("polygon_pdf") or [
                [x0, y1], [x1, y1], [x1, y0], [x0, y0]]
            polygon_station_z = [[
                round((x-station_x0)/span, 6),
                round((baseline-y)*PT_TO_MM*scale, 1) if scale else 0.0,
            ] for x, y in polygon_pdf]
            panels.append({
                "candidate_id": row["id"],
                "s0": round((x0-station_x0)/span, 6),
                "s1": round((x1-station_x0)/span, 6),
                "polygon_station_z": polygon_station_z,
            })
        confidence = float(decisions.get(symbol, {}).get("confidence") or 0)
        status = "verified" if accepted and scale else "review"
        profile = WallElevationProfile(
            profile_id=f"profile_{symbol}", wall_symbol=symbol,
            source_page=page.number, source_view_bbox=tuple(box),
            panels=panels, from_level="level_1", to_level="level_2",
            grid_start=grid_sequence[0] if grid_sequence else "",
            grid_end=grid_sequence[-1] if grid_sequence else "",
            grid_sequence=grid_sequence, grid_stations=grid_stations,
            scale_ratio=scale, scale_status=scale_status,
            confidence=confidence if accepted else 0.55, status=status,
            warnings=[] if status == "verified" else [
                "Profile uses deterministic fill fallback or unverified scale."])
        profiles[symbol] = asdict(profile)
    return profiles


def _keyplan_crop(page: fitz.Page) -> fitz.Rect | None:
    blocks = page.get_text("blocks")
    title = next((b for b in blocks if "WALL KEY PLAN" in
                  " ".join(str(b[4]).upper().split())), None)
    if title is None:
        return None
    cx = (title[0] + title[2]) / 2
    return fitz.Rect(max(page.rect.x0, cx-450), max(page.rect.y0, title[1]-720),
                     min(page.rect.x1, cx+450), title[1]-10)


def _grid_anchors(page: fitz.Page, clip: fitz.Rect | None = None) -> dict:
    """Find primary grid bubbles, excluding smaller column-mark bubbles."""
    drawings = page.get_drawings()
    bubbles = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None or not (28 <= rect.width <= 48 and
                                28 <= rect.height <= 48):
            continue
        if clip is not None and not clip.intersects(rect):
            continue
        if not any(item and item[0] == "c" for item in drawing.get("items", [])):
            continue
        bubbles.append(rect)
    anchors = {}
    for word in page.get_text("words", clip=clip):
        label = str(word[4]).strip().upper()
        if not re.fullmatch(r"(?:[A-Z]|\d{1,2})", label):
            continue
        cx, cy = (word[0]+word[2])/2, (word[1]+word[3])/2
        bubble = next((rect for rect in bubbles if rect.contains((cx, cy))), None)
        if bubble is None:
            continue
        anchors[label] = [round((bubble.x0+bubble.x1)/2, 3),
                          round((bubble.y0+bubble.y1)/2, 3)]
    return anchors


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([source, np.ones(len(source))])
    matrix, *_ = np.linalg.lstsq(design, target, rcond=None)
    return matrix


def _apply_affine(point, matrix: np.ndarray) -> list[float]:
    value = np.array([point[0], point[1], 1.0]) @ matrix
    return [float(value[0]), float(value[1])]


def _solve_grid_registration(keyplan_anchors: dict,
                             plan_anchors: dict) -> dict:
    labels = sorted(set(keyplan_anchors) & set(plan_anchors))
    numeric = [label for label in labels if label.isdigit()]
    alpha = [label for label in labels if label.isalpha()]
    # Grid bubbles are deliberately offset outside the drawing. Numeric
    # bubbles constrain the X station; alphabetic bubbles constrain the Y
    # station. Fitting their full 2D bubble centres would introduce fake shear.
    if len(numeric) >= 2 and len(alpha) >= 2:
        key_x = np.array([keyplan_anchors[label][0] for label in numeric])
        plan_x = np.array([plan_anchors[label][0] for label in numeric])
        key_y = np.array([keyplan_anchors[label][1] for label in alpha])
        plan_y = np.array([plan_anchors[label][1] for label in alpha])
        ax, bx = np.polyfit(key_x, plan_x, 1)
        ay, by = np.polyfit(key_y, plan_y, 1)
        x_errors = np.abs(ax*key_x + bx-plan_x)
        y_errors = np.abs(ay*key_y + by-plan_y)
        inlier_numeric = [label for label, error in zip(numeric, x_errors)
                          if error <= 8.0]
        inlier_alpha = [label for label, error in zip(alpha, y_errors)
                        if error <= 8.0]
        inliers = inlier_numeric + inlier_alpha
        errors = np.concatenate([x_errors[x_errors <= 8.0],
                                 y_errors[y_errors <= 8.0]])
        rms = float(np.sqrt(np.mean(errors**2))) if len(errors) else 1e9
        status = ("verified" if len(inlier_numeric) >= 2 and
                  len(inlier_alpha) >= 2 and rms <= 5.0 else "review")
        matrix = np.array([[ax, 0.0], [0.0, ay], [bx, by]])
        return {
            "status": status, "registration_kind": "orthogonal_grid_axes",
            "labels": labels, "inliers": inliers,
            "matrix": [[round(float(value), 9) for value in row]
                       for row in matrix],
            "rms_error_pt": round(rms, 3),
            "warnings": [] if status == "verified" else [
                "Orthogonal grid registration lacks independent X/Y consensus."],
        }
    if len(labels) < 3:
        return {"status": "review", "labels": labels, "inliers": [],
                "matrix": None, "rms_error_pt": None,
                "warnings": ["Fewer than three shared grid anchors."]}
    source = np.array([keyplan_anchors[label] for label in labels], dtype=float)
    target = np.array([plan_anchors[label] for label in labels], dtype=float)
    best = None
    for indices in itertools.combinations(range(len(labels)), 3):
        sample = source[list(indices)]
        if abs(np.cross(sample[1]-sample[0], sample[2]-sample[0])) < 1.0:
            continue
        matrix = _fit_affine(sample, target[list(indices)])
        predicted = np.column_stack([source, np.ones(len(source))]) @ matrix
        errors = np.linalg.norm(predicted-target, axis=1)
        inliers = np.where(errors <= 8.0)[0]
        score = (len(inliers), -float(np.median(errors[inliers]))
                 if len(inliers) else -1e9)
        if best is None or score > best[0]:
            best = (score, inliers, matrix)
    if best is None or len(best[1]) < 3:
        return {"status": "review", "labels": labels, "inliers": [],
                "matrix": None, "rms_error_pt": None,
                "warnings": ["Grid anchors are collinear or inconsistent."]}
    inliers = best[1]
    matrix = _fit_affine(source[inliers], target[inliers])
    predicted = np.column_stack([source[inliers], np.ones(len(inliers))]) @ matrix
    errors = np.linalg.norm(predicted-target[inliers], axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    status = "verified" if rms <= 5.0 else "review"
    return {
        "status": status,
        "labels": labels,
        "inliers": [labels[i] for i in inliers],
        "matrix": [[round(float(value), 9) for value in row]
                   for row in matrix],
        "rms_error_pt": round(rms, 3),
        "warnings": [] if status == "verified" else [
            "Grid registration residual exceeds 5 PDF points."],
    }


def _line_segments(page: fitz.Page, clip: fitz.Rect) -> list[tuple]:
    segments = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            if not clip.contains(p1) or not clip.contains(p2):
                continue
            length = math.hypot(p2.x-p1.x, p2.y-p1.y)
            if length >= 20:
                segments.append((p1.x, p1.y, p2.x, p2.y, length))
    return segments


def _point_segment_distance(px, py, seg) -> float:
    x1, y1, x2, y2, _ = seg
    dx, dy = x2-x1, y2-y1
    if dx*dx + dy*dy == 0:
        return math.hypot(px-x1, py-y1)
    t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy)/(dx*dx+dy*dy)))
    return math.hypot(px-(x1+t*dx), py-(y1+t*dy))


def _extract_keyplan(page: fitz.Page) -> dict:
    clip = _keyplan_crop(page)
    if clip is None:
        return {"status": "missing", "symbols": {}, "warnings": [
            "No WALL KEY PLAN title found."]}
    segments = _line_segments(page, clip)
    symbols = {}
    for word in page.get_text("words", clip=clip):
        symbol = str(word[4]).upper()
        if not _PERIMETER_RE.fullmatch(symbol):
            continue
        cx, cy = (word[0]+word[2])/2, (word[1]+word[3])/2
        nearby = [seg for seg in segments if _point_segment_distance(cx, cy, seg) <= 35]
        if not nearby:
            continue
        seg = max(nearby, key=lambda s: s[4])
        orientation = "horizontal" if abs(seg[2]-seg[0]) >= abs(seg[3]-seg[1]) else "vertical"
        symbols[symbol] = {
            "label_bbox": [round(v, 2) for v in word[:4]],
            "segment": [round(v, 2) for v in seg[:4]],
            "orientation": orientation,
        }
    grid_anchors = _grid_anchors(page, clip)
    return {"status": "verified" if symbols and len(grid_anchors) >= 3 else "review",
            "page": page.number + 1, "crop": tuple(clip),
            "symbols": symbols, "grid_anchors": grid_anchors,
            "warnings": [] if symbols and len(grid_anchors) >= 3 else [
                "Key plan needs wall centerlines and at least three grid anchors."]}


def _render_keyplan(page: fitz.Page, keyplan: dict, path: Path) -> None:
    from PIL import Image, ImageDraw
    factor = 150 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(image)
    for symbol, row in keyplan.get("symbols", {}).items():
        x1, y1, x2, y2 = row["segment"]
        draw.line((x1*factor, y1*factor, x2*factor, y2*factor),
                  fill=(0, 180, 220), width=6)
        draw.text((x1*factor, y1*factor), symbol, fill=(0, 90, 150))
    image.save(path)


def build_wall_source_registry(pdf_path: str, analysis, cfg,
                               out_dir: Path, use_ai: bool = True) -> dict:
    """Build one reusable registry before parallel plan-page extraction."""
    doc = fitz.open(pdf_path)
    elevation_pages = list(analysis.wall_elevation_pages or [])
    registry = {"status": "review", "profiles": {}, "keyplan": {},
                "source_pages": [p+1 for p in elevation_pages], "warnings": []}
    expected_profile_symbols = {s for b in analysis.buildings for f in b.floors
                                for s in (f.walls or {}) if _PERIMETER_RE.fullmatch(s)}
    for pi in elevation_pages:
        page = doc[pi]
        anchors = _title_anchors(page)
        boxes = _view_boxes(page, anchors)
        boxes = {symbol: box for symbol, box in boxes.items()
                 if symbol in expected_profile_symbols}
        candidates = _profile_candidates(page, boxes)
        if not candidates:
            continue
        overlay_path = out_dir / f"wall_elevation_candidates_p{pi+1:02d}.png"
        image = _render_candidates(page, candidates, overlay_path)
        (out_dir / f"wall_elevation_candidates_p{pi+1:02d}.json").write_text(
            json.dumps({"views": boxes, "candidates": candidates}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        prompt = f"""You are the semantic judge for vector wall-elevation panels.
The image labels code-generated filled vector candidates. Select only IDs that
form the actual concrete wall body for each WALL ELEVATION title. Do not select
slabs, columns, reinforcement, notes, soil, dimensions or title content. Return
only supplied IDs; never invent coordinates.

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}
"""
        (out_dir / "wall_source_planner_prompt.txt").write_text(
            prompt, encoding="utf-8")
        decision = {"profiles": [], "warnings": []}
        if use_ai:
            try:
                decision = gemini_client.call_gemini_json(
                    prompt, [image], _JUDGE_SCHEMA, cfg.gemini_model,
                    log_path=str(out_dir / "prompts.log"),
                    tag="wall_elevation_profile_judge",
                    raw_path=str(out_dir / "wall_source_planner_raw.txt"))
            except Exception as exc:
                registry["warnings"].append(f"Wall profile judge failed: {exc}")
        (out_dir / "wall_source_planner.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
        registry["profiles"].update(
            _profiles_from_decision(page, boxes, candidates, decision))
        keyplan = _extract_keyplan(page)
        if keyplan.get("symbols"):
            registry["keyplan"] = keyplan
            _render_keyplan(page, keyplan, out_dir / "wall_keyplan_topology.png")
            (out_dir / "wall_keyplan_topology.json").write_text(
                json.dumps(keyplan, indent=2, ensure_ascii=False), encoding="utf-8")
    doc.close()
    verified_profiles = {s for s, p in registry["profiles"].items()
                         if p.get("status") == "verified"}
    registry["status"] = ("verified" if expected_profile_symbols
                          and expected_profile_symbols <= verified_profiles
                          and registry.get("keyplan", {}).get("status") == "verified"
                          else "review")
    missing = sorted(expected_profile_symbols - set(registry["profiles"]))
    if missing:
        registry["warnings"].append("Missing elevation profiles: " + ", ".join(missing))
    (out_dir / "wall_elevation_profiles.json").write_text(
        json.dumps(registry["profiles"], indent=2, ensure_ascii=False), encoding="utf-8")
    return registry


def _anchor_for_symbol(page: fitz.Page, symbol: str) -> tuple[float, float] | None:
    for word in page.get_text("words"):
        if str(word[4]).upper().strip(".,:()") == symbol.upper():
            return ((word[0]+word[2])/2, (word[1]+word[3])/2)
    return None


def _wall_center_from_polygon(wall: WallFootprint, orientation: str) -> float:
    c = wall.polygon.centroid
    return c.y if orientation == "horizontal" else c.x


def resolve_plan_wall_topology(page: fitz.Page, slab_union, walls: list,
                               wall_types: dict, expected: dict,
                               registry: dict, scale: float,
                               out_dir: Path) -> tuple[list, dict]:
    """Recover expected perimeter wall runs from boundary + key-plan evidence."""
    if slab_union is None or slab_union.is_empty or not scale:
        return walls, {"status": "review", "warnings": [
            "No slab boundary/scale for wall topology recovery."]}
    largest = max(getattr(slab_union, "geoms", [slab_union]), key=lambda g: g.area)
    minx, miny, maxx, maxy = largest.bounds
    page_no = page.number + 1
    initial_rows = [{
        "label": wall.label,
        "bounds": [round(v, 3) for v in wall.polygon.bounds],
        "length_mm": wall.l_mm,
        "thickness_mm": wall.w_mm,
        "source": wall.source,
        "mapping_status": wall.mapping_status,
    } for wall in walls]
    (out_dir / f"wall_plan_candidates_p{page_no:02d}.json").write_text(
        json.dumps(initial_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_plan_resolution(
        page, walls, [], out_dir / f"wall_plan_candidates_p{page_no:02d}.png")

    key_symbols = registry.get("keyplan", {}).get("symbols", {})
    plan_grid_anchors = _grid_anchors(page)
    grid_registration = _solve_grid_registration(
        registry.get("keyplan", {}).get("grid_anchors", {}),
        plan_grid_anchors)
    affine = (np.array(grid_registration["matrix"], dtype=float)
              if grid_registration.get("matrix") else None)
    profiles = registry.get("profiles", {})
    retained = list(walls)
    rows, warnings = [], []
    for symbol, count in (expected or {}).items():
        if count != 1 or not _PERIMETER_RE.fullmatch(symbol):
            continue
        anchor = _anchor_for_symbol(page, symbol)
        existing = [w for w in retained if w.label == symbol]
        key = key_symbols.get(symbol, {})
        orientation = key.get("orientation")
        projected_segment = None
        if affine is not None and key.get("segment"):
            segment = key["segment"]
            p1 = _apply_affine(segment[:2], affine)
            p2 = _apply_affine(segment[2:], affine)
            projected_segment = p1 + p2
            orientation = ("horizontal" if abs(p2[0]-p1[0]) >=
                           abs(p2[1]-p1[1]) else "vertical")
        if not orientation and existing:
            bx = existing[0].polygon.bounds
            orientation = "horizontal" if bx[2]-bx[0] >= bx[3]-bx[1] else "vertical"
        if not anchor or orientation not in {"horizontal", "vertical"}:
            rows.append({"symbol": symbol, "status": "review",
                         "reason": "missing label or key-plan orientation"})
            continue
        ax, ay = anchor
        sides = ([('top', miny), ('bottom', maxy)] if orientation == "horizontal"
                 else [('left', minx), ('right', maxx)])
        side, edge_coord = min(sides, key=lambda item: abs(
            (ay if orientation == "horizontal" else ax) - item[1]))
        thickness = float(getattr(wall_types.get(symbol), "thickness_mm", 0) or 0)
        if thickness <= 0:
            rows.append({"symbol": symbol, "status": "review",
                         "reason": "missing wall thickness"})
            continue
        thickness_pt = thickness / (PT_TO_MM * scale)
        if existing:
            center = float(sum(_wall_center_from_polygon(w, orientation)
                               for w in existing) / len(existing))
        elif orientation == "horizontal":
            center = edge_coord + (thickness_pt/2 if side == 'top' else -thickness_pt/2)
        else:
            center = edge_coord + (thickness_pt/2 if side == 'left' else -thickness_pt/2)
        profile = profiles.get(symbol, {})
        grid_start = profile.get("grid_start", "")
        grid_end = profile.get("grid_end", "")
        start_anchor = plan_grid_anchors.get(grid_start)
        end_anchor = plan_grid_anchors.get(grid_end)
        has_grid_scope = bool(start_anchor and end_anchor)
        if orientation == "horizontal":
            start_coord = start_anchor[0] if has_grid_scope else minx
            end_coord = end_anchor[0] if has_grid_scope else maxx
            points = [(start_coord, center), (end_coord, center)]
        else:
            start_coord = start_anchor[1] if has_grid_scope else miny
            end_coord = end_anchor[1] if has_grid_scope else maxy
            points = [(center, start_coord), (center, end_coord)]
        line = LineString(points)
        footprint = line.buffer(thickness_pt/2, cap_style=2, join_style=2)
        verified = (bool(existing) and key.get("orientation") == orientation
                    and grid_registration.get("status") == "verified"
                    and profile.get("status") == "verified"
                    and has_grid_scope)
        recovered = WallFootprint(
            label=symbol, polygon=footprint, w_mm=thickness,
            l_mm=round(line.length*PT_TO_MM*scale, 1), wall_type=(
                getattr(wall_types.get(symbol), "wall_category", "wall") or "wall"),
            centerline=points, source="boundary_keyplan_recovery",
            confidence=0.95 if verified else 0.65,
            profile_id=profile.get("profile_id", ""),
            grid_start=grid_start, grid_end=grid_end,
            mapping_status="verified" if verified else "review")
        retained = [w for w in retained if w.label != symbol] + [recovered]
        prevented_gap_mm = max(0.0, recovered.l_mm - sum(w.l_mm for w in existing))
        rows.append({"symbol": symbol, "status": recovered.mapping_status,
                     "side": side, "orientation": orientation,
                     "length_mm": recovered.l_mm,
                     "recovered_missing_length_mm": round(prevented_gap_mm, 1),
                     "profile_id": recovered.profile_id,
                     "grid_start": grid_start,
                     "grid_end": grid_end,
                     "grid_scope_status": ("verified" if has_grid_scope
                                           else "missing"),
                     "projected_keyplan_segment": projected_segment,
                     "evidence": ["plan_label", "slab_outer_boundary"]
                     + (["existing_wall_fragment"] if existing else [])
                     + (["keyplan_topology"] if key else [])
                     + (["grid_ransac_registration"] if
                         grid_registration.get("status") == "verified" else [])
                     + (["elevation_grid_scope"] if has_grid_scope else [])})
    detected = {}
    for wall in retained:
        detected[wall.label] = detected.get(wall.label, 0) + 1
    missing = {s: n-detected.get(s, 0) for s, n in (expected or {}).items()
               if detected.get(s, 0) < n}
    perimeter_expected = {s for s, n in (expected or {}).items()
                          if n and _PERIMETER_RE.fullmatch(s)}
    perimeter_rows = [r for r in rows if r["symbol"] in perimeter_expected]
    if not perimeter_expected:
        status = "not_required"
    else:
        status = "verified" if not missing and perimeter_rows and all(
            r["status"] == "verified" for r in perimeter_rows) else "review"
    report = {"status": status, "expected": dict(expected or {}),
              "detected": detected, "missing": missing,
              "instances": rows, "warnings": warnings}
    registration = {
        **grid_registration,
        "mode": "grid_ransac_keyplan_to_floor_plan",
        "page": page_no,
        "keyplan_anchors": registry.get("keyplan", {}).get("grid_anchors", {}),
        "plan_anchors": plan_grid_anchors,
        "anchors": [{
            "symbol": row["symbol"],
            "keyplan_orientation": row.get("orientation"),
            "plan_side": row.get("side"),
            "profile_id": row.get("profile_id"),
        } for row in rows],
        "transform": grid_registration.get("matrix"),
        "reason": (
            "RANSAC grid registration projects key-plan wall orientation into "
            "the floor plan. Exact extent is then snapped to existing wall "
            "fragments and the floor-plan structural boundary."),
        "warnings": grid_registration.get("warnings", []) +
                    ([] if status == "verified" else [
                        "Wall topology could not be verified from all three sources."]),
    }
    (out_dir / f"wall_grid_registration_p{page_no:02d}.json").write_text(
        json.dumps(registration, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"wall_plan_resolved_p{page_no:02d}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_plan_resolution(page, retained, rows,
                            out_dir / f"wall_plan_resolved_p{page_no:02d}.png")
    _render_plan_resolution(
        page, retained, rows,
        out_dir / f"wall_grid_registration_p{page_no:02d}.png",
        registration=registration)
    mapping_report = {
        "page": page_no,
        "status": status,
        "walls": [{
            "symbol": wall.label,
            "plan_source": wall.source,
            "plan_status": wall.mapping_status,
            "profile_id": wall.profile_id,
            "profile_status": profiles.get(wall.label, {}).get("status", "missing"),
            "ruby_mode": ("elevation_profile_solid" if wall.profile_id and
                           profiles.get(wall.label, {}).get("status") == "verified"
                           else "debug_full_storey_prism"),
        } for wall in retained],
    }
    (out_dir / f"wall_3d_mapping_report_p{page_no:02d}.json").write_text(
        json.dumps(mapping_report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"wall_readiness_report_p{page_no:02d}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return retained, report


def _render_plan_resolution(page: fitz.Page, walls: list, rows: list,
                            path: Path, registration: dict | None = None) -> None:
    from PIL import Image, ImageDraw
    factor = 120 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for wall in walls:
        color = ((142, 36, 170, 150) if wall.mapping_status == "verified"
                 else (240, 140, 20, 140))
        for geom in getattr(wall.polygon, "geoms", [wall.polygon]):
            pts = [(x*factor, y*factor) for x, y in geom.exterior.coords]
            draw.polygon(pts, fill=color, outline=color[:3]+(255,))
        if wall.centerline:
            draw.line([(x*factor, y*factor) for x, y in wall.centerline],
                      fill=(0, 190, 220, 255), width=4)
    if registration:
        for label, point in registration.get("plan_anchors", {}).items():
            x, y = point[0]*factor, point[1]*factor
            radius = 7
            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         fill=(30, 90, 240, 230), outline=(0, 30, 140, 255))
            draw.text((x+8, y-8), label, fill=(0, 20, 120, 255))
        for row in rows:
            segment = row.get("projected_keyplan_segment")
            if segment:
                draw.line(tuple(v*factor for v in segment),
                          fill=(0, 220, 240, 255), width=6)
    image = Image.alpha_composite(image, overlay).convert("RGB")
    title = "purple=verified wall, orange=review, cyan=projected/centerline"
    if registration:
        title += (f", blue=grid anchors, RMS="
                  f"{registration.get('rms_error_pt')}pt")
    ImageDraw.Draw(image).text((10, 10), title, fill="black")
    image.save(path)
