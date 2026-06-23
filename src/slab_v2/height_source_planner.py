"""Semantic source planning and deterministic elevation-datum measurement.

Gemini selects useful pages/views. All coordinates, scales, distances and
final height evidence are produced and validated by code.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict
from pathlib import Path

import fitz

from src.slab_v2 import gemini_client
from src.slab_v2.models import HeightEvidence


_SOURCE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sources": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "page": {"type": "INTEGER"},
                    "building": {"type": "STRING"},
                    "view_type": {"type": "STRING"},
                    "levels_visible": {
                        "type": "ARRAY", "items": {"type": "STRING"}},
                    "level_pairs": {
                        "type": "ARRAY", "items": {"type": "STRING"}},
                    "scale_text": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["page", "building", "view_type",
                             "levels_visible", "confidence", "reason"],
            },
        },
        "warnings": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["sources", "warnings"],
}

_LEVEL_RE = re.compile(r"\b(?:LEVEL|LVL)\s*0*(\d{1,2})\b", re.I)
_SCALE_RE = re.compile(r"\bSCALE\s*(?:AT\s*A\d\s*)?[:=]?\s*1\s*[:/]\s*(\d{1,4})\b", re.I)
_DATUM_RE = re.compile(
    r"\b(FFL|RL|EL|AHD)\b\s*(?:=|:|\+)?\s*"
    r"([+-]?\d{1,5}(?:\.\d{1,3})?)\s*(MM|M|MAHD)?\b", re.I)
_SOURCE_WORDS = ("ELEVATION", "SECTION", "LEVEL", "FFL", "RL", "EL", "AHD", "SCALE")


def _candidate_pages(doc: fitz.Document, analysis) -> list[dict]:
    rows = []
    hinted = set(analysis.wall_elevation_pages or [])
    for key in ("elevation_pages", "section_pages", "height_schedule_pages",
                "storey_height_pages"):
        for value in (analysis.raw or {}).get(key, []) or []:
            try:
                hinted.add(int(value) - 1)
            except (TypeError, ValueError):
                pass
    for pi, page in enumerate(doc):
        text = page.get_text("text")[:14000]
        upper = text.upper()
        levels = sorted({int(x) for x in _LEVEL_RE.findall(upper)})
        score = sum(upper.count(word) for word in _SOURCE_WORDS)
        score += 8 if pi in hinted else 0
        score += min(len(levels), 6) * 3
        if len(levels) >= 2 and ("ELEVATION" in upper or "SECTION" in upper):
            score += 12
        if score < 8:
            continue
        rows.append({
            "page": pi + 1,
            "score": score,
            "levels": levels,
            "has_scale": bool(_SCALE_RE.search(upper)),
            "text": " ".join(text.split())[:3500],
        })
    return sorted(rows, key=lambda r: (-r["score"], r["page"]))[:12]


def _thumbnail(page: fitz.Page) -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(0.75, 0.75), alpha=False)
    return pix.tobytes("png")


def plan_height_sources(doc: fitz.Document, analysis, cfg, out_dir: Path) -> tuple[dict, list[dict]]:
    """One compact Gemini call. Failure leaves deterministic candidates usable."""
    candidates = _candidate_pages(doc, analysis)
    (out_dir / "height_candidate_pages.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    if not candidates:
        result = {"status": "no_candidates", "sources": [],
                  "warnings": ["No elevation/section height source candidates."]}
        (out_dir / "height_source_planner.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        return result, candidates

    selected = candidates[:8]
    image_pages = [r["page"] for r in selected]
    # Contact sheet is an audit artifact, not model input.
    try:
        from PIL import Image, ImageDraw
        thumbs = []
        for page_no in image_pages:
            image = Image.open(io.BytesIO(_thumbnail(doc[page_no - 1]))).convert("RGB")
            image.thumbnail((600, 800))
            canvas = Image.new("RGB", (image.width, image.height + 28), "white")
            canvas.paste(image, (0, 28))
            ImageDraw.Draw(canvas).text((8, 7), f"Page {page_no}", fill="black")
            thumbs.append(canvas)
        width = max(x.width for x in thumbs)
        height = sum(x.height for x in thumbs)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for image in thumbs:
            sheet.paste(image, (0, y))
            y += image.height
        sheet.save(out_dir / "height_candidate_pages.png")
    except Exception:
        pass
    prompt_rows = [{k: v for k, v in row.items() if k != "text"}
                   | {"text_excerpt": row["text"]} for row in selected]
    buildings = [{"name": b.name,
                  "levels": [f.level_id for f in b.floors]}
                 for b in analysis.buildings]
    prompt = f"""You are a semantic source planner for structural storey heights.

CPU prefiltered the following pages. IMAGE order exactly matches page order
{image_pages}. Select only elevation, section, level schedule, or stair-section
views that visibly contain at least two vertically separated level datums.
Ignore title-block addresses such as 'Level 18', plan-level names, slab
thickness, setdowns and grid numbers. A printed scale is useful but you must
never calculate or estimate a height. Code will find datum lines and measure.

KNOWN BUILDINGS/LEVELS:
{json.dumps(buildings, ensure_ascii=False)}

CANDIDATE PAGES:
{json.dumps(prompt_rows, ensure_ascii=False)}

Return page (1-based), building, view_type, levels_visible, possible level_pairs,
visible scale_text, confidence and a concise reason. Return no source rather
than guessing a building or interpreting an address as a datum.
"""
    (out_dir / "height_source_planner_prompt.txt").write_text(
        prompt, encoding="utf-8")
    raw_path = out_dir / "height_source_planner_raw.txt"
    try:
        data = gemini_client.call_gemini_json(
            prompt, [_thumbnail(doc[p - 1]) for p in image_pages],
            _SOURCE_SCHEMA, cfg.gemini_model,
            log_path=str(out_dir / "prompts.log"),
            tag="height_source_planner", raw_path=str(raw_path))
        valid_pages = {r["page"] for r in candidates}
        sources = []
        seen_sources = set()
        for row in data.get("sources", []):
            if row.get("page") not in valid_pages:
                continue
            if float(row.get("confidence") or 0) < 0.45:
                continue
            source_key = (row.get("page"), row.get("building"))
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            sources.append(row)
        result = {**data, "status": "ok", "sources": sources}
    except Exception as exc:
        result = {
            "status": "planner_failed", "sources": [],
            "warnings": [f"Height source planner failed: {exc}"],
        }
    (out_dir / "height_source_planner.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, candidates


def _segments(page: fitz.Page) -> list[tuple]:
    rows = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
            if max(dx, dy) < 35:
                continue
            if dy <= 1.5:
                rows.append(("h", p1.x, p1.y, p2.x, p2.y))
            elif dx <= 1.5:
                rows.append(("v", p1.x, p1.y, p2.x, p2.y))
    return rows


def _level_labels(page: fitz.Page) -> list[dict]:
    labels = []
    words = page.get_text("words")
    for i, word in enumerate(words):
        token = str(word[4]).strip().upper().rstrip(":")
        if token not in {"LEVEL", "LVL"}:
            continue
        for nxt in words[i + 1:i + 4]:
            value = str(nxt[4]).strip(".,:()")
            if re.fullmatch(r"0*\d{1,2}", value):
                labels.append({
                    "level": int(value),
                    "text": f"{word[4]} {nxt[4]}",
                    "bbox": (word[0], min(word[1], nxt[1]),
                             nxt[2], max(word[3], nxt[3])),
                })
                break
    return labels


def _nearest_datum(label: dict, segments: list[tuple]) -> tuple | None:
    x0, y0, x1, y1 = label["bbox"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    best = None
    for seg in segments:
        axis, ax, ay, bx, by = seg
        if axis == "h":
            lo, hi = sorted((ax, bx))
            along = 0 if lo - 80 <= cx <= hi + 80 else min(abs(cx-lo), abs(cx-hi))
            distance = abs(cy - ay) + along * 0.2
        else:
            lo, hi = sorted((ay, by))
            along = 0 if lo - 80 <= cy <= hi + 80 else min(abs(cy-lo), abs(cy-hi))
            distance = abs(cx - ax) + along * 0.2
        if distance <= 28 and (best is None or distance < best[0]):
            best = (distance, seg)
    return best[1] if best else None


def _scale_for_view(page: fitz.Page, bbox: tuple) -> tuple[float | None, str, list[str]]:
    page_text = page.get_text("text")
    all_scales = sorted({int(x) for x in _SCALE_RE.findall(page_text)})
    rect = fitz.Rect(bbox)
    pad = max(page.rect.width, page.rect.height) * 0.08
    clip = fitz.Rect(max(page.rect.x0, rect.x0 - pad),
                     max(page.rect.y0, rect.y0 - pad),
                     min(page.rect.x1, rect.x1 + pad),
                     min(page.rect.y1, rect.y1 + pad))
    local = page.get_text("text", clip=clip)
    local_scales = sorted({int(x) for x in _SCALE_RE.findall(local)})
    if len(local_scales) == 1:
        return float(local_scales[0]), "verified_local_text", [f"scale:1:{local_scales[0]}"]
    if not local_scales and len(all_scales) == 1:
        return float(all_scales[0]), "verified_unique_page_text", [f"scale:1:{all_scales[0]}"]
    return None, "missing_or_ambiguous", []


def _view_groups(labels: list[dict]) -> list[list[dict]]:
    rows = [x for x in labels if x.get("datum")]
    groups: list[list[dict]] = []
    for row in rows:
        seg = row["datum"]
        axis = seg[0]
        center = ((seg[1] + seg[3]) / 2 if axis == "h"
                  else (seg[2] + seg[4]) / 2)
        target = None
        for group in groups:
            gseg = group[0]["datum"]
            gcenter = ((gseg[1] + gseg[3]) / 2 if axis == "h"
                       else (gseg[2] + gseg[4]) / 2)
            if gseg[0] == axis and abs(center - gcenter) <= 140:
                target = group
                break
        if target is None:
            groups.append([row])
        else:
            target.append(row)
    return groups


def measure_planned_sources(doc: fitz.Document, analysis, planner: dict,
                            out_dir: Path) -> tuple[list[HeightEvidence], list[dict]]:
    evidence: list[HeightEvidence] = []
    viewport_reports = []
    known_buildings = {b.name for b in analysis.buildings}
    level_ids = {}
    for known_building in analysis.buildings:
        level_ids[known_building.name] = {}
        for floor in known_building.floors:
            match = re.search(r"(\d+)", floor.level_id or "")
            if match:
                level_ids[known_building.name][int(match.group(1))] = floor.level_id
    source_rows = planner.get("sources", [])
    for source in source_rows:
        pi = int(source["page"]) - 1
        if not 0 <= pi < len(doc):
            continue
        page = doc[pi]
        labels = _level_labels(page)
        segments = _segments(page)
        for label in labels:
            label["datum"] = _nearest_datum(label, segments)
        building = str(source.get("building") or "")
        if building not in known_buildings:
            building = next(iter(known_buildings), "") if len(known_buildings) == 1 else ""
        for vi, group in enumerate(_view_groups(labels), 1):
            unique = {row["level"]: row for row in group}
            if len(unique) < 2 or not building:
                continue
            axis = next(iter(unique.values()))["datum"][0]
            coords = []
            for level, row in unique.items():
                seg = row["datum"]
                coord = seg[2] if axis == "h" else seg[1]
                coords.append((level, coord, row))
            bbox = (min(r["bbox"][0] for _, _, r in coords),
                    min(r["bbox"][1] for _, _, r in coords),
                    max(r["bbox"][2] for _, _, r in coords),
                    max(r["bbox"][3] for _, _, r in coords))
            scale, scale_status, scale_ids = _scale_for_view(page, bbox)
            viewport_id = f"p{pi + 1:02d}_v{vi:02d}"
            normalized = [(lvl, round(coord / max(page.rect.width, page.rect.height), 4))
                          for lvl, coord, _ in sorted(coords)]
            fingerprint = hashlib.sha1(json.dumps({
                "axis": axis, "levels": normalized,
                "text": re.sub(r"\s+", " ", " ".join(r["text"] for _, _, r in coords).upper()),
            }, sort_keys=True).encode()).hexdigest()[:16]
            report = {"viewport_id": viewport_id, "page": pi + 1,
                      "building": building, "axis": axis,
                      "levels": sorted(unique), "bbox": bbox,
                      "scale_ratio": scale, "scale_status": scale_status,
                      "source_fingerprint": fingerprint, "measurements": []}
            ordered = sorted(coords)
            for (lower, c1, r1), (upper, c2, r2) in zip(ordered, ordered[1:]):
                if upper != lower + 1:
                    continue
                delta_pt = abs(c2 - c1)
                value = delta_pt * 25.4 / 72.0 * scale if scale else None
                if value is None or not 2000 <= value <= 8000:
                    report["measurements"].append({
                        "from": lower, "to": upper, "delta_pt": round(delta_pt, 2),
                        "value_mm": value, "status": "review"})
                    continue
                eid = f"elevation_{pi + 1:02d}_{vi:02d}_{lower}_{upper}"
                datum1, datum2 = tuple(r1["datum"]), tuple(r2["datum"])
                evidence.append(HeightEvidence(
                    eid, building,
                    level_ids.get(building, {}).get(lower, f"level_{lower}"),
                    level_ids.get(building, {}).get(upper, f"level_{upper}"),
                    "scaled_elevation_measurement", round(value, 1),
                    page_index=pi, bbox=bbox,
                    source_text=(f"LEVEL {lower:02d} to LEVEL {upper:02d}; "
                                 f"delta={delta_pt:.2f}pt; scale=1:{int(scale)}"),
                    extraction_method="datum_line_spacing",
                    confidence=0.80, is_absolute_datum=False,
                    viewport_id=viewport_id, source_fingerprint=fingerprint,
                    independence_group=fingerprint,
                    datum_line_from=datum1, datum_line_to=datum2,
                    scale_ratio=scale, scale_status=scale_status,
                    scale_evidence_ids=scale_ids))
                report["measurements"].append({
                    "id": eid, "from": lower, "to": upper,
                    "delta_pt": round(delta_pt, 2), "value_mm": round(value, 1),
                    "status": "measured"})
            viewport_reports.append(report)
    (out_dir / "height_viewports.json").write_text(
        json.dumps(viewport_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    return evidence, viewport_reports


def extract_planned_absolute_datums(doc: fitz.Document, analysis,
                                    planner: dict) -> list[HeightEvidence]:
    """Bind explicit FFL/RL/EL/AHD values to nearby level labels."""
    out = []
    known_buildings = {b.name: b for b in analysis.buildings}
    seen = set()
    for source in planner.get("sources", []):
        pi = int(source.get("page", 0)) - 1
        building = str(source.get("building") or "")
        if building not in known_buildings and len(known_buildings) == 1:
            building = next(iter(known_buildings))
        if pi < 0 or pi >= len(doc) or building not in known_buildings:
            continue
        labels = _level_labels(doc[pi])
        level_map = {}
        for floor in known_buildings[building].floors:
            match = re.search(r"(\d+)", floor.level_id or "")
            if match:
                level_map[int(match.group(1))] = floor.level_id
        for block in doc[pi].get_text("blocks"):
            text = " ".join(str(block[4]).split())
            if re.search(r"\b(SETDOWN|STEP|REBATE|THICKNESS|SLAB)\b", text, re.I):
                continue
            for match in _DATUM_RE.finditer(text):
                raw_value = float(match.group(2))
                unit = (match.group(3) or "").upper()
                value_mm = (raw_value if unit == "MM" or
                            (not unit and abs(raw_value) >= 1000)
                            else raw_value * 1000.0)
                if not -20000 <= value_mm <= 500000:
                    continue
                bx = tuple(block[:4])
                cx, cy = (bx[0]+bx[2])/2, (bx[1]+bx[3])/2
                nearest = None
                for label in labels:
                    lx0, ly0, lx1, ly1 = label["bbox"]
                    lx, ly = (lx0+lx1)/2, (ly0+ly1)/2
                    distance = ((cx-lx)**2 + (cy-ly)**2) ** 0.5
                    if distance <= 120 and (nearest is None or distance < nearest[0]):
                        nearest = (distance, label)
                if nearest is None:
                    continue
                level_num = nearest[1]["level"]
                level_id = level_map.get(level_num)
                if not level_id:
                    continue
                key = (building, level_id, round(value_mm, 1), pi)
                if key in seen:
                    continue
                seen.add(key)
                out.append(HeightEvidence(
                    f"explicit_source_{pi+1:02d}_{level_num:02d}_{len(out)+1}",
                    building, None, level_id, "explicit_datum", value_mm,
                    page_index=pi, bbox=bx, source_text=text,
                    extraction_method="source_view_datum_text",
                    confidence=0.95, is_absolute_datum=True,
                    viewport_id=f"p{pi+1:02d}_absolute"))
    return out


def write_measurement_overlays(doc: fitz.Document, evidence: list[HeightEvidence],
                               out_dir: Path,
                               viewport_reports: list[dict] | None = None) -> None:
    from PIL import Image, ImageDraw
    by_page = {}
    for row in evidence:
        if row.page_index >= 0 and row.datum_line_from and row.datum_line_to:
            by_page.setdefault(row.page_index, []).append(row)
    report_by_page = {}
    for report in viewport_reports or []:
        report_by_page.setdefault(int(report["page"]) - 1, []).append(report)
    for pi, reports in report_by_page.items():
        (out_dir / f"height_viewports_p{pi + 1:02d}.json").write_text(
            json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        factor = 120 / 72.0
        pix = doc[pi].get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        draw = ImageDraw.Draw(image)
        for report in reports:
            x0, y0, x1, y1 = report["bbox"]
            draw.rectangle((x0*factor, y0*factor, x1*factor, y1*factor),
                           outline=(170, 40, 190), width=4)
            draw.text((x0*factor, max(0, y0*factor-15)),
                      report["viewport_id"], fill=(170, 40, 190))
        image.save(out_dir / f"height_viewports_p{pi + 1:02d}.png")
    for pi, rows in by_page.items():
        factor = 150 / 72.0
        pix = doc[pi].get_pixmap(matrix=fitz.Matrix(factor, factor), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        draw = ImageDraw.Draw(image)
        for row in rows:
            for seg, color in ((row.datum_line_from, (20, 100, 230)),
                               (row.datum_line_to, (20, 170, 80))):
                _axis, x1, y1, x2, y2 = seg
                draw.line((x1*factor, y1*factor, x2*factor, y2*factor),
                          fill=color, width=5)
            x, y = row.datum_line_to[1] * factor, row.datum_line_to[2] * factor
            draw.text((x, y), f"{row.id} {row.value_mm:.0f}mm", fill=(180, 20, 20))
        image.save(out_dir / f"height_datum_measurements_p{pi + 1:02d}.png")


def build_consensus(measurements: list[HeightEvidence]) -> tuple[list[HeightEvidence], list[dict]]:
    """Deduplicate views and promote only tight clusters of 3+ independent sources."""
    import numpy as np
    groups = {}
    for row in measurements:
        groups.setdefault((row.building, row.from_level, row.to_level), []).append(row)
    promoted, reports = [], []
    for key, rows in groups.items():
        unique = {}
        for row in sorted(rows, key=lambda x: -x.confidence):
            unique.setdefault(row.independence_group or row.source_fingerprint or row.id, row)
        values = np.array([r.value_mm for r in unique.values()], dtype=float)
        median = float(np.median(values)) if len(values) else 0.0
        mad = float(np.median(np.abs(values - median))) if len(values) else 0.0
        spread = float(values.max() - values.min()) if len(values) else 0.0
        verified = len(values) >= 3 and spread <= 25.0 and mad <= 10.0
        reports.append({"building": key[0], "from_level": key[1],
                        "to_level": key[2], "independent_count": len(values),
                        "median_mm": round(median, 1), "range_mm": round(spread, 1),
                        "mad_mm": round(mad, 1),
                        "status": "verified_consensus" if verified else "inferred"})
        if verified:
            ids = [r.id for r in unique.values()]
            promoted.append(HeightEvidence(
                f"consensus_{len(promoted)+1:03d}", key[0], key[1], key[2],
                "verified_consensus", round(median, 1),
                source_text=f"Median of independent evidence {ids}",
                extraction_method="independent_measurement_consensus",
                confidence=0.95, is_absolute_datum=False,
                independence_group="|".join(sorted(unique))))
        else:
            promoted.extend(unique.values())
    return promoted, reports
