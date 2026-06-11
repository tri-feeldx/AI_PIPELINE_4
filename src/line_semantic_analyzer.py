"""
Line semantic intelligence for structural PDFs.

This module builds a compact catalog of vector line styles and asks Gemini what
those styles mean in the current drawing context. Geometry remains deterministic:
Gemini classifies style intent; downstream code decides polygon construction.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

from src.document_intelligence import _load_gemini_client, _strip_json_fence


PROMPT_LINE_SEMANTICS = r"""
You are a principal structural engineer reading structural PDF drawings.
You will receive a compact catalog of vector line styles from plan pages,
nearby extracted text, and already-extracted legend semantic rules.

Your task is semantic classification only. Do not invent coordinates. Do not
draw polygons. Classify what each line style likely means for geometry code.

Return ONLY valid compact JSON:
{
  "line_semantic_status": "ok"|"uncertain",
  "style_rules": [
    {
      "style_id": string,
      "semantic": "building_boundary"|"slab_edge"|"wall"|"site_boundary"|"grid"|"dimension"|"annotation"|"joint"|"reference"|"unknown",
      "use_for": ["slab_enclosure"|"wall_detection"|"context_only"|"ignore"|"review"],
      "draw_policy": "inside"|"outside"|"do_not_draw_from_this"|"snap_only"|"review",
      "confidence": number,
      "reason": string
    }
  ],
  "warnings": [string]
}

Rules:
- building_boundary and slab_edge may support no-fill slab enclosure.
- wall supports wall/core evidence but should not alone define slab inside/outside.
- site_boundary, grid, dimension, annotation, and generic reference lines must not create slab enclosure.
- joint lines are usually snap/review only and should not cut gross slab unless explicitly penetration/opening.
- If unsure, use semantic unknown with use_for ["review"].
"""


def _color_bucket(color) -> str:
    if color is None or len(color) < 3:
        return "none"
    r, g, b = [float(v) for v in color[:3]]
    return f"{round(r, 1):.1f},{round(g, 1):.1f},{round(b, 1):.1f}"


def _style_key(d: dict) -> str:
    width = float(d.get("width") or 0)
    width_bucket = round(width * 2) / 2.0
    dashes = d.get("dashes")
    dashed = bool(dashes and str(dashes).strip() not in ("[]", "()", "None", ""))
    return f"c={_color_bucket(d.get('color'))}|w={width_bucket:.1f}|dash={int(dashed)}"


def _segment(item):
    if item[0] != "l":
        return None
    p1 = (float(item[1].x), float(item[1].y))
    p2 = (float(item[2].x), float(item[2].y))
    if math.dist(p1, p2) < 3:
        return None
    return p1, p2


def _orientation(p1, p2) -> str:
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if dx <= 2:
        return "vertical"
    if dy <= 2:
        return "horizontal"
    return "diagonal"


def _line_bbox(p1, p2) -> list[float]:
    return [min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])]


def _nearby_text(page: fitz.Page, bbox: list[float], radius: float = 70.0, limit: int = 8) -> list[str]:
    base = fitz.Rect(bbox)
    rect = fitz.Rect(base.x0 - radius, base.y0 - radius, base.x1 + radius, base.y1 + radius)
    hits: list[str] = []
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", []):
        if block.get("type") != 0:
            continue
        b = fitz.Rect(block.get("bbox"))
        if not rect.intersects(b):
            continue
        parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    parts.append(t)
        text = " ".join(parts).strip()
        if text and text not in hits:
            hits.append(text[:140])
        if len(hits) >= limit:
            break
    return hits


def _page_context_text(page: fitz.Page, limit: int = 50) -> list[str]:
    keywords = (
        "WALL", "BOUNDARY", "SLAB", "S.O.G", "SOG", "BUILDING", "PROPERTY",
        "SITE", "GRID", "JOINT", "P.M.J", "C.J", "REFER", "DRAWING",
    )
    lines = []
    for raw in page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE).splitlines():
        text = " ".join(raw.split())
        if not text:
            continue
        upper = text.upper()
        if any(k in upper for k in keywords):
            lines.append(text[:180])
        if len(lines) >= limit:
            break
    return lines


def build_line_catalog_for_pages(pdf_path: str, page_indices: list[int], max_styles_per_page: int = 12) -> dict[str, Any]:
    """Build a compact style catalog for Gemini; do not include every raw segment."""
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= doc.page_count:
                continue
            page = doc[page_index]
            page_area = max(page.rect.width * page.rect.height, 1.0)
            groups: dict[str, dict] = {}
            for d in page.get_drawings():
                style_id = _style_key(d)
                width = float(d.get("width") or 0)
                dashes = d.get("dashes")
                dashed = bool(dashes and str(dashes).strip() not in ("[]", "()", "None", ""))
                for item in d.get("items", []):
                    seg = _segment(item)
                    if seg is None:
                        continue
                    p1, p2 = seg
                    length = math.dist(p1, p2)
                    if length < 8:
                        continue
                    g = groups.setdefault(style_id, {
                        "style_id": style_id,
                        "color_bucket": _color_bucket(d.get("color")),
                        "width": round(width, 3),
                        "dashed": dashed,
                        "segment_count": 0,
                        "total_length": 0.0,
                        "long_segment_count": 0,
                        "orientations": {"horizontal": 0, "vertical": 0, "diagonal": 0},
                        "sample_bboxes": [],
                        "nearby_text": [],
                    })
                    g["segment_count"] += 1
                    g["total_length"] += length
                    if length >= math.sqrt(page_area) * 0.045:
                        g["long_segment_count"] += 1
                    g["orientations"][_orientation(p1, p2)] += 1
                    if len(g["sample_bboxes"]) < 5:
                        bbox = _line_bbox(p1, p2)
                        g["sample_bboxes"].append([round(v, 2) for v in bbox])
                        for txt in _nearby_text(page, bbox, radius=55, limit=4):
                            if txt not in g["nearby_text"] and len(g["nearby_text"]) < 10:
                                g["nearby_text"].append(txt)
            styles = sorted(
                groups.values(),
                key=lambda x: (x["long_segment_count"], x["total_length"], x["segment_count"]),
                reverse=True,
            )[:max_styles_per_page]
            for s in styles:
                s["total_length"] = round(s["total_length"], 2)
            pages.append({
                "page": page_index + 1,
                "page_index": page_index,
                "context_text": _page_context_text(page),
                "line_styles": styles,
            })
    finally:
        doc.close()
    return {"pdf": Path(pdf_path).name, "pages": pages}


def _parse_json_response(raw: str) -> tuple[dict, dict]:
    cleaned = _strip_json_fence(raw)
    report = {
        "parse_status": "ok",
        "parse_error": None,
        "raw_response_length": len(raw or ""),
        "cleaned_response_length": len(cleaned or ""),
    }
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or "style_rules" not in parsed:
            report["parse_status"] = "schema_empty"
            report["parse_error"] = "Valid JSON parsed, but style_rules missing."
        return parsed, report
    except json.JSONDecodeError as exc:
        report["parse_status"] = "invalid_json"
        report["parse_error"] = f"{exc.msg} at line {exc.lineno} column {exc.colno}"
    return {"line_semantic_status": "invalid", "style_rules": [], "warnings": [report["parse_error"]]}, report


def analyze_line_semantics(
    pdf_path: str,
    page_indices: list[int],
    output_dir: str | Path,
    legend_semantics: dict | None = None,
) -> tuple[dict, str, str, str, str]:
    catalog = build_line_catalog_for_pages(pdf_path, page_indices)
    compact_legend = {
        "rules_for_code": (legend_semantics or {}).get("rules_for_code", {}),
        "wall_detection_items": (legend_semantics or {}).get("wall_detection_items", [])[:20],
        "slab_boundary_items": (legend_semantics or {}).get("slab_boundary_items", [])[:20],
    }
    prompt = (
        PROMPT_LINE_SEMANTICS
        + "\n\nLEGEND_SEMANTICS_JSON:\n"
        + json.dumps(compact_legend, ensure_ascii=False)
        + "\n\nLINE_CATALOG_JSON:\n"
        + json.dumps(catalog, ensure_ascii=False)
    )
    client, model = _load_gemini_client()
    response = client.models.generate_content(model=model, contents=prompt)
    raw = response.text or ""
    parsed, report = _parse_json_response(raw)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"line_semantics_{ts}_raw.txt"
    json_path = output_dir / f"line_semantics_{ts}.json"
    report_path = output_dir / f"line_semantics_{ts}_parse_report.json"
    catalog_path = output_dir / f"line_semantics_{ts}_catalog.json"
    metadata = {
        **report,
        "raw_response_path": str(raw_path),
        "parsed_json_path": str(json_path),
        "parse_report_path": str(report_path),
        "catalog_path": str(catalog_path),
        "page_count": len(page_indices),
    }
    result = {
        "_metadata": metadata,
        "line_catalog": catalog,
        "gemini_result": parsed,
    }
    raw_path.write_text(raw, encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, str(json_path), str(raw_path), str(report_path), str(catalog_path)
