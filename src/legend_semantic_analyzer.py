"""
Gemini semantic classification for detected legend crops.

Input geometry is still deterministic: legend_locator finds the crop, this module
extracts indexed text items and asks Gemini which legend entries are useful for
wall/slab/material detection.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

from src.document_intelligence import _load_gemini_client, _strip_json_fence
from src.legend_locator import locate_legends_for_pages


PROMPT_LEGEND_SEMANTICS = r"""
You are a principal structural engineer and construction-document analyst.
You will receive indexed text items extracted from one or more detected legend crops.

Your job:
- Decide which indexed items describe wall evidence, slab evidence, concrete/material evidence, void/cut evidence, column/foundation evidence, or ignore/noise.
- Tell the downstream geometry code which legend classes are useful for detecting/drawing walls and slabs.
- For slabs, separate surface/material evidence, boundary cues, and true net-slab cut items.
- Do NOT invent geometry or coordinates.
- Prefer conservative classifications. Notes/title/client/keyplan/revision text should be ignored.

Definitions:
- wall_evidence: concrete wall, RC wall, loadbearing concrete/masonry wall, core filled blockwork, precast wall/column, load-bearing element under/over if usable to identify wall/core linework.
- slab_evidence: slab fill/material, floor structure, SOG, slab on grade, slab construction/movement/temporary joint, slab setdown/step/thickness, slab penetration. Some are not slab boundaries but help recognize slab annotations.
- concrete_material: concrete, reinforced/core filled blockwork, floor structure fill, load-bearing concrete element, material hatch/fill.
- void_or_cut: slab penetration, stair/lift/core/opening/void/shaft, non-slab zones that may cut net slab.
- column_or_foundation: concrete column, steel column, pile cap, footing, pad footing, pile, PC/PF/F symbols.
- ignore: title block, keyplan, drawing number, revision, tender notes, generic references, drawing notes not a legend rule.

Return ONLY valid compact JSON:
{
  "legend_semantic_status": "ok"|"uncertain",
  "wall_detection_items": [
    {
      "index": number,
      "label": string,
      "class": "wall_evidence"|"concrete_material",
      "use_for": ["wall_detector","boundary_snap","material_mask"],
      "confidence": number,
      "reason": string
    }
  ],
  "slab_detection_items": [
    {
      "index": number,
      "label": string,
      "class": "slab_evidence"|"concrete_material"|"void_or_cut",
      "use_for": ["slab_fill","slab_boundary","net_slab_cut","ignore_for_gross_slab"],
      "confidence": number,
      "reason": string
    }
  ],
  "slab_surface_items": [
    {
      "index": number,
      "label": string,
      "surface_type": "slab_fill"|"floor_structure"|"concrete_material"|"sog"|"white_no_fill"|"unknown",
      "use_for": ["slab_surface","gross_slab_candidate","material_mask"],
      "confidence": number,
      "reason": string
    }
  ],
  "slab_boundary_items": [
    {
      "index": number,
      "label": string,
      "cue_type": "joint"|"setdown"|"step"|"edge"|"thickness"|"annotation",
      "use_for": ["boundary_snap","region_scoring","review_only"],
      "auto_cut_gross_slab": false,
      "confidence": number,
      "reason": string
    }
  ],
  "slab_cut_items": [
    {
      "index": number,
      "label": string,
      "cut_type": "penetration"|"opening"|"void"|"pipe_penetration"|"shaft"|"unknown",
      "cut_policy": "auto_cut_high_confidence"|"review_before_cut"|"do_not_cut",
      "confidence": number,
      "reason": string
    }
  ],
  "material_items": [
    {
      "index": number,
      "label": string,
      "material": "concrete"|"steelwork"|"masonry"|"existing"|"unknown",
      "confidence": number
    }
  ],
  "void_cut_items": [
    {
      "index": number,
      "label": string,
      "cut_policy": "auto_cut_high_confidence"|"review_before_cut"|"do_not_cut",
      "confidence": number,
      "reason": string
    }
  ],
  "ignored_items": [
    {"index": number, "label": string, "reason": string}
  ],
  "rules_for_code": {
    "wall_keywords": [string],
    "wall_hatch_or_line_styles": [string],
    "slab_fill_keywords": [string],
    "slab_surface_keywords": [string],
    "slab_boundary_keywords": [string],
    "net_slab_cut_keywords": [string],
    "never_cut_keywords": [string],
    "fallback_policy": "use_semantic_fill"|"use_evidence_guided_no_fill_boundary"|"review_manually",
    "notes": string
  },
  "warnings": [string]
}
"""


def _rect_contains_or_intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    overlap = fitz.Rect(a)
    overlap.intersect(fitz.Rect(b))
    return not overlap.is_empty


def extract_indexed_legend_items(page: fitz.Page, bbox: list[float]) -> list[dict[str, Any]]:
    rect = fitz.Rect(*bbox)
    items = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        block_rect = fitz.Rect(block.get("bbox"))
        if not _rect_contains_or_intersects(rect, block_rect):
            continue
        texts = []
        for line in block.get("lines", []):
            spans = [s.get("text", "").strip() for s in line.get("spans", []) if s.get("text", "").strip()]
            if spans:
                texts.append(" ".join(spans))
        text = " ".join(texts).strip()
        if not text:
            continue
        items.append({
            "bbox": [float(v) for v in block_rect],
            "text": text,
        })
    items.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    indexed = []
    for i, item in enumerate(items):
        indexed.append({
            "index": i,
            "text": item["text"],
            "bbox": item["bbox"],
        })
    return indexed


def _build_prompt_payload(pdf_path: str, page_items: list[dict[str, Any]]) -> str:
    lines = [f"PDF: {Path(pdf_path).name}", "INDEXED_LEGEND_TEXT_ITEMS:"]
    for page in page_items:
        lines.append(f"\n--- PAGE {page['page']} | side={page['side']} | confidence={page['confidence']} ---")
        for item in page["items"]:
            lines.append(f"[{item['global_index']}] {item['text']}")
    return "\n".join(lines)


def _parse_legend_json(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cleaned = _strip_json_fence(raw or "")
    report = {
        "parse_status": "ok",
        "parse_error": None,
        "raw_response_length": len(raw or ""),
        "cleaned_response_length": len(cleaned or ""),
        "response_ended_cleanly": cleaned.rstrip().endswith(("}", "]")),
    }
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        report["parse_status"] = "invalid_json"
        report["parse_error"] = str(exc)
        return {"_parse_error": str(exc), "_raw_preview": (raw or "")[:2000]}, report
    if not isinstance(parsed, dict) or "legend_semantic_status" not in parsed:
        report["parse_status"] = "schema_empty"
        report["parse_error"] = "Valid JSON parsed, but legend semantic schema is missing."
    return parsed, report


def analyze_legend_semantics(
    pdf_path: str,
    page_indices: list[int],
    output_dir: str | Path,
) -> tuple[dict[str, Any], str, str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    located = locate_legends_for_pages(pdf_path, page_indices, dpi=144)
    doc = fitz.open(pdf_path)
    page_items: list[dict[str, Any]] = []
    global_index = 0
    try:
        for cand in located.get("candidates", []):
            page_index = int(cand["page_index"])
            items = extract_indexed_legend_items(doc[page_index], cand["bbox"])
            for item in items:
                item["local_index"] = item.pop("index")
                item["global_index"] = global_index
                global_index += 1
            page_items.append({
                "page_index": page_index,
                "page": page_index + 1,
                "side": cand.get("side"),
                "confidence": cand.get("confidence"),
                "bbox": cand.get("bbox"),
                "items": items,
            })
    finally:
        doc.close()

    client, model = _load_gemini_client()
    payload = _build_prompt_payload(pdf_path, page_items)
    response = client.models.generate_content(
        model=model,
        contents=[PROMPT_LEGEND_SEMANTICS, payload],
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"legend_semantics_{ts}_raw.txt"
    json_path = output_dir / f"legend_semantics_{ts}.json"
    report_path = output_dir / f"legend_semantics_{ts}_parse_report.json"
    input_path = output_dir / f"legend_semantics_{ts}_input_items.json"

    raw_text = (response.text or "").strip()
    raw_path.write_text(raw_text, encoding="utf-8")
    parsed, report = _parse_legend_json(raw_text)
    report.update({
        "raw_response_path": str(raw_path),
        "parsed_json_path": str(json_path),
        "parse_report_path": str(report_path),
        "input_items_path": str(input_path),
        "page_count": len(page_items),
        "item_count": sum(len(p["items"]) for p in page_items),
    })
    result = {
        "_metadata": report,
        "legend_detection": located,
        "input_items": page_items,
        "gemini_result": parsed,
    }
    input_path.write_text(json.dumps(page_items, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, str(json_path), str(raw_path), str(report_path)
