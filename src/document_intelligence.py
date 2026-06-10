"""
Full-PDF document intelligence for structural drawings.

Gemini reads extracted text once per uploaded PDF and returns semantic metadata:
column symbols, foundation symbols, height evidence, schedule/legend pages, and floor summaries.
Geometry is still handled by deterministic vector detectors.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz


PROMPT_DOCUMENT_INTELLIGENCE = r"""
You are a principal structural engineer and construction-document analyst.
You will receive FULL extracted text from every page of one structural PDF.
Your job is DOCUMENT INTELLIGENCE, not geometry. Do not invent coordinates.

Return ONLY valid JSON. No markdown, no explanation. Keep the response compact.

Critical reading rules:
- Read legends, schedules, plan notes, title blocks, and slab/floor plan text.
- Column symbols may be graphical/textual, not only schedule rows.
- A concrete column may appear as a circle containing C over a number. Normalize it as C<number>:
  C over 5 => C5, C over 10 => C10, C over 15 => C15.
- A dashed circle C/number means concrete_column_under_only.
- Text OVER near C/number means concrete_column_over_only.
- CH/SH/UC/UB style symbols are steel column/steel member symbols only if the legend/schedule supports that.
- Do NOT omit a symbol just because width/depth is unknown. Use null for unknown dimensions.
- Separate true floor/column instances from legend symbols and detail examples.
- Counts may be inferred from plan text occurrences, but mark confidence accordingly.
- Foundation/footing symbols may appear in footing schedules or footing plan legends; include them if found.
- Foundation symbols can be PF1, F1, PC1, P1, R1, SF1, pad footing, pile cap, raft, strip footing, or similar.
- If foundation schedule pages are found but individual symbols are unclear, still return the schedule pages and warnings.
- Storey heights must be evidence-based. Extract FFL/RL/EL/FL/AHD/NGL values, explicit "floor-to-floor"/"storey height"
  text, and elevation/section pages with LEVEL/ROOF labels. Do not invent heights.
- If a page appears useful for measuring heights but lacks explicit numeric elevations, return it as an elevation/section source
  with recommended_action="measure_level_spacing_from_drawing".
- Avoid generating duplicate symbol variants that only repeat the same schedule information. Prefer compact canonical symbols
  and put detailed ambiguity in warnings. If output would be too large, prioritize buildings/floors/page mapping,
  schedule pages, column/foundation symbol families, and warnings.

Required JSON schema:
{
  "document_summary": {
    "project_name": string|null,
    "page_count": number,
    "detection_confidence": "high"|"medium"|"low",
    "notes": string
  },
  "legend_rules": {
    "columns": [
      {
        "symbol_pattern": string,
        "normalized_family": string,
        "meaning": string,
        "status": "normal"|"under_only"|"over_only"|"unknown",
        "source_pages": [number]
      }
    ],
    "foundations": [
      {
        "symbol_pattern": string,
        "normalized_family": string,
        "meaning": string,
        "source_pages": [number]
      }
    ]
  },
  "column_symbols": {
    "C10": {
      "family": "concrete_column"|"steel_column"|"unknown_column",
      "status": "normal"|"under_only"|"over_only"|"unknown",
      "width_mm": number|null,
      "depth_mm": number|null,
      "count_total": number|null,
      "source": "legend"|"schedule"|"plan_text"|"inferred",
      "source_pages": [number]
    }
  },
  "foundation_symbols": {
    "PF1": {
      "type": "pad"|"pile_cap"|"raft"|"strip"|"unknown",
      "width_mm": number|null,
      "depth_mm": number|null,
      "thickness_mm": number|null,
      "depth_below_gl_mm": number|null,
      "pile_count": number|null,
      "pile_diameter_mm": number|null,
      "count_total": number|null,
      "source": "legend"|"schedule"|"plan_text"|"inferred",
      "source_pages": [number]
    }
  },
  "buildings": [
    {
      "name": string,
      "floors": [
        {
          "level_name": string,
          "slab_plan_pages": [number],
          "column_summary": {
            "total_columns": number|null,
            "by_symbol": {"C10": number}
          },
          "foundation_summary": {
            "total_foundations": number|null,
            "by_symbol": {"PF1": number}
          }
        }
      ]
    }
  ],
  "schedule_pages": {
    "column_schedule_pages": [number],
    "foundation_schedule_pages": [number],
    "footing_plan_pages": [number],
    "legend_pages": [number],
    "detail_pages": [number]
  },
  "height_sources": [
    {
      "type": "plan_level_text"|"elevation_page"|"section_page"|"schedule"|"note"|"unknown",
      "page": number,
      "level": string|null,
      "elevation_m": number|null,
      "height_mm": number|null,
      "source_text": string,
      "has_numeric_levels": boolean,
      "recommended_action": "use_explicit_value"|"measure_level_spacing_from_drawing"|"review_manually",
      "confidence": number
    }
  ],
  "storey_heights": [
    {
      "from_level": string,
      "to_level": string,
      "height_mm": number|null,
      "source": "explicit_text"|"schedule"|"elevation_text"|"inferred"|"unknown",
      "source_pages": [number],
      "confidence": number
    }
  ],
  "warnings": [string]
}
"""


def extract_full_pdf_text(pdf_path: str) -> tuple[str, int]:
    """Extract all PDF text with stable page markers for Gemini."""
    doc = fitz.open(pdf_path)
    parts = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE).strip()
            parts.append(f"\n=== PAGE {i + 1} ===\n{text}")
        return "\n".join(parts), doc.page_count
    finally:
        doc.close()


def _load_gemini_client():
    from dotenv import load_dotenv
    from google import genai
    from google.oauth2 import service_account

    load_dotenv(Path(".env"))
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    if not creds_path or not project:
        raise EnvironmentError("Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT in .env")

    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(vertexai=True, project=project, location=location, credentials=creds)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return client, model


def _strip_json_fence(raw: str) -> str:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned, count=1).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def _looks_truncated(raw: str, cleaned: str) -> bool:
    text = (cleaned or "").rstrip()
    if not text:
        return True
    if (raw or "").strip().startswith("```") and not (raw or "").strip().endswith("```"):
        return True
    if not text.endswith(("}", "]")):
        return True
    return text.count("{") != text.count("}") or text.count("[") != text.count("]")


def _parse_json_response(raw: str) -> tuple[dict, dict]:
    cleaned = _strip_json_fence(raw)
    report = {
        "parse_status": "ok",
        "parse_error": None,
        "raw_response_length": len(raw or ""),
        "cleaned_response_length": len(cleaned or ""),
        "response_ended_cleanly": not _looks_truncated(raw or "", cleaned),
    }
    try:
        parsed = json.loads(cleaned)
        if not any(parsed.get(k) for k in ("buildings", "column_symbols", "foundation_symbols", "schedule_pages", "height_sources")):
            report["parse_status"] = "schema_empty"
            report["parse_error"] = "Valid JSON parsed, but semantic schema is empty."
        return parsed, report
    except json.JSONDecodeError as exc:
        report["parse_status"] = "truncated" if _looks_truncated(raw or "", cleaned) else "invalid_json"
        report["parse_error"] = f"{exc.msg} at line {exc.lineno} column {exc.colno}"
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                parsed = json.loads(m.group(0))
                report["parse_status"] = "ok"
                report["parse_error"] = None
                return parsed, report
            except json.JSONDecodeError as inner_exc:
                report["parse_error"] = (
                    f"{report['parse_error']}; object-slice parse failed: "
                    f"{inner_exc.msg} at line {inner_exc.lineno} column {inner_exc.colno}"
                )
                pass
    return {
        "_parse_error": "Gemini response was not valid JSON",
        "_parse_status": report["parse_status"],
    }, report


def normalize_document_intelligence(raw: dict, page_count: int = 0, parse_report: dict | None = None) -> dict:
    """Ensure expected top-level keys exist."""
    raw = raw or {}
    parse_report = parse_report or {"parse_status": "ok"}
    if raw.get("_parse_error"):
        summary = {
            "project_name": None,
            "page_count": page_count,
            "detection_confidence": "low",
            "notes": raw.get("_parse_error"),
        }
        return {
            "_parse_error": raw.get("_parse_error"),
            "_parse_status": parse_report.get("parse_status", "invalid_json"),
            "_metadata": dict(parse_report),
            "document_summary": summary,
            "legend_rules": {"columns": [], "foundations": []},
            "column_symbols": {},
            "foundation_symbols": {},
            "buildings": [],
            "schedule_pages": {
                "column_schedule_pages": [],
                "foundation_schedule_pages": [],
                "footing_plan_pages": [],
                "legend_pages": [],
                "detail_pages": [],
            },
            "height_sources": [],
            "storey_heights": [],
            "warnings": [raw.get("_parse_error")],
        }
    summary = raw.get("document_summary") or {}
    summary.setdefault("project_name", None)
    summary.setdefault("page_count", page_count)
    summary.setdefault("detection_confidence", "low")
    summary.setdefault("notes", "")
    schedule_pages = raw.get("schedule_pages") or {}
    for key in ("column_schedule_pages", "foundation_schedule_pages", "footing_plan_pages", "legend_pages", "detail_pages"):
        schedule_pages.setdefault(key, [])
    parsed = {
        "_parse_status": parse_report.get("parse_status", "ok"),
        "_metadata": dict(parse_report),
        "document_summary": summary,
        "legend_rules": raw.get("legend_rules") or {"columns": [], "foundations": []},
        "column_symbols": raw.get("column_symbols") or {},
        "foundation_symbols": raw.get("foundation_symbols") or {},
        "buildings": raw.get("buildings") or [],
        "schedule_pages": schedule_pages,
        "height_sources": raw.get("height_sources") or [],
        "storey_heights": raw.get("storey_heights") or [],
        "warnings": raw.get("warnings") or [],
    }
    if parse_report.get("parse_status") != "ok":
        parsed.setdefault("warnings", []).append(parse_report.get("parse_error") or parse_report.get("parse_status"))
    return parsed


def analyze_document_intelligence(pdf_path: str, output_dir: str | Path) -> tuple[dict, str, Optional[str]]:
    """Call Gemini once on full PDF text and cache parsed/raw output files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_text, page_count = extract_full_pdf_text(pdf_path)
    client, model = _load_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=[
            PROMPT_DOCUMENT_INTELLIGENCE,
            f"PDF page_count={page_count}\nFULL_TEXT:\n{full_text}",
        ],
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"document_intelligence_{ts}.json"
    raw_path = output_dir / f"document_intelligence_{ts}_raw.txt"
    report_path = output_dir / f"document_intelligence_{ts}_parse_report.json"
    raw_text = (response.text or "").strip()
    raw_path.write_text(raw_text, encoding="utf-8")
    raw_parsed, parse_report = _parse_json_response(raw_text)
    parse_report.update({
        "raw_response_path": str(raw_path),
        "parsed_json_path": str(json_path),
        "parse_report_path": str(report_path),
    })
    parsed = normalize_document_intelligence(raw_parsed, page_count=page_count, parse_report=parse_report)
    parsed["_metadata"].update(parse_report)
    report_path.write_text(json.dumps(parse_report, indent=2, ensure_ascii=False), encoding="utf-8")
    json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    return parsed, str(json_path), str(raw_path)
