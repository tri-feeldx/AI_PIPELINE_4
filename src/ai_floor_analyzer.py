"""
AI-powered floor structure extraction using Google Vertex AI + Gemini.

Reads ALL page text from selected PDF pages, asks Gemini to identify:
  - Buildings in the project
  - Floor levels per building (with FFL values)
  - Which pages contain the slab plan for each floor

Returns structured JSON saved to debug_ai/ folder for user review.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import fitz
from src.pdf_processor import extract_text_blocks


# ── Gemini client (lazy-initialised) ──────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    from dotenv import load_dotenv
    load_dotenv()

    from google import genai
    from google.oauth2 import service_account

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project    = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location   = os.environ.get("VERTEX_LOCATION", "us-central1")

    if not creds_path or not project:
        raise EnvironmentError(
            "GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT "
            "must be set in .env"
        )

    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    _client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=creds,
    )
    return _client


def _get_model_name() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


# ── Text extraction ────────────────────────────────────────────────────────────

MAX_CHARS_PER_PAGE = 1000   # ~250 tokens; keeps total prompt < 200K tokens for 200 pages


def extract_pdf_text_for_ai(pdf_path: str, page_indices: list) -> str:
    """
    Extract text from selected pages, formatted for Gemini prompt.
    One line per page: "[Page N]: text..."
    """
    doc = fitz.open(pdf_path)
    parts = []
    try:
        for idx in page_indices:
            if idx >= doc.page_count:
                continue
            page = doc[idx]
            blocks = extract_text_blocks(page)
            page_text = " | ".join(
                b["text"].strip() for b in blocks if b["text"].strip()
            )
            page_text = page_text[:MAX_CHARS_PER_PAGE]
            parts.append(f"[Page {idx + 1}]: {page_text}")
    finally:
        doc.close()
    return "\n".join(parts)


# ── Gemini prompt ──────────────────────────────────────────────────────────────

_PROMPT = """\
You are analyzing a structural engineering PDF (Australian or international construction drawings).

EXTRACTED TEXT — one line per page:
{page_texts}

TASK:
Identify the unique floor levels that have SLAB PLANS or FLOOR PLANS.
Ignore pages that are: sections, elevations, retention plans, detail sheets, schedules, \
general notes, cover sheets, or steelwork details.

Return ONLY valid JSON (no explanation, no markdown, just the JSON object):
{{
  "buildings": [
    {{
      "name": "Building A",
      "floors": [
        {{
          "level_name": "Level 1",
          "level_id": "level_1",
          "ffl_m": 44.000,
          "slab_plan_pages": [7, 8, 9],
          "page_titles": ["LEVEL 01 OUTLINE PLAN - 200PT SLAB", "LEVEL 01 PART PLAN ZONE A", "LEVEL 01 PART PLAN ZONE B"]
        }}
      ]
    }}
  ],
  "total_unique_floors": 5,
  "detection_confidence": "high",
  "notes": "brief summary of what was found"
}}

RULES:
- level_id must be lowercase snake_case: "ground", "level_1", "level_2", "mezzanine", \
"podium", "lower_roof", "upper_roof", "roof", "basement", "carpark"
- slab_plan_pages: 1-indexed page numbers for slab/floor PLAN pages only.
  Include ALL zone/area/part variants of the same floor (Zone A, Zone B, Part Plan, etc.) \
  — they must ALL be processed to get complete slab geometry.
- page_titles: For EACH page number in slab_plan_pages (same order), copy the EXACT drawing \
  title text found on THAT page (e.g. "LEVEL 02 OUTLINE PLAN - 200 POST TENSIONED SLAB U.N.O"). \
  Read the title from the page content itself, NOT from the drawing index or table of contents. \
  This field is mandatory and will be used to verify your floor assignments.
- ffl_m: REQUIRED — provide a numeric value in metres for EVERY floor. Never leave null.
  Search ALL pages (not just plan pages) for elevation data: FFL, RL, EL, FL, AHD, NGL, \
  "finished floor level", "reduced level", floor schedules, section drawings, key plans.
  If value is in mm (e.g. 44000), convert to m (44.000). \
  If the datum is unclear but storey heights are shown, accumulate from ground level.
  FALLBACK (only if truly no elevation data anywhere in the document): \
  estimate using 0.000m for Ground/Level 1, +3.500m per floor above, -3.500m per basement.
  Use null ONLY as absolute last resort when nothing at all can be inferred.
- detection_confidence: "high" (clear keywords), "medium" (inferred), or "low" (guessed).
- If no clear floor plans are found, set buildings to [] and total_unique_floors to 0.
- If no building separation is evident, use a single building named after the project.
"""


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from Gemini response. Handles markdown code fences."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    json_str = m.group(1) if m else text.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        m2 = re.search(r"\{[\s\S]*\}", json_str)
        if m2:
            try:
                return json.loads(m2.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ── Page-title validation ──────────────────────────────────────────────────────

def _validate_page_titles(parsed: dict) -> dict:
    """
    Cross-check page_titles against level assignments.
    If a page title says "LEVEL 02" but the floor is level_1, move that page
    to the correct floor (level_2). Logs any corrections made.
    """
    import logging
    logger = logging.getLogger(__name__)

    for bld in parsed.get("buildings", []):
        floors = bld.get("floors", [])

        # Build lookup: level_number (int) → floor dict
        num_to_floor: dict = {}
        for fl in floors:
            m = re.search(r"(\d+)", fl.get("level_id", ""))
            if m:
                num_to_floor[int(m.group(1))] = fl

        for fl in floors:
            fl_m = re.search(r"(\d+)", fl.get("level_id", ""))
            if not fl_m:
                continue
            fl_num = int(fl_m.group(1))

            pages = fl.get("slab_plan_pages", [])
            titles = fl.get("page_titles", [])
            # Pad titles list if Gemini returned fewer entries than pages
            titles += [""] * max(0, len(pages) - len(titles))

            keep_pages, keep_titles = [], []
            for pg, title in zip(pages, titles):
                m = re.search(r"LEVEL\s+0*(\d+)", str(title).upper())
                if m:
                    title_num = int(m.group(1))
                    if title_num != fl_num:
                        correct = num_to_floor.get(title_num)
                        if correct and correct is not fl:
                            logger.warning(
                                f"Page {pg} title='{title}' → moved from "
                                f"{fl.get('level_id')} to {correct.get('level_id')}"
                            )
                            correct.setdefault("slab_plan_pages", []).append(pg)
                            correct.setdefault("page_titles", []).append(title)
                            continue  # don't add to wrong floor
                keep_pages.append(pg)
                keep_titles.append(title)

            fl["slab_plan_pages"] = keep_pages
            fl["page_titles"] = keep_titles

    return parsed


# ── Main entry point ───────────────────────────────────────────────────────────

def analyze_floor_structure(
    pdf_path: str,
    page_indices: list,
    save_dir: Optional[str] = None,
) -> Tuple[dict, str]:
    """
    Call Gemini to analyze the floor structure of the PDF.

    Args:
        pdf_path:     path to the PDF file
        page_indices: 0-indexed page indices to scan
        save_dir:     directory to save the raw Gemini output JSON (default: pdf folder/debug_ai)

    Returns:
        (result_dict, saved_file_path)

        result_dict keys:
          buildings          — list of {name, floors: [{level_name, level_id, ffl_m, slab_plan_pages}]}
          total_unique_floors
          detection_confidence
          pages_to_process   — 0-indexed list of all slab plan pages (added by this function)
          notes
    """
    # 1. Extract page texts
    page_texts = extract_pdf_text_for_ai(pdf_path, page_indices)

    # 2. Build prompt and call Gemini
    prompt = _PROMPT.format(page_texts=page_texts)
    client = _get_client()
    model  = _get_model_name()
    response = client.models.generate_content(model=model, contents=prompt)
    raw_text = response.text

    # 3. Parse JSON
    parsed = _parse_json_response(raw_text)
    if parsed is None:
        raise ValueError(
            f"Gemini returned unparseable JSON.\nFirst 300 chars:\n{raw_text[:300]}"
        )

    # 3b. Validate page_titles — auto-correct mis-assigned pages
    parsed = _validate_page_titles(parsed)

    # 4. Flatten all slab_plan_pages → 0-indexed pages_to_process
    all_1idx: set = set()
    for bld in parsed.get("buildings", []):
        for floor in bld.get("floors", []):
            for p in floor.get("slab_plan_pages", []):
                if isinstance(p, int) and p >= 1:
                    all_1idx.add(p)
    # Convert 1-indexed page numbers → 0-indexed, validate against page_indices
    max_page = max(page_indices) + 1 if page_indices else 0
    parsed["pages_to_process"] = sorted(
        p - 1 for p in all_1idx if 1 <= p <= max_page
    )

    # 5. Save raw output for user review
    if save_dir is None:
        save_dir = str(Path(pdf_path).parent / "debug_ai")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(save_dir) / f"gemini_floors_{ts}.json"
    out_file.write_text(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "pages_scanned": len(page_indices),
                "raw_gemini_response": raw_text,
                "parsed_result": parsed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return parsed, str(out_file)
