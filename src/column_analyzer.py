"""
Column & Foundation Census — Gemini text-only scan.

Reads column schedules, foundation schedules, and per-page column counts
from the PDF text, using Gemini to parse and structure the information.

Does NOT use Vision API — text extraction only.

Public API:
    analyze_columns_and_foundations(pdf_path, page_indices, ai_floor_result) -> dict
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import fitz


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_CENSUS = """You are a structural engineer reading a set of structural PDF drawing pages.

I will give you the extracted text from multiple pages of a structural drawing set.
Your job is to identify ALL columns and foundations described in this drawing set.

━━━ TASK 1 — COLUMN SCHEDULE ━━━
Find the column schedule table(s) in the text. Extract:
- Each unique column symbol (e.g. "SH", "PG1", "C1", "RC1")
- Its cross-section dimensions in mm (width × depth)
- Total count if stated

━━━ TASK 2 — COLUMN COUNT PER PAGE / FLOOR / BUILDING ━━━
For each slab plan page, identify:
- Which building it belongs to
- Which floor level
- How many columns of each type appear on that page

━━━ TASK 3 — DETAIL / SECTION PAGES ━━━
Identify pages that are "DETAIL", "CONNECTION DETAIL", "SECTION", or
"TYPICAL DETAIL" drawings. Columns shown on these pages are NOT real column
locations — they are blow-up details of a specific connection.
List these as "detail_pages".

━━━ TASK 4 — ORPHAN COLUMNS ━━━
If the column schedule lists more columns than are assigned to any
specific building/floor, note the difference as "orphan_columns".
These are columns whose location is unknown or ambiguous.

━━━ TASK 5 — FOUNDATION SCHEDULE ━━━
Find foundation/footing schedule tables. Extract:
- Each unique footing symbol (e.g. "PF1", "PF2", "PC1", "R1")
- Type: "pad" (pad footing / spread footing), "pile_cap", "raft", or "strip"
- Plan dimensions in mm (width × depth)
- Depth below ground level in mm (if stated)

━━━ TASK 6 — FOOTING PLAN PAGES ━━━
Identify which pages are footing/foundation plans
(titles like "FOOTING PLAN", "PILE CAP PLAN", "FOUNDATION PLAN").
List their 1-indexed page numbers as "footing_plan_pages".

━━━ OUTPUT ━━━
Return ONLY valid JSON (no markdown, no explanation):
{
  "column_types": {
    "SH":  {"width_mm": 600, "depth_mm": 600, "count_total": 10},
    "PG1": {"width_mm": 800, "depth_mm": 400, "count_total": 20}
  },
  "buildings": [
    {
      "name": "Building A",
      "floors": [
        {
          "level_name": "Ground Floor",
          "slab_plan_pages": [10],
          "columns": {"SH": 5, "PG1": 3},
          "total_columns": 8
        }
      ]
    }
  ],
  "detail_pages": [22, 23],
  "orphan_columns": {"SH": 2},
  "foundation_types": {
    "PF1": {"width_mm": 1500, "depth_mm": 1500, "type": "pad", "depth_below_gl_mm": 600},
    "PC1": {"width_mm": 2000, "depth_mm": 2000, "type": "pile_cap", "depth_below_gl_mm": 0}
  },
  "footing_plan_pages": [5, 6],
  "detection_confidence": "high"
}

If no column schedule is found, return "column_types": {}.
If no foundation schedule is found, return "foundation_types": {}.
All page numbers are 1-indexed.
"""


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_page_text(args: tuple) -> tuple:
    """Worker: extract text from a single page."""
    pdf_path, page_idx = args
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        return page_idx, text[:4000]   # cap per page to avoid token overflow
    finally:
        doc.close()


def _collect_all_text(pdf_path: str, page_indices: list) -> str:
    """Parallel text extraction, assembled into one string with page markers."""
    args = [(pdf_path, idx) for idx in page_indices]
    with ThreadPoolExecutor(max_workers=min(len(args), 8)) as ex:
        results = list(ex.map(_extract_page_text, args))

    parts = []
    for page_idx, text in sorted(results):
        parts.append(f"\n=== PAGE {page_idx + 1} ===\n{text}")
    return "\n".join(parts)


# ── Gemini call ───────────────────────────────────────────────────────────────

def _get_gemini_client():
    from google import genai
    from google.oauth2 import service_account

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project    = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location   = os.environ.get("VERTEX_LOCATION", "us-central1")

    if not creds_path or not project:
        raise EnvironmentError(
            "Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT in .env"
        )
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True, project=project, location=location, credentials=creds,
    ), os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _call_gemini_text(client, model: str, text_payload: str) -> dict:
    response = client.models.generate_content(
        model=model,
        contents=[_PROMPT_CENSUS, text_payload],
    )
    raw = response.text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```$",           "", raw, flags=re.MULTILINE).strip()

    for attempt in range(2):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 0:
                raw = re.sub(r',\s*\]', ']', raw)
                raw = re.sub(r',\s*\}', '}', raw)
            else:
                print(f"[ColumnAnalyzer] JSON parse failed. Raw: {raw[:300]}")
                return {}
    return {}


# ── Post-processing ───────────────────────────────────────────────────────────

def _normalize_result(raw: dict) -> dict:
    """Ensure all expected keys exist with sensible defaults."""
    return {
        "column_types":          raw.get("column_types", {}),
        "buildings":             raw.get("buildings", []),
        "detail_pages":          raw.get("detail_pages", []),
        "orphan_columns":        raw.get("orphan_columns", {}),
        "foundation_types":      raw.get("foundation_types", {}),
        "footing_plan_pages":    raw.get("footing_plan_pages", []),
        "detection_confidence":  raw.get("detection_confidence", "low"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_columns_and_foundations(
    pdf_path: str,
    page_indices: list,
    ai_floor_result: dict = None,
) -> dict:
    """
    Scan the PDF text with Gemini to build a column and foundation census.

    Args:
        pdf_path:        Path to the structural PDF.
        page_indices:    0-indexed page numbers to scan.
        ai_floor_result: Optional floor detection result for context (not required).

    Returns:
        dict with keys: column_types, buildings, detail_pages, orphan_columns,
                        foundation_types, footing_plan_pages, detection_confidence.
    """
    if not page_indices:
        return _normalize_result({})

    print(f"[ColumnAnalyzer] Scanning {len(page_indices)} pages for columns & foundations...")
    all_text = _collect_all_text(pdf_path, page_indices)

    # Optionally prepend floor context so Gemini can cross-reference
    context_prefix = ""
    if ai_floor_result:
        buildings_summary = json.dumps(
            [{"name": b["name"], "floors": [f["level_name"] for f in b.get("floors", [])]}
             for b in ai_floor_result.get("buildings", [])],
            indent=2
        )
        context_prefix = (
            f"Known buildings and floors from prior analysis:\n{buildings_summary}\n\n"
            f"Now analyze the following page text:\n"
        )

    client, model = _get_gemini_client()
    raw = _call_gemini_text(client, model, context_prefix + all_text)
    result = _normalize_result(raw)

    n_col_types = len(result["column_types"])
    n_fdn_types = len(result["foundation_types"])
    print(f"[ColumnAnalyzer] Found {n_col_types} column type(s), "
          f"{n_fdn_types} foundation type(s). "
          f"Confidence: {result['detection_confidence']}")
    return result
