"""
V2 Column Census — production-grade Gemini prompt for 10K+ PDFs.

Extracts EXACT column symbols as they appear on drawings (ACC52, CHCH95,
PG56, SHM, C1, …), maps them to buildings/floors with counts and
dimensions.  Never invents or normalizes symbols.

Public API:
    analyze_column_census(pdf_path, page_indices, floor_result, save_dir) -> dict
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import fitz


_PROMPT = r"""You are a PRINCIPAL STRUCTURAL ENGINEER analysing extracted text
from a structural PDF drawing set. Your task: build a complete COLUMN CENSUS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULE — EXACT SYMBOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column symbols MUST be the EXACT text as written on the drawing.
NEVER invent symbols.  NEVER use dimensions (e.g. "350x900") as symbols.
NEVER normalize or reformat.  If the drawing says "ACC52", return "ACC52".

A column symbol is a SHORT ALPHANUMERIC LABEL that identifies a column TYPE
in a column schedule or on a plan.  It is NOT a dimension string.

Examples of REAL column symbols seen in production drawings:
  C1, C2, C10, C100           — simple numbered concrete columns
  SH, SH1, SH2                — shear-head / steel-head
  PG1, PG2, PG56              — post/ground / precast / project-specific
  ACC52, CHCH95                — project-specific prefixes
  RC1, RC2                     — reinforced concrete
  UC150, UB200                 — universal column/beam (steel)
  P1, P2                       — piles or posts
  SC1, SC2                     — steel columns

On some drawings, columns appear as a CIRCLE containing "C" above a number
(e.g. "C" on top and "5" below — representing column C5).  In extracted
text this often appears as separate tokens "C" and "5" near each other.
Combine them: C5, C10, C15, etc.

Dashed or dotted circle around C/number means "column below only".
The word "OVER" near C/number means "column above only".

DO NOT include wall labels (W1, W2, W3, BW1, LW1, etc.) as column types.
Walls are separate structural elements — they belong in a wall schedule,
not the column census.  Only include symbols that represent COLUMNS.

━━━ STEP 1 — FIND COLUMN SCHEDULE ━━━
Search ALL pages for column schedule tables.  A column schedule is a table
with rows like:

  MARK | SIZE (mm)    | or  SYMBOL | WIDTH | DEPTH
  C1   | 600 × 600    |     SH     | 350   | 900
  C2   | 400 × 800    |     PG1    | 450   | 450

Extract EVERY unique column symbol with its cross-section dimensions
(width_mm × depth_mm).  Record which page(s) contain the schedule.
For each column, determine material:
  • "RC" — reinforced concrete column with rectangular/square WxD
    cross-section (C1, C2, RC1, ACC01, BCC025, project-specific marks).
  • "STEEL" — steel section / steel column mark (UC, UB, SH, SC, CH,
    SHS, CHS, RHS, I/H/channel section designations).
  • "UNKNOWN" — cannot determine from drawing evidence.

If NO schedule table is found, proceed to Step 2 — symbols may still be
found from plan text labels.

━━━ STEP 2 — SCAN PLAN PAGES FOR COLUMN LABELS ━━━
On each GA / slab plan page, column marks appear as:
  • Small filled or outlined rectangles (the column footprint)
  • A text label near the rectangle — that label IS the column symbol
  • The label may be inside the rectangle, adjacent, or connected by a
    leader line

Count how many instances of each symbol appear on each page.

If a page has column rectangles but NO text labels, report them as
unlabeled in warnings — do NOT invent a symbol name.

━━━ STEP 3 — MAP TO BUILDINGS & FLOORS ━━━
Using page titles (e.g. "LEVEL 1 GA PLAN - BUILDING A"):
  • Group column instances by building name
  • Group by floor level within each building
  • Sum counts per symbol per floor

━━━ STEP 4 — DETAIL & SECTION PAGES ━━━
Identify pages that are DETAIL, CONNECTION DETAIL, SECTION, or TYPICAL
DETAIL drawings.  Columns on these pages are blow-up illustrations, NOT
real column locations.  List as "detail_pages".

━━━ STEP 5 — FOUNDATION SCHEDULE ━━━
Find foundation/footing schedule tables.  Extract:
  • Each unique footing symbol (PF1, PF2, PC1, R1, F1, SF1, …)
  • Type: "pad", "pile_cap", "raft", or "strip"
  • Plan dimensions in mm (width × depth)
  • Depth below ground level in mm (if stated)

Identify footing/foundation plan pages (titled "FOOTING PLAN",
"PILE CAP PLAN", "FOUNDATION PLAN").

━━━ OUTPUT — VALID JSON ONLY ━━━
Return ONLY valid JSON.  No markdown fences, no explanation.
KEEP THE RESPONSE COMPACT — this is critical to avoid truncation on large PDFs.

{
  "column_types": {
    "<EXACT_SYMBOL>": {
      "width_mm": <number or null if unknown>,
      "depth_mm": <number or null if unknown>,
      "count_total": <number or null>,
      "material": "RC" | "STEEL" | "UNKNOWN",
      "source": "schedule" | "plan_text" | "legend"
    }
  },
  "buildings": [
    {
      "name": "Building A",
      "floors": [
        {
          "level_name": "Ground Floor",
          "slab_plan_pages": [10],
          "columns": {"C1": 5, "SH": 3},
          "total_columns": 8
        }
      ]
    }
  ],
  "column_schedule_pages": [<1-indexed page numbers>],
  "detail_pages": [<1-indexed page numbers>],
  "foundation_types": {
    "<EXACT_SYMBOL>": {
      "width_mm": <number or null>,
      "depth_mm": <number or null>,
      "type": "pad" | "pile_cap" | "raft" | "strip",
      "depth_below_gl_mm": <number or null>
    }
  },
  "footing_plan_pages": [<1-indexed page numbers>],
  "detection_confidence": "high" | "medium" | "low"
}

━━━ COMPACTNESS RULES — MUST FOLLOW ━━━
Large drawing sets (50+ pages, 4+ buildings) can have 200+ column symbols.
To avoid output truncation:

1. In "column_types": list EVERY unique symbol but use MINIMAL whitespace.
   Write one symbol per line, no extra spaces:
   "A-CC01":{"width_mm":350,"depth_mm":900,"count_total":1,"material":"RC","source":"schedule"},

2. In "buildings": for floors with many columns, list only symbols that
   actually appear on that floor. Use compact format:
   "columns":{"A-CC01":2,"A-CC03":4}

3. Do NOT include "orphan_columns" or "warnings" — omit them entirely.

4. Do NOT add ANY explanation text — JSON only.

━━━ VALIDATION — READ BEFORE ANSWERING ━━━
Before returning, check every key in "column_types":
  ✗ "350x900"         → WRONG — that is a dimension, not a symbol
  ✗ "C350x900"        → WRONG — dimension with "C" prefix
  ✗ "Column Type 1"   → WRONG — description, not a symbol
  ✓ "C1"              → CORRECT — actual label from drawing
  ✓ "SH"              → CORRECT — actual label from drawing
  ✓ "ACC52"           → CORRECT — actual label from drawing
  ✓ "A-CC01"          → CORRECT — building-prefixed column mark

If you can only find dimensions but no symbol labels anywhere in the PDF,
return "column_types": {} with detection_confidence "low".

All page numbers are 1-indexed.
"""


# ── Text extraction ──────────────────────────────────────────────────────────

def _extract_page_text(args: tuple) -> tuple:
    pdf_path, page_idx = args
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        return page_idx, text[:4000]
    finally:
        doc.close()


def _collect_all_text(pdf_path: str, page_indices: list) -> str:
    args = [(pdf_path, idx) for idx in page_indices]
    with ThreadPoolExecutor(max_workers=min(len(args), 8)) as ex:
        results = list(ex.map(_extract_page_text, args))
    parts = []
    for page_idx, text in sorted(results):
        parts.append(f"\n=== PAGE {page_idx + 1} ===\n{text}")
    return "\n".join(parts)


# ── Gemini call ──────────────────────────────────────────────────────────────

def _get_gemini_client():
    from google import genai
    from google.oauth2 import service_account

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    if not creds_path or not project:
        raise EnvironmentError(
            "Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT "
            "in .env")
    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True, project=project,
        location=location, credentials=creds,
    ), os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair of JSON truncated by output token limit."""
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
        elif ch in ('}', ']'):
            if stack:
                stack.pop()
    text = text.rstrip()
    if text.endswith(','):
        text = text[:-1]
    while text and text[-1] not in ('}', ']', '"', '0', '1', '2', '3',
                                     '4', '5', '6', '7', '8', '9',
                                     'l', 'e', 'u'):
        text = text[:-1]
    if text and text[-1] == ',':
        text = text[:-1]
    text += ''.join(reversed(stack))
    return text


def _call_gemini(client, model: str, payload: str) -> tuple[dict, str]:
    from google.genai import types
    try:
        thinking_cfg = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        thinking_cfg = None
    cfg_kwargs = dict(
        max_output_tokens=65536,
        response_mime_type="application/json",
    )
    if thinking_cfg is not None:
        cfg_kwargs["thinking_config"] = thinking_cfg
    response = client.models.generate_content(
        model=model, contents=[_PROMPT, payload],
        config=types.GenerateContentConfig(**cfg_kwargs))
    raw = (response.text or "").strip()
    print(f"[ColumnCensus] Raw response length: {len(raw)} chars")
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    for attempt in range(3):
        try:
            return json.loads(cleaned), raw
        except json.JSONDecodeError:
            if attempt == 0:
                cleaned = re.sub(r',\s*\]', ']', cleaned)
                cleaned = re.sub(r',\s*\}', '}', cleaned)
            elif attempt == 1:
                cleaned = _repair_truncated_json(cleaned)
                print("[ColumnCensus] JSON truncated — attempting repair")
            else:
                print(f"[ColumnCensus] JSON parse failed after repair. "
                      f"Raw length: {len(raw)}")
                return {}, raw
    return {}, raw


# ── Post-processing ──────────────────────────────────────────────────────────

_DIM_PATTERN = re.compile(r'^C?\d+[xX×]\d+$')
_WALL_PATTERN = re.compile(r'^(W|BW|LW|SW|CW)\d*$', re.I)

_STEEL_SYMBOL_RE = re.compile(r'^(UC|UB|SHS|CHS|RHS|SH|SC|CH)\d*', re.I)
_RC_SYMBOL_RE = re.compile(r'^(?:RC)?C\d+[A-Z]*$', re.I)


def infer_column_material(symbol: str, info: dict | None) -> str:
    """Infer RC/STEEL/UNKNOWN when Gemini omits material."""
    info = info or {}
    explicit = str(info.get("material") or "").strip().upper()
    if explicit in {"RC", "STEEL", "UNKNOWN"}:
        return explicit

    s = str(symbol or "").strip().upper()
    if _STEEL_SYMBOL_RE.match(s):
        return "STEEL"
    if _RC_SYMBOL_RE.match(s):
        return "RC"

    try:
        width = float(info.get("width_mm") or 0)
        depth = float(info.get("depth_mm") or 0)
    except (TypeError, ValueError):
        width = depth = 0

    if width >= 200 and depth >= 200:
        return "RC"
    if width > 0 and depth > 0 and min(width, depth) < 200:
        return "STEEL"
    return "UNKNOWN"


def _validate_symbols(col_types: dict) -> tuple[dict, list[str]]:
    """Drop symbols that are dimensions, not actual labels."""
    valid, warnings = {}, []
    for sym, info in col_types.items():
        sym = str(sym).strip()
        if not sym:
            continue
        if sym.endswith("*"):
            clean = sym.rstrip("*")
            if clean in col_types or clean in valid:
                warnings.append(
                    f"dropped duplicate asterisked symbol '{sym}'")
                continue
            sym = clean
        if _DIM_PATTERN.match(sym):
            warnings.append(
                f"dropped dimension-style symbol '{sym}' — "
                f"not an actual column label")
            continue
        if _WALL_PATTERN.match(sym):
            warnings.append(
                f"dropped wall symbol '{sym}' from column census")
            continue
        if len(sym) == 1:
            warnings.append(
                f"dropped single-letter family marker '{sym}' — "
                f"not an actual column instance label")
            continue
        valid[sym] = info
    return valid, warnings


def _normalize_result(raw: dict) -> tuple[dict, list[str]]:
    from collections import Counter

    col_types = raw.get("column_types", {})
    col_types, sym_warnings = _validate_symbols(col_types)
    for sym, info in list(col_types.items()):
        if isinstance(info, dict):
            info["material"] = infer_column_material(sym, info)

    # ── backfill: types seen in floor counts but missing from column_types ──
    floor_symbols: set[str] = set()
    floor_symbol_counts: Counter[str] = Counter()
    for b in raw.get("buildings", []):
        for f in b.get("floors", []):
            for symbol, count in (f.get("columns", {}) or {}).items():
                clean = str(symbol).strip().rstrip("*")
                if not clean:
                    continue
                floor_symbols.add(clean)
                try:
                    floor_symbol_counts[clean] += int(count or 0)
                except (TypeError, ValueError):
                    pass

    rc_dims: list[tuple[float, float]] = []
    for info in col_types.values():
        if isinstance(info, dict) and info.get("material") == "RC":
            w, d = info.get("width_mm"), info.get("depth_mm")
            if w and d:
                rc_dims.append((w, d))
    common_dim = Counter(rc_dims).most_common(1)[0][0] if rc_dims else None

    backfilled: list[str] = []
    for sym in sorted(floor_symbols):
        sym_clean = str(sym).strip().rstrip("*")
        if not sym_clean or sym_clean in col_types:
            continue
        if len(sym_clean) == 1:
            continue
        material = infer_column_material(sym_clean, {})
        if material == "STEEL":
            continue
        col_types[sym_clean] = {
            "width_mm": common_dim[0] if common_dim else None,
            "depth_mm": common_dim[1] if common_dim else None,
            "count_total": floor_symbol_counts.get(sym_clean, 0),
            "material": material,
            "source": "backfill_from_floor_counts",
        }
        backfilled.append(sym_clean)

    warnings = list(raw.get("warnings", []))
    warnings.extend(sym_warnings)
    if backfilled:
        warnings.append(
            f"backfilled {len(backfilled)} type(s) from floor counts "
            f"(not in schedule): {', '.join(backfilled)}")

    unresolved = sorted(
        str(sym).strip().rstrip("*") for sym in floor_symbols
        if str(sym).strip().rstrip("*") not in col_types
        and infer_column_material(str(sym).strip().rstrip("*"), {}) != "STEEL"
    )
    requested_confidence = str(raw.get("detection_confidence", "low") or "low")
    effective_confidence = requested_confidence
    consistency_status = "consistent"
    if backfilled:
        consistency_status = "recovered"
        if requested_confidence.lower() == "high":
            effective_confidence = "medium"
    if unresolved:
        consistency_status = "inconsistent"
        effective_confidence = "low"
        warnings.append(
            "column census remains inconsistent; floor symbols without "
            f"type definitions: {', '.join(unresolved)}")

    return {
        "column_types": col_types,
        "buildings": raw.get("buildings", []),
        "column_schedule_pages": raw.get("column_schedule_pages", []),
        "detail_pages": raw.get("detail_pages", []),
        "orphan_columns": raw.get("orphan_columns", {}),
        "foundation_types": raw.get("foundation_types", {}),
        "footing_plan_pages": raw.get("footing_plan_pages", []),
        "detection_confidence": effective_confidence,
        "consistency_report": {
            "status": consistency_status,
            "requested_confidence": requested_confidence,
            "effective_confidence": effective_confidence,
            "floor_symbols": sorted(floor_symbols),
            "backfilled_types": backfilled,
            "unresolved_types": unresolved,
        },
        "warnings": warnings,
    }, warnings


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_column_census(
    pdf_path: str,
    page_indices: list,
    floor_result: dict | None = None,
    save_dir: str | None = None,
) -> dict:
    """
    Production-grade column census via Gemini.

    Returns dict matching column_analyzer's output schema so the rest of
    the v2 pipeline (doc_analyze.py, app_v2.py) works unchanged.
    """
    if not page_indices:
        return {"column_types": {}, "buildings": [],
                "detail_pages": [], "orphan_columns": {},
                "foundation_types": {}, "footing_plan_pages": [],
                "detection_confidence": "low"}

    print(f"[ColumnCensus] Scanning {len(page_indices)} pages...")
    all_text = _collect_all_text(pdf_path, page_indices)

    context_prefix = ""
    if floor_result:
        buildings_summary = json.dumps(
            [{"name": b["name"],
              "floors": [f["level_name"]
                         for f in b.get("floors", [])]}
             for b in floor_result.get("buildings", [])],
            indent=2)
        context_prefix = (
            f"Known buildings and floors from prior analysis:\n"
            f"{buildings_summary}\n\n"
            f"Now analyze the following page text:\n")

    client, model = _get_gemini_client()
    parsed, raw_text = _call_gemini(
        client, model, context_prefix + all_text)
    result, warnings = _normalize_result(parsed)

    n_col = len(result["column_types"])
    n_fdn = len(result["foundation_types"])
    print(f"[ColumnCensus] Found {n_col} column type(s), "
          f"{n_fdn} foundation type(s). "
          f"Confidence: {result['detection_confidence']}")
    if warnings:
        for w in warnings:
            print(f"[ColumnCensus] WARN: {w}")

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out / f"column_census_{ts}_raw.txt").write_text(
            raw_text, encoding="utf-8")
        (out / f"column_census_{ts}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8")

    return result
