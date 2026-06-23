"""
V2 Wall Census — production-grade Gemini prompt for structural PDFs.

Extracts EXACT wall symbols as they appear on drawings (W1, SW1, BW1,
CORE-A, …), maps them to buildings/floors with counts and dimensions.
Never invents or normalizes symbols.

Two goals:
  GOAL A — WHERE are walls on plan pages (symbol + count per floor)
  GOAL B — WHAT do walls look like (schedule/legend/elevation pages)

Public API:
    analyze_wall_census(pdf_path, page_indices, floor_result, save_dir) -> dict
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
from a structural PDF drawing set. Your task: build a complete WALL CENSUS.

This census answers TWO questions:
  GOAL A — WHERE are walls? Which plan pages show wall positions, what are
           their symbols, how many of each symbol per floor/level?
  GOAL B — WHAT do walls look like? Which pages contain wall schedules,
           wall elevation drawings, wall legends, or wall tag definitions
           that show thickness, height, material, and cross-section shape?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULE — EXACT SYMBOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Wall symbols MUST be the EXACT text as written on the drawing.
NEVER invent symbols.  NEVER use dimensions (e.g. "200THK") as symbols.
NEVER normalize or reformat.  If the drawing says "SW-A1", return "SW-A1".

A wall symbol is a SHORT ALPHANUMERIC LABEL that identifies a wall TYPE
in a wall schedule or on a plan.  It is NOT a dimension string.

Examples of REAL wall symbols seen in production drawings:
  W1, W2, W3, W10              — simple numbered walls
  SW1, SW2, SW-A1              — shear walls
  BW1, BW2                     — blockwork walls / brick walls
  CW1, CW2                     — cavity walls / core walls
  RW1, RW2                     — retaining walls
  WL1, WL2                     — wall legs (L-shaped)
  CORE-A, CORE-B               — core walls (lift/stair shafts)
  GW1, GW2                     — ground-floor walls

On some drawings, walls may be labeled with a LEADER LINE pointing to
the wall rectangle.  The label at the end of the leader IS the symbol.
Walls in plan view appear as LONG FILLED or HATCHED RECTANGLES (aspect
ratio >= 3:1, typically 150-400mm thick).

━━━ STEP 1 — FIND WALL SCHEDULE / LEGEND (→ GOAL B) ━━━
Search ALL pages for wall specification tables or legends.  These are the
pages that tell us WHAT each wall type looks like.

A. "WALL SCHEDULE" or "SCHEDULE OF WALLS" — table with rows like:

  MARK | THICKNESS | HEIGHT  | MATERIAL   | TYPE
  W1   | 200       | 3000    | RC         | WALL
  SW1  | 300       | FULL HT | RC         | SHEAR WALL
  BW1  | 200       | 2700    | BLOCKWORK  | PARTITION

Extract from each row:
  - symbol (EXACT label)
  - thickness_mm (wall thickness: 150, 200, 250, 300, etc.)
  - height_mm (if stated; null if "FULL STOREY" or not stated)
  - material: "RC" | "MASONRY" | "BLOCKWORK" | "CAVITY" | "AAC" | "PRECAST"
  - wall_category: decide from the TYPE column or from the symbol prefix:
      "shear_wall" — SW prefix, or marked SHEAR, or diagonal hatch
      "core_wall" — around lift/stair core, CORE prefix
      "retaining_wall" — RW prefix, or marked RETAINING
      "wall" — default for all others

B. "WALL LEGEND" or "PLAN LEGEND" — a box showing fill patterns:
  Extract the same fields from the legend description text.

C. General notes that mention wall dimensions:
  "ALL WALLS 200mm THICK UNLESS NOTED OTHERWISE"
  → create a default entry with symbol "DEFAULT"

Record ALL pages containing schedule/legend/notes → wall_schedule_pages.

━━━ STEP 2 — FIND WALL ELEVATION & DETAIL PAGES (→ GOAL B) ━━━
Identify pages that show wall ELEVATION drawings or SECTION details.
These pages reveal the wall's HEIGHT, OPENINGS, and CROSS-SECTION SHAPE.

Look for page titles containing:
  • "WALL ELEVATION" + wall symbol (e.g. "WALL ELEVATION LW6")
  • "SHEAR WALL DETAIL" or "SHEAR WALL ELEVATION"
  • "CORE WALL SECTION"
  • "WALL SECTION" or "TYPICAL WALL SECTION"

For each elevation/detail page:
  1. Record the page number (1-indexed)
  2. Record which wall symbols appear on that page (from title or labels)
  3. CRITICAL — extract wall THICKNESS from the elevation drawing:
     • Look for a DIMENSION NUMBER near the wall title or section cut
     • Common patterns: "350" standalone, "275 THK", "200" near the wall
     • The thickness is usually a 3-digit number (150, 200, 250, 275, 300,
       350, 400) that is NOT a rebar callout
     • REBAR callouts look like "N28-200 E.F.V" or circled numbers with
       "N" prefix — these are NOT wall thickness
     • Dimension numbers like "1000 MIN." or "1853 MIN." are clearances,
       NOT wall thickness
     • The thickness is the SMALLEST dimension annotation on the elevation
       that represents the wall cross-section width (perpendicular to the
       wall face)

  If you find thickness from an elevation, UPDATE the corresponding
  wall_types entry with that thickness_mm value.

Record → wall_elevation_pages (list of page numbers).
Record → wall_detail_pages (section/connection detail pages).
Record → wall_elevations (mapping: symbol → page + thickness found).

━━━ STEP 3 — SCAN PLAN PAGES FOR WALL POSITIONS (→ GOAL A) ━━━
On each GA / slab plan / outline plan page, walls appear as:
  • Long filled or hatched rectangles (the wall in plan view)
  • A text label near the rectangle — that label IS the wall symbol
  • The label may be adjacent, on top, or connected by a leader line
  • Shear walls often have DIAGONAL HATCH pattern inside
  • Core walls cluster around lift shafts and stair cores
  • Unlabeled walls may show thickness annotation (e.g. "200" near wall)

For each plan page:
  1. Find ALL wall symbols that appear as text labels
  2. Count how many instances of each symbol on that page
  3. Note the page number (1-indexed)

If a page has wall rectangles but NO text labels, report as
"unlabeled_walls": true for that floor — do NOT invent a symbol name.

━━━ STEP 4 — MAP TO BUILDINGS & FLOORS (→ GOAL A) ━━━
Using page titles (e.g. "LEVEL 1 GA PLAN - BUILDING A"):
  • Group wall instances by building name
  • Group by floor level within each building
  • Sum counts per symbol per floor
  • Record which plan page(s) belong to each floor

This is the same grouping as column census — walls on the same plan page
as columns belong to the same building/floor/level.

━━━ OUTPUT — VALID JSON ONLY ━━━
Return ONLY valid JSON.  No markdown fences, no explanation.
KEEP THE RESPONSE COMPACT — critical to avoid truncation.

{
  "wall_types": {
    "<EXACT_SYMBOL>": {
      "thickness_mm": <number or null if unknown>,
      "height_mm": <number or null if unknown>,
      "material": <string or null>,
      "wall_category": "shear_wall" | "core_wall" | "retaining_wall" | "wall",
      "count_total": <number or null>,
      "source": "schedule" | "plan_text" | "legend" | "notes"
    }
  },
  "buildings": [
    {
      "name": "Building A",
      "floors": [
        {
          "level_name": "Level 01",
          "slab_plan_pages": [11],
          "walls": {"W1": 4, "SW1": 2},
          "total_walls": 6,
          "unlabeled_walls": false
        }
      ]
    }
  ],
  "wall_elevations": {
    "<EXACT_SYMBOL>": {
      "pages": [<1-indexed page numbers>],
      "thickness_mm": <number or null>
    }
  },
  "wall_schedule_pages": [<1-indexed page numbers>],
  "wall_elevation_pages": [<1-indexed page numbers>],
  "wall_detail_pages": [<1-indexed page numbers>],
  "detection_confidence": "high" | "medium" | "low"
}

━━━ COMPACTNESS RULES — MUST FOLLOW ━━━
1. In "wall_types": list EVERY unique symbol with MINIMAL whitespace.
   "W1":{"thickness_mm":200,"height_mm":null,"material":"RC","wall_category":"wall","count_total":12,"source":"schedule"},

2. In "buildings": list only symbols that actually appear on that floor.
   "walls":{"W1":4,"SW1":2}

3. Do NOT add ANY explanation text — JSON only.

━━━ VALIDATION — READ BEFORE ANSWERING ━━━
Before returning, check every key in "wall_types":
  ✗ "200THK"           → WRONG — that is a dimension
  ✗ "200mm wall"       → WRONG — description, not a symbol
  ✗ "Concrete Wall"    → WRONG — description
  ✓ "W1"               → CORRECT — actual label from drawing
  ✓ "SW-A1"            → CORRECT — actual label from drawing
  ✓ "CORE-B"           → CORRECT — actual label from drawing
  ✓ "BW1"              → CORRECT — actual label from drawing

If you can only find wall rectangles but no symbol labels anywhere,
return "wall_types": {} with detection_confidence "low".

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
    print(f"[WallCensus] Raw response length: {len(raw)} chars")
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
                print("[WallCensus] JSON truncated — attempting repair")
            else:
                print(f"[WallCensus] JSON parse failed after repair. "
                      f"Raw length: {len(raw)}")
                return {}, raw
    return {}, raw


# ── Post-processing ──────────────────────────────────────────────────────────

_DIM_PATTERN = re.compile(r'^C?\d+[xX×]\d+$')
_THK_PATTERN = re.compile(r'^\d+THK$', re.IGNORECASE)
_MM_PATTERN = re.compile(r'^\d+mm$', re.IGNORECASE)


def _validate_symbols(wall_types: dict) -> tuple[dict, list[str]]:
    """Drop symbols that are dimensions or descriptions, not actual labels."""
    valid, warnings = {}, []
    for sym, info in wall_types.items():
        sym = str(sym).strip()
        if not sym:
            continue
        if _DIM_PATTERN.match(sym):
            warnings.append(
                f"dropped dimension-style symbol '{sym}' — "
                f"not an actual wall label")
            continue
        if _THK_PATTERN.match(sym):
            warnings.append(
                f"dropped thickness-style symbol '{sym}' — "
                f"not an actual wall label")
            continue
        if _MM_PATTERN.match(sym):
            warnings.append(
                f"dropped mm-style symbol '{sym}' — "
                f"not an actual wall label")
            continue
        valid[sym] = info
    return valid, warnings


def _normalize_result(raw: dict) -> tuple[dict, list[str]]:
    wall_types = raw.get("wall_types", {})
    wall_types, sym_warnings = _validate_symbols(wall_types)
    warnings = list(raw.get("warnings", []))
    warnings.extend(sym_warnings)

    # backfill thickness from wall_elevations into wall_types
    wall_elevations = raw.get("wall_elevations", {})
    for sym, elev_info in wall_elevations.items():
        sym = str(sym).strip()
        if not sym or not isinstance(elev_info, dict):
            continue
        elev_thk = elev_info.get("thickness_mm")
        if elev_thk and sym in wall_types:
            existing = wall_types[sym]
            if isinstance(existing, dict) and not existing.get("thickness_mm"):
                existing["thickness_mm"] = elev_thk

    return {
        "wall_types": wall_types,
        "buildings": raw.get("buildings", []),
        "wall_elevations": wall_elevations,
        "wall_schedule_pages": raw.get("wall_schedule_pages", []),
        "wall_elevation_pages": raw.get("wall_elevation_pages", []),
        "wall_detail_pages": raw.get("wall_detail_pages", []),
        "detection_confidence": raw.get("detection_confidence", "low"),
    }, warnings


# ── Fallback: common-thickness backfill ─────────────────────────────────


def _backfill_common_thickness(wall_types: dict) -> list[str]:
    """For walls missing thickness, inherit from the most common thickness
    among walls of the same material.  Zero Gemini calls, zero PDF reads.

    Structural convention: walls of the same material in one project often
    share the same thickness (e.g. all RC walls = 250mm).  When Gemini
    returns thickness for some walls but not others, the missing ones
    likely share the majority thickness.
    """
    from collections import Counter
    warnings = []

    # group known thicknesses by material
    mat_thk: dict[str, list[float]] = {}
    for sym, info in wall_types.items():
        if not isinstance(info, dict):
            continue
        thk = info.get("thickness_mm")
        mat = (info.get("material") or "").upper() or "UNKNOWN"
        if thk and thk > 0:
            mat_thk.setdefault(mat, []).append(thk)

    # find majority thickness per material
    mat_common: dict[str, float] = {}
    for mat, thks in mat_thk.items():
        freq = Counter(thks)
        mat_common[mat] = freq.most_common(1)[0][0]

    # backfill missing walls
    for sym, info in wall_types.items():
        if not isinstance(info, dict):
            continue
        if info.get("thickness_mm"):
            continue
        mat = (info.get("material") or "").upper() or "UNKNOWN"
        common = mat_common.get(mat)
        if common:
            info["thickness_mm"] = common
            print(f"[WallCensus] common-thickness fallback: {sym} = "
                  f"{common}mm (majority {mat} walls)")
        else:
            warnings.append(f"{sym}: no thickness found, no {mat} "
                            f"reference walls")

    return warnings


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_wall_census(
    pdf_path: str,
    page_indices: list,
    floor_result: dict | None = None,
    save_dir: str | None = None,
) -> dict:
    """
    Production-grade wall census via Gemini.

    Returns dict with wall_types, buildings (with per-floor wall counts),
    wall_schedule_pages, wall_elevation_pages, wall_detail_pages.
    """
    if not page_indices:
        return {"wall_types": {}, "buildings": [],
                "wall_schedule_pages": [], "wall_elevation_pages": [],
                "wall_detail_pages": [],
                "detection_confidence": "low"}

    print(f"[WallCensus] Scanning {len(page_indices)} pages...")
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

    # fallback: walls missing thickness inherit from the most common
    # thickness among walls of the same material
    if result["wall_types"]:
        fb_warnings = _backfill_common_thickness(result["wall_types"])
        warnings.extend(fb_warnings)

    n_wall = len(result["wall_types"])
    n_elev = len(result["wall_elevation_pages"])
    print(f"[WallCensus] Found {n_wall} wall type(s), "
          f"{n_elev} elevation page(s). "
          f"Confidence: {result['detection_confidence']}")
    if warnings:
        for w in warnings:
            print(f"[WallCensus] WARN: {w}")

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out / f"wall_census_{ts}_raw.txt").write_text(
            raw_text, encoding="utf-8")
        (out / f"wall_census_{ts}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8")

    return result
