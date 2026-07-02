import re
import fitz  # pymupdf
from pathlib import Path
from PIL import Image
import io


FLOOR_PLAN_KEYWORDS = [
    "FLOOR PLAN", "SLAB PLAN", "LEVEL PLAN", "OUTLINE PLAN",
    "SUSPENDED SLAB", "DIAPHRAGM", "FRAMING PLAN", "STRUCTURAL PLAN",
    "LEVEL 0", "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5",
    "GROUND FLOOR", "FIRST FLOOR", "SECOND FLOOR", "ROOF", "PODIUM",
]

_GEOMETRY_TITLE_RE = re.compile(
    r"\b(GENERAL\s+ARRANGEMENT\s+PLAN|GA\s+PLAN|OUTLINE\s+PLAN|"
    r"FLOOR\s+PLAN|SLAB\s+PLAN|FRAMING\s+PLAN|STEELWORK\s+PLAN|"
    r"STEEL\s+MARKING\s+PLAN|MARKING\s+PLAN|LEVEL\s+\d+\s+PLAN|"
    r"LEVEL\s+\d+\s+OUTLINE\s+PLAN)\b",
    re.IGNORECASE,
)
_FOUNDATION_TITLE_RE = re.compile(
    r"\b(FOUNDATION\s+PLAN|FOOTING\s+PLAN|PILE\s+CAP\s+PLAN|"
    r"PILE\s+PLAN|PAD\s+FOOTING\s+PLAN)\b",
    re.IGNORECASE,
)
_LOADING_TITLE_RE = re.compile(r"\b(LOADING\s+PLAN|LOAD\s+PLAN)\b", re.IGNORECASE)
_EVIDENCE_TITLE_RE = re.compile(
    r"\b(SCHEDULE|DETAIL|ELEVATION|SECTION|LEGEND|NOTES)\b",
    re.IGNORECASE,
)

FFL_PATTERN = re.compile(
    r"(?:FFL|RL|EL|FL|AHD|NGL)\s*[=:+]?\s*([+-]?\d{1,4}(?:\.\d{1,3})?)\s*(?:m|M|mAHD)?",
    re.IGNORECASE,
)
SCALE_PATTERN = re.compile(r"\b1\s*[:/]\s*(\d{1,4})\b", re.IGNORECASE)
_EXPLICIT_SCALE_RE = re.compile(
    r"\bSCALE\b[^A-Z0-9]{0,20}\b1\s*[:/]\s*(\d{1,4})\b",
    re.IGNORECASE,
)
_TIME_LIKE_RE = re.compile(
    r"\b\d{1,2}\s*:\s*\d{2}(?:\s*:\s*\d{2})?\s*(?:AM|PM)?\b",
    re.IGNORECASE,
)
_DATE_LIKE_RE = re.compile(r"\b\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}\b")
SLAB_LABEL_PATTERN = re.compile(
    r"\b([A-Z]?S\d+[A-Z]?)\b|\b(SL[\s-]?\d+)\b|\b([A-Z]-S\d+-[A-Z]{2}\d+)\b",
    re.IGNORECASE,
)


_PT_TO_MM = 25.4 / 72.0


def load_pdf(path: str) -> fitz.Document:
    return fitz.open(path)


def get_pdf_metadata(doc: fitz.Document) -> dict:
    meta = doc.metadata or {}
    return {
        "page_count": doc.page_count,
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "page_size_pts": (doc[0].rect.width, doc[0].rect.height) if doc.page_count > 0 else (0, 0),
    }


def get_page_thumbnail(page: fitz.Page, dpi: int = 72) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def extract_text_blocks(page: fitz.Page) -> list[dict]:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    result = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    result.append({
                        "text": text,
                        "bbox": span["bbox"],  # (x0, y0, x1, y1) in PDF points
                        "size": span.get("size", 0),
                    })
    return result


def classify_pages(doc: fitz.Document) -> list[dict]:
    """Return list of page info dicts with is_floor_plan flag."""
    pages = []
    for i, page in enumerate(doc):
        text_blocks = extract_text_blocks(page)
        full_text = " ".join(b["text"] for b in text_blocks).upper()

        is_floor_plan = any(kw in full_text for kw in FLOOR_PLAN_KEYWORDS)
        title = _extract_page_title(text_blocks)
        ffl_values = extract_ffl_values(text_blocks)
        scale = detect_scale_from_blocks(text_blocks)

        pages.append({
            "index": i,
            "label": f"Page {i + 1}: {title}" if title else f"Page {i + 1}",
            "title": title,
            "is_floor_plan": is_floor_plan,
            "ffl_values": ffl_values,
            "scale": scale,
        })
    return pages


def _extract_page_title(text_blocks: list[dict]) -> str:
    """Heuristic: largest font text near bottom of page = drawing title."""
    if not text_blocks:
        return ""
    # Sort by font size descending, take top candidates
    sorted_blocks = sorted(text_blocks, key=lambda b: b["size"], reverse=True)
    for b in sorted_blocks[:5]:
        t = b["text"].strip()
        if len(t) > 4 and not t.replace(".", "").replace("-", "").replace(" ", "").isdigit():
            return t[:80]
    return ""


def extract_ffl_values(text_blocks: list[dict]) -> list[dict]:
    """Parse FFL annotations like 'FFL 44.000' or 'FFL=40.700'."""
    results = []
    for block in text_blocks:
        matches = FFL_PATTERN.findall(block["text"])
        for match in matches:
            try:
                ffl_m = float(match)
                results.append({
                    "ffl_m": ffl_m,
                    "ffl_mm": ffl_m * 1000,
                    "bbox": block["bbox"],
                    "source_text": block["text"],
                })
            except ValueError:
                pass
    return results


def detect_scale_from_blocks(text_blocks: list[dict]) -> int | None:
    """Find a drawing scale ratio like 'SCALE - 1 : 100'.

    Ratios are only trusted when they are tied to explicit SCALE context.  This
    intentionally rejects timestamp/date/path ratios such as '9:11:15 PM'.
    """
    return collect_scale_candidates_from_blocks(text_blocks).get("chosen_scale")


def collect_scale_candidates_from_blocks(text_blocks: list[dict]) -> dict:
    """Return accepted/rejected scale candidates with provenance.

    The detector is deliberately conservative: a bare ``1:15`` is not a
    drawing scale unless it is on or near a SCALE label.  This prevents footer
    timestamps and revision dates from silently shrinking/expanding models.
    """
    candidates: list[dict] = []
    blocks = list(text_blocks or [])

    for idx, block in enumerate(blocks):
        text = str(block.get("text", "") or "")
        for match in SCALE_PATTERN.finditer(text):
            candidates.append(_score_scale_candidate(
                ratio_text=match.group(0),
                denominator_text=match.group(1),
                context=text,
                bbox=block.get("bbox"),
                source=f"block:{idx}",
                match_start=match.start(),
                match_end=match.end(),
            ))

    joined_parts = [str(b.get("text", "") or "") for b in blocks]
    joined = " ".join(t for t in joined_parts if t)
    for match in _EXPLICIT_SCALE_RE.finditer(joined):
        denom = match.group(1)
        context = joined[max(0, match.start() - 80):match.end() + 80]
        if not any(c.get("denominator") == _safe_int(denom)
                   and "SCALE" in c.get("context", "").upper()
                   for c in candidates):
            candidates.append(_score_scale_candidate(
                ratio_text=match.group(0),
                denominator_text=denom,
                context=context,
                bbox=None,
                source="joined_text",
                match_start=max(0, context.upper().find("1")),
                match_end=len(context),
            ))

    accepted = [c for c in candidates if c.get("accepted")]
    accepted.sort(key=lambda c: (-float(c.get("score", 0)), c.get("source", "")))
    chosen = accepted[0]["denominator"] if accepted else None
    return {
        "schema": "scale_candidates_v2",
        "chosen_scale": chosen,
        "status": "verified" if chosen else "missing",
        "candidates": candidates,
        "reason": (
            f"Selected SCALE 1:{chosen} from explicit scale context."
            if chosen else
            "No numeric scale with explicit SCALE context was found."
        ),
    }


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _score_scale_candidate(
    *,
    ratio_text: str,
    denominator_text: str,
    context: str,
    bbox,
    source: str,
    match_start: int,
    match_end: int,
) -> dict:
    denominator = _safe_int(denominator_text)
    upper = context.upper()
    local = context[max(0, match_start - 40):match_end + 40]
    local_upper = local.upper()
    reasons: list[str] = []
    reject_reason = ""
    score = 0.0

    if denominator is None or not (10 <= denominator <= 2000):
        reject_reason = "denominator_out_of_range"
    elif _TIME_LIKE_RE.search(local) and "SCALE" not in local_upper:
        reject_reason = "timestamp_or_time_ratio"
    elif _DATE_LIKE_RE.search(local) and "SCALE" not in local_upper:
        reject_reason = "date_ratio"
    elif "SCALE" not in upper:
        reject_reason = "missing_scale_keyword"
    elif "DO NOT SCALE" in upper and not _EXPLICIT_SCALE_RE.search(context):
        reject_reason = "do_not_scale_without_numeric_scale"
    else:
        if _EXPLICIT_SCALE_RE.search(context):
            score += 100.0
            reasons.append("explicit_scale_phrase")
        if "SCALE" in local_upper:
            score += 50.0
            reasons.append("same_block_scale_keyword")
        if denominator in {50, 100, 200, 250, 500}:
            score += 5.0
            reasons.append("common_architectural_scale")
        if bbox:
            try:
                # Title block scales are usually near sheet lower area.
                y0 = float(bbox[1])
                score += max(0.0, min(10.0, y0 / 250.0))
            except Exception:
                pass

    return {
        "ratio_text": ratio_text,
        "denominator": denominator,
        "context": context[:240],
        "bbox": list(bbox) if bbox else None,
        "source": source,
        "score": round(score, 3),
        "accepted": not bool(reject_reason),
        "reject_reason": reject_reason,
        "evidence": reasons,
    }


def classify_page_role_from_blocks(text_blocks: list[dict]) -> dict:
    """Classify whether a page should export geometry or only provide evidence."""
    text = " ".join(str(b.get("text", "") or "") for b in text_blocks or [])
    upper = text.upper()
    title = _extract_page_title(text_blocks)
    title_upper = title.upper()
    floor_hits = [kw for kw in FLOOR_PLAN_KEYWORDS if kw in upper]
    evidence_hits = [
        kw for kw in (
            "SCHEDULE", "DETAIL", "ELEVATION", "SECTION", "LEGEND",
            "TYPICAL", "NOTES", "REINFORCEMENT", "DIAPHRAGM",
        )
        if kw in upper
    ]

    # Sheet title/page role is stronger than incidental words elsewhere on
    # the sheet.  Foundation/loading sheets often contain "ROOF", "LEVEL", or
    # "SCHEDULE" in notes; those terms must not make them slab geometry pages.
    if _FOUNDATION_TITLE_RE.search(title_upper):
        role = "foundation_plan"
        reason = "Sheet title indicates foundation/footing geometry, not a slab floor plan."
    elif _LOADING_TITLE_RE.search(title_upper):
        role = "evidence_only"
        reason = "Sheet title indicates a loading plan; use as evidence, not slab geometry authority."
    elif _GEOMETRY_TITLE_RE.search(title_upper) or "GENERAL ARRANGEMENT PLAN" in upper:
        role = "geometry_plan"
        reason = "Page title/text indicates a drawable floor/GA/outline/framing plan."
    elif evidence_hits:
        role = "evidence_only"
        reason = "Page appears to be schedule/detail/elevation/notes evidence only."
    elif floor_hits:
        role = "geometry_plan"
        reason = "Page text contains floor-plan terms and no stronger evidence-only title."
    else:
        role = "unknown"
    return {
        "schema": "page_role_classification_v2",
        "role": role,
        "floor_plan_evidence": floor_hits,
        "evidence_only_terms": evidence_hits,
        "title": title,
        "reason": (
            reason if role != "unknown" else "No strong page role evidence found."
        ),
    }


def extract_slab_labels(text_blocks: list[dict]) -> list[dict]:
    """Find slab label annotations like S1, S2, B-S1-CW01."""
    results = []
    for block in text_blocks:
        matches = SLAB_LABEL_PATTERN.findall(block["text"])
        for match_groups in matches:
            label = next((g for g in match_groups if g), None)
            if label:
                results.append({
                    "label": label.strip(),
                    "bbox": block["bbox"],
                })
    return results


def convert_pts_to_mm(pts: float, scale: float = 100.0) -> float:
    """
    Convert PDF points to real-world millimeters using the drawing scale.
    
    scale: denominator of the scale ratio (e.g., 100 means 1:100).
    """
    return pts * _PT_TO_MM * scale


# ── Storey height from elevation ─────────────────────────────────────────────

_LEVEL_LABEL_RE = re.compile(r"^0*(\d{1,2})$")


def extract_storey_heights_from_elevation(doc: fitz.Document) -> dict[int, float]:
    """
    Derive inter-storey heights from elevation/section drawing pages.

    Approach: level label text ("LEVEL 01", "LEVEL 02"…) is placed AT each floor line
    in elevation drawings. The Y-coordinate difference between adjacent labels, scaled by
    the drawing scale, gives the storey height.

    Returns: {level_num: storey_height_m}  e.g. {1: 3.800, 2: 4.500, 3: 4.000, ...}
    where storey_height_m is the floor-to-floor height FROM that level to the level above.
    """
    best_result: dict[int, float] = {}
    best_count = 0

    for page in doc:
        words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,word)

        # Collect LEVEL N and ROOF label positions
        positions: list[tuple] = []  # (level_key, x_center, y_center)
        for i, w in enumerate(words):
            txt = w[4].strip()
            if txt.upper() == "LEVEL":
                for j in range(i + 1, min(i + 3, len(words))):
                    nxt = words[j][4].strip(",.")
                    m = _LEVEL_LABEL_RE.fullmatch(nxt)
                    if m:
                        lvl = int(m.group(1))
                        if 1 <= lvl <= 20:
                            positions.append((lvl, (w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
                        break
            elif txt.upper() == "ROOF":
                positions.append(("ROOF", (w[0] + w[2]) / 2, (w[1] + w[3]) / 2))

        if len(positions) < 3:
            continue

        # Detect scale for this page (default 100)
        blocks = extract_text_blocks(page)
        scale = detect_scale_from_blocks(blocks) or 100

        # Cluster positions by X bucket (50pt bins) — each cluster = one elevation view
        from collections import defaultdict
        x_bucket = lambda x: round(x / 50) * 50
        by_bucket: dict = defaultdict(list)
        for lvl_key, xc, yc in positions:
            by_bucket[x_bucket(xc)].append((lvl_key, xc, yc))

        # Find the column with the most complete, monotone level sequence
        for bucket, group in by_bucket.items():
            # Sort by Y ascending (top of page = higher floor in elevation drawings)
            col = sorted(group, key=lambda p: p[2])

            # Extract only integer levels from this column
            int_levels = [(lvl, y) for lvl, _, y in col if isinstance(lvl, int)]
            if len(int_levels) < 2:
                continue

            # Check monotone (levels must be in descending order as Y increases)
            lvl_seq = [l for l, _ in int_levels]
            if lvl_seq != sorted(lvl_seq, reverse=True):
                continue  # mixed elevations, not a clean column

            heights: dict[int, float] = {}
            for i in range(len(int_levels) - 1):
                upper_lvl, upper_y = int_levels[i]
                lower_lvl, lower_y = int_levels[i + 1]
                if lower_lvl != upper_lvl - 1:
                    continue  # non-consecutive, skip
                diff_pt = lower_y - upper_y
                storey_m = round(diff_pt * _PT_TO_MM * scale / 1000.0, 3)
                if 2.0 <= storey_m <= 8.0:  # sanity: 2m–8m per storey
                    heights[lower_lvl] = storey_m  # height FROM lower_lvl TO lower_lvl+1

            if len(heights) > best_count:
                best_count = len(heights)
                best_result = heights

    return best_result
