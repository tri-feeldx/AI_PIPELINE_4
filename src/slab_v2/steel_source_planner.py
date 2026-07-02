"""Document-level steel source planning.

This module is intentionally conservative.  It finds pages and symbols that
look like steel evidence, but it does not create geometry.  The page detector
and Ruby exporter use this census to decide where steel is expected and why a
page may legitimately have zero steel.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import DocAnalysis


_CORE_STEEL_PREFIXES = (
    "SHS", "CHS", "RHS", "PFC", "RFB", "UC", "UB", "SH", "SC", "CH", "PF", "RB",
)
_PROJECT_STEEL_PREFIXES = (
    "BT", "CT", "TF", "LA", "EA", "UA", "PFB", "PFC", "D",
)
_ALL_STEEL_PREFIXES = tuple(dict.fromkeys(_CORE_STEEL_PREFIXES + _PROJECT_STEEL_PREFIXES))
_STEEL_MARK_RE = re.compile(
    r"\b(?:SHS|CHS|RHS|PFC|RFB|UC|UB|SH|CH|SC|PF|RB|BT|CT|TF|LA|EA|UA|PFB|D)\s*[-/]?\s*"
    r"(?:\d+[A-Z0-9*]*|[A-Z]\d+[A-Z0-9*]*|\*)\b",
    re.I)
_STEEL_PREFIX_RE = re.compile(
    r"^(SHS|CHS|RHS|PFC|RFB|UC|UB|SH|SC|CH|PF|RB|BT|CT|TF|LA|EA|UA|PFB|D)\w*",
    re.I)
_STEEL_WORD_STOPLIST = {
    "SCALE", "SCALES", "SCABBLE", "SCHEDULE", "SCHEDULES", "SCREW",
    "SHADING", "SHADOW", "SHALE", "SHALL", "SHEET", "SHEETING",
    "SHELF", "SHOP", "SHORT", "SHOULD", "SHOWN",
    "CHANGE", "CHAMFER", "CHFOR", "CHREFER",
    "UBAND", "UBREFER", "UBTO", "UBUC",
    "UCPLAN", "UCTO", "UCCOLUMN", "UCCOLUMNS",
    "SHREFER", "SHDENOTES", "SHSTAIR",
    "SHSBEAMS", "SHSCOLUMN", "SHSHEADER", "SHSMAX", "SHSPLAN",
    "SHSPOST", "SHSPURLINS", "SHSSHS", "SHSUNO",
}
_STEEL_CONTEXT_RE = re.compile(
    r"\b(STEEL|STEELWORK|STRUCTURAL STEEL|STEEL COLUMN|STEEL BEAM|"
        r"FRAMING|BRACING|TRUSS|RAFTER|PURLIN|GIRT|FACADE|"
        r"UC|UB|SHS|CHS|RHS|PFC|RFB|NLB|BT|CT|TF|LA)\b", re.I)
_BEAM_CONTEXT_RE = re.compile(r"\b(BEAM|FRAMING|RAFTER|PURLIN|JOIST)\b", re.I)
_COLUMN_CONTEXT_RE = re.compile(r"\b(COLUMN|POST|STANCHION)\b", re.I)
_BRACE_CONTEXT_RE = re.compile(r"\b(BRAC|X[- ]?BRAC|DIAGONAL|TRUSS)\b", re.I)
_FLOOR_CONTEXT_RE = re.compile(r"\b(STEEL DECK|METAL DECK|COMPOSITE|BONDEK)\b", re.I)
_MARKING_PLAN_RE = re.compile(r"\b(STEEL\s+MARKING\s+PLAN|MARKING\s+PLAN)\b", re.I)
_DIAPHRAGM_RE = re.compile(r"\b(SEISMIC\s+DIAPHRAGM|DIAPHRAGM\s+REINFORCEMENT)\b", re.I)
_ELEVATION_RE = re.compile(r"\b(ELEVATION|WALL\s+ELEVATION)\b", re.I)
_SECTION_RE = re.compile(r"\b(SECTION|LONGITUDINAL\s+SECTION|CROSS\s+SECTION)\b", re.I)
_SCHEDULE_RE = re.compile(r"\b(SCHEDULE|STEEL\s+COLUMN\s+SCHEDULE|BEAM\s+SCHEDULE)\b", re.I)
_DETAIL_RE = re.compile(r"\b(DETAIL|TYPICAL|CONNECTION)\b", re.I)
_PLAN_RE = re.compile(r"\b(PLAN|GA\s+PLAN|OUTLINE\s+PLAN)\b", re.I)
_LEVEL_RE = re.compile(r"\b(?:LEVEL|L)\s*0?(\d+)|\b(ROOF)\b", re.I)
_LEVEL_PLAN_TITLE_RE = re.compile(
    r"\bLEVEL\s*0?(\d+)\b[^\n\r]{0,160}?"
    r"\b(?:GENERAL\s+ARRANGEMENT\s+PLAN|GA\s+PLAN|OUTLINE\s+PLAN|STEEL\s+PLAN|STEELWORK\s+PLAN|FRAMING\s+PLAN|PLAN)\b",
    re.I,
)
_PLAN_LEVEL_TITLE_RE = re.compile(
    r"\b(?:GENERAL\s+ARRANGEMENT\s+PLAN|GA\s+PLAN|OUTLINE\s+PLAN|STEEL\s+PLAN|STEELWORK\s+PLAN|FRAMING\s+PLAN|PLAN)\b"
    r"[^\n\r]{0,160}?\bLEVEL\s*0?(\d+)\b",
    re.I,
)
_ROOF_PLAN_TITLE_RE = re.compile(
    r"\b(?:LOWER\s+|UPPER\s+)?ROOF\b[^\n\r]{0,160}?"
    r"\b(?:OUTLINE\s+PLAN|STEEL\s+PLAN|STEELWORK\s+PLAN|FRAMING\s+PLAN|PLAN)\b",
    re.I,
)
_PLAN_ROOF_TITLE_RE = re.compile(
    r"\b(?:GENERAL\s+ARRANGEMENT\s+PLAN|GA\s+PLAN|OUTLINE\s+PLAN|STEEL\s+PLAN|STEELWORK\s+PLAN|FRAMING\s+PLAN|PLAN)\b"
    r"[^\n\r]{0,160}?\b(?:LOWER\s+|UPPER\s+)?ROOF\b",
    re.I,
)
_STEELWORK_REFERENCE_RE = re.compile(
    r"\bREFER(?:\s+TO)?\s+(?:DRAWING|DRG|PLAN)[^\n\r]{0,120}?"
    r"\b(?:STEELWORK|STEEL\s+PLAN|STEEL\s+STAIRS|STEEL\s+UNDER)\b",
    re.I,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text or "").upper())


def _aliases(symbol: str) -> list[str]:
    norm = _normalize(symbol)
    out = {norm}
    m = re.match(r"^([A-Z]+)([0-9].*)$", norm)
    if m:
        out.add(f"{m.group(1)} {m.group(2)}")
        out.add(f"{m.group(1)}-{m.group(2)}")
    return sorted(x for x in out if x)


def _symbol_prefix(symbol: str) -> str:
    norm = _normalize(symbol)
    m = re.match(r"^([A-Z]+)", norm)
    return m.group(1) if m else ""


def _prefix_role(prefix: str, context: str = "", source_type: str = "") -> str:
    prefix = str(prefix or "").upper()
    if source_type == "diaphragm_reinforcement":
        return "DIAPHRAGM_REINFORCEMENT"
    if _BRACE_CONTEXT_RE.search(context):
        return "STEEL_BRACING"
    if _FLOOR_CONTEXT_RE.search(context):
        return "STEEL_FLOOR_DECK"
    if _COLUMN_CONTEXT_RE.search(context):
        return "STEEL_COLUMN"
    if re.search(r"\b(PURLIN|GIRT|RAFTER|FASCIA|FACADE|FLY\s+BRACING)\b", context, re.I):
        return "PURLIN_GIRT"
    if _BEAM_CONTEXT_RE.search(context):
        return "STEEL_BEAM"
    if prefix in {"UB", "PFC", "RFB", "PF", "RB", "BT", "CT", "TF", "LA", "EA", "UA", "PFB"}:
        return "STEEL_BEAM"
    if prefix in {"SHS", "CHS", "RHS", "UC", "SC", "SH", "CH"}:
        return "STEEL_COLUMN"
    return "UNKNOWN"


def _classify_source_page(text: str) -> str:
    """Classify a steel source page by sheet text, without creating geometry."""
    text = text or ""
    if _DIAPHRAGM_RE.search(text):
        return "diaphragm_reinforcement"
    if _MARKING_PLAN_RE.search(text):
        return "marking_plan"
    # Sheet title/page role must win over incidental notes.  Many GA/floor
    # plans contain "refer schedule" text; those pages are still position
    # sources and must be scanned for anchors/geometry.
    if _is_level_plan(text):
        if _is_floor_plan_with_steel_marks(text):
            return "floor_plan_with_steel_marks"
        return "plan"
    if _is_floor_plan_with_steel_marks(text):
        return "floor_plan_with_steel_marks"
    if _SCHEDULE_RE.search(text):
        return "schedule"
    if _ELEVATION_RE.search(text):
        return "elevation"
    if _SECTION_RE.search(text):
        return "section"
    if _DETAIL_RE.search(text):
        return "detail"
    if _PLAN_RE.search(text):
        return "plan"
    return "reference_only"


def _steel_mark_hits(text: str) -> list[str]:
    hits: list[str] = []
    for m in _STEEL_MARK_RE.finditer(text or ""):
        mark = _normalize(m.group(0))
        ctx = _page_excerpt(text or "", m.start())
        if _is_plausible_steel_symbol(
            mark, source="pdf_text_scan", context=ctx):
            hits.append(mark)
    return hits


def _is_level_plan(text: str) -> bool:
    text = text or ""
    return bool(
        _LEVEL_PLAN_TITLE_RE.search(text)
        or _PLAN_LEVEL_TITLE_RE.search(text)
        or _ROOF_PLAN_TITLE_RE.search(text)
        or _PLAN_ROOF_TITLE_RE.search(text)
    )


def _is_floor_plan_with_steel_marks(text: str) -> bool:
    """Return true for floor/outline plans that carry drawable steel marks.

    Some structural floor plans contain the word "schedule" in notes or title
    blocks.  Those pages are still position sources when the sheet title is a
    level plan and local steel symbols/callouts are present.
    """
    text = text or ""
    if not _is_level_plan(text):
        return False
    hits = _steel_mark_hits(text)
    if len(set(hits)) >= 2:
        return True
    return bool(hits and (_STEEL_CONTEXT_RE.search(text) or _STEELWORK_REFERENCE_RE.search(text)))


def _steel_role(symbol: str, context: str = "", source_type: str = "") -> str:
    s = _normalize(symbol)
    if source_type == "diaphragm_reinforcement":
        return "DIAPHRAGM_REINFORCEMENT"
    if source_type == "reference_only" and not _STEEL_CONTEXT_RE.search(context):
        return "REFERENCE_ONLY"
    if _BRACE_CONTEXT_RE.search(context):
        return "STEEL_BRACING"
    if _FLOOR_CONTEXT_RE.search(context):
        return "STEEL_FLOOR_DECK"
    if re.search(r"\b(PURLIN|GIRT|RAFTER|FASCIA|FACADE|FLY\s+BRACING)\b", context, re.I):
        return "PURLIN_GIRT"
    if _BEAM_CONTEXT_RE.search(context):
        return "STEEL_BEAM"
    role = _prefix_role(_symbol_prefix(s), context, source_type)
    if role != "UNKNOWN":
        return role
    return "REFERENCE_ONLY" if source_type == "reference_only" else "STEEL_COLUMN"


def _member_type_from_role(role: str) -> str:
    role = str(role or "").upper()
    if role in {"STEEL_BRACING", "BRACING"}:
        return "BRACING"
    if role in {"STEEL_BEAM", "PURLIN_GIRT", "FACADE_STEEL", "BEAM"}:
        return "BEAM"
    if role in {"STEEL_FLOOR_DECK", "DIAPHRAGM_REINFORCEMENT", "FLOOR"}:
        return "FLOOR"
    if role in {"REFERENCE_ONLY"}:
        return "REVIEW_ONLY"
    return "COLUMN"


def _symbol_kind(symbol: str, context: str = "", source_type: str = "") -> str:
    return _member_type_from_role(_steel_role(symbol, context, source_type))


def _levels_from_context(context: str) -> list[str]:
    levels: list[str] = []
    for m in _LEVEL_RE.finditer(context or ""):
        if m.group(2):
            value = "ROOF"
        else:
            value = f"LEVEL {int(m.group(1)):02d}"
        if value not in levels:
            levels.append(value)
    return levels[:6]


def _level_name(raw_num: str) -> str:
    try:
        return f"LEVEL {int(raw_num):02d}"
    except Exception:
        return f"LEVEL {raw_num}"


def _source_level_hints(text: str, source_type: str) -> list[str]:
    """Extract level hints without letting reference notes dominate plan pages."""
    text = text or ""
    hints: list[str] = []
    if _ROOF_PLAN_TITLE_RE.search(text) or _PLAN_ROOF_TITLE_RE.search(text):
        hints.append("ROOF")
    for m in _LEVEL_PLAN_TITLE_RE.finditer(text):
        hints.append(_level_name(m.group(1)))
    for m in _PLAN_LEVEL_TITLE_RE.finditer(text):
        hints.append(_level_name(m.group(1)))
    if hints:
        return list(dict.fromkeys(hints))[:4]
    if source_type in {"elevation", "section", "detail", "schedule"}:
        return _levels_from_context(text[:3600])
    return []


def _is_plausible_steel_symbol(norm: str, *, source: str = "", context: str = "",
                               source_type: str = "") -> bool:
    if not norm or norm in _STEEL_WORD_STOPLIST:
        return False
    if norm in {"CH", "SH", "UC", "UB", "SC", "SHS", "CHS", "RHS", "BT", "CT", "TF", "LA", "EA", "UA", "D"}:
        # Generic prefixes are only useful when a higher-level semantic source
        # already called them steel.  Raw PDF text often has false hits.
        return source in {
            "document_column_census",
            "per_floor_column_census",
            "steel_census",
        }
    prefix = _symbol_prefix(norm)
    if prefix not in _ALL_STEEL_PREFIXES:
        return False
    if not re.match(
        r"^(?:SHS|CHS|RHS|PFC|RFB|UC|UB|SC|SH|CH|PF|RB|BT|CT|TF|LA|EA|UA|PFB|D)(?:\d|[A-Z]\d)",
        norm,
    ):
        return False
    if prefix in _PROJECT_STEEL_PREFIXES:
        trusted_sources = {
            "document_column_census",
            "per_floor_column_census",
            "steel_census",
        }
        trusted_source_types = {
            "marking_plan", "floor_plan_with_steel_marks", "elevation",
            "section", "detail", "schedule",
        }
        if source in trusted_sources or source_type in trusted_source_types:
            return True
        return bool(_STEEL_CONTEXT_RE.search(context or ""))
    return True


def _add_symbol(rows: dict, symbol: str, *, source: str, pages=None,
                context: str = "", confidence: float = 0.75,
                source_type: str = "") -> None:
    norm = _normalize(symbol)
    if not norm or not _STEEL_PREFIX_RE.match(norm):
        return
    if not _is_plausible_steel_symbol(
        norm, source=source, context=context, source_type=source_type):
        return
    role = _steel_role(norm, context, source_type)
    member_type = _member_type_from_role(role)
    item = rows.setdefault(norm, {
        "symbol": norm,
        "aliases": _aliases(norm),
        "member_type": member_type,
        "role": role,
        "section": norm,
        "expected_pages": [],
        "source_types": [],
        "detail_pages": [],
        "export_default": role not in {"DIAPHRAGM_REINFORCEMENT", "REFERENCE_ONLY"},
        "source": source,
        "confidence": confidence,
        "evidence": [],
    })
    kind = member_type
    # Prefer the most specific context gathered across the document.  A symbol
    # seen in a bracing/beam schedule should not remain a generic column just
    # because a shorter text hit was processed earlier.
    if item.get("member_type") == "COLUMN" and kind not in {"COLUMN", "REVIEW_ONLY"}:
        item["member_type"] = kind
        item["role"] = role
    if source_type and source_type not in item["source_types"]:
        item["source_types"].append(source_type)
    if source_type in {"elevation", "section", "detail", "schedule"} and pages:
        existing = set(item.get("detail_pages") or [])
        item["detail_pages"] = sorted(existing | {int(p) for p in pages})
    if role in {"DIAPHRAGM_REINFORCEMENT", "REFERENCE_ONLY"}:
        item["export_default"] = False
    if pages:
        existing = set(item.get("expected_pages") or [])
        item["expected_pages"] = sorted(existing | {int(p) for p in pages})
    if context:
        excerpt = " ".join(context.split())[:240]
        if excerpt not in item["evidence"]:
            item["evidence"].append(excerpt)
    item["confidence"] = max(float(item.get("confidence") or 0), confidence)


def _symbol_families(symbol_rows: list[dict]) -> list[dict]:
    families: dict[str, dict] = {}
    for row in symbol_rows:
        symbol = str(row.get("symbol") or "")
        prefix = _symbol_prefix(symbol)
        if not prefix:
            continue
        fam = families.setdefault(prefix, {
            "prefix": prefix,
            "expected_role": row.get("role") or _prefix_role(prefix),
            "sample_symbols": [],
            "source_pages": [],
            "source_types": [],
            "confidence": 0.0,
            "scan_policy": (
                "project_context_required"
                if prefix in _PROJECT_STEEL_PREFIXES else "known_steel_prefix"
            ),
        })
        if symbol and symbol not in fam["sample_symbols"]:
            fam["sample_symbols"].append(symbol)
        for page in row.get("expected_pages") or []:
            if page not in fam["source_pages"]:
                fam["source_pages"].append(page)
        for source_type in row.get("source_types") or []:
            if source_type not in fam["source_types"]:
                fam["source_types"].append(source_type)
        fam["confidence"] = max(float(fam["confidence"]), float(row.get("confidence") or 0))
    out = []
    for fam in families.values():
        fam["sample_symbols"] = sorted(fam["sample_symbols"])[:12]
        fam["source_pages"] = sorted(fam["source_pages"])[:20]
        fam["source_types"] = sorted(fam["source_types"])
        out.append(fam)
    return sorted(out, key=lambda row: row["prefix"])


def _page_excerpt(text: str, match_start: int, width: int = 260) -> str:
    lo = max(0, match_start - width // 2)
    hi = min(len(text), match_start + width // 2)
    return " ".join(text[lo:hi].split())


def _write_source_page_images(pdf_path: str, source_pages: list[dict],
                              out_dir: Path, cfg: SlabV2Config) -> None:
    if not getattr(cfg, "debug_images", False):
        return
    doc = fitz.open(pdf_path)
    try:
        for row in source_pages[:12]:
            try:
                page_no = int(row.get("page", 0))
                if page_no < 1 or page_no > len(doc):
                    continue
                path = out_dir / f"steel_source_page_p{page_no:02d}.png"
                pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False)
                pix.save(str(path))
                row["preview_path"] = str(path)
            except Exception as exc:
                row["preview_error"] = str(exc)
    finally:
        doc.close()


def build_steel_census(pdf_path: str, analysis: DocAnalysis,
                       cfg: SlabV2Config, out_dir: Path) -> dict:
    """Build a compact steel census from document analysis and PDF text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols: dict[str, dict] = {}
    source_pages: list[dict] = []
    detail_members: list[dict] = []
    detail_seen: set[tuple[str, int, str]] = set()
    classification_rows: list[dict] = []
    warnings: list[str] = []

    # Existing doc analysis may already know material=STEEL. Keep RC detector
    # behavior unchanged; this is only input for the steel subsystem.
    for sym, ct in (analysis.column_types or {}).items():
        material = str(getattr(ct, "material", "") or "").upper()
        if material == "STEEL" or _STEEL_PREFIX_RE.match(_normalize(sym)):
            pages = []
            for entry in analysis.columns_per_floor or []:
                counts = entry.get("counts") or {}
                if sym in counts or _normalize(sym) in {_normalize(k) for k in counts}:
                    pages.extend(int(p) + 1 for p in entry.get("pages", []) or [])
            _add_symbol(symbols, sym, source="document_column_census",
                        pages=pages, context=f"{sym} material={material}",
                        confidence=0.85 if material == "STEEL" else 0.72)

    for entry in analysis.columns_per_floor or []:
        for sym in (entry.get("counts") or {}):
            if _STEEL_PREFIX_RE.match(_normalize(sym)):
                _add_symbol(symbols, sym, source="per_floor_column_census",
                            pages=[int(p) + 1 for p in entry.get("pages", []) or []],
                            context=f"{sym} from per-floor census",
                            confidence=0.72)

    prompt = {
        "task": "deterministic steel source planner",
        "note": "No geometry is created here. Steel must be verified later by vector geometry.",
        "pdf": str(pdf_path),
    }
    (out_dir / "steel_source_planner_prompt.txt").write_text(
        json.dumps(prompt, indent=2, ensure_ascii=False), encoding="utf-8")

    doc = fitz.open(pdf_path)
    try:
        for pi, page in enumerate(doc):
            text = page.get_text("text") or ""
            source_type = _classify_source_page(text)
            score = 0
            hits = []
            for m in _STEEL_CONTEXT_RE.finditer(text):
                score += 4
                hits.append(_page_excerpt(text, m.start()))
            mark_hits = _steel_mark_hits(text)
            for m in _STEEL_MARK_RE.finditer(text):
                mark = _normalize(m.group(0))
                ctx = _page_excerpt(text, m.start())
                if not _is_plausible_steel_symbol(
                    mark, source="pdf_text_scan", context=ctx,
                    source_type=source_type):
                    continue
                if mark not in mark_hits:
                    mark_hits.append(mark)
                _add_symbol(symbols, mark, source="pdf_text_scan",
                            pages=[pi + 1], context=ctx, confidence=0.62,
                            source_type=source_type)
                if source_type in {"elevation", "section", "detail", "schedule", "marking_plan"}:
                    role = _steel_role(mark, ctx, source_type)
                    key = (mark, pi + 1, source_type)
                    if key not in detail_seen:
                        detail_seen.add(key)
                        detail_members.append({
                            "symbol": mark,
                            "aliases": _aliases(mark),
                            "role": role,
                            "member_type": _member_type_from_role(role),
                            "source_page": pi + 1,
                            "source_type": source_type,
                            "level_range": _levels_from_context(ctx),
                            "section": mark,
                            "evidence": [ctx],
                            "confidence": 0.76 if source_type in {"elevation", "section", "detail"} else 0.68,
                            "export_default": role not in {"DIAPHRAGM_REINFORCEMENT", "REFERENCE_ONLY"},
                        })
            mark_hits = sorted(set(mark_hits))
            if mark_hits:
                score += min(18, len(mark_hits) * 2)
            if _BEAM_CONTEXT_RE.search(text):
                score += 3
            if _BRACE_CONTEXT_RE.search(text):
                score += 3
            if _FLOOR_CONTEXT_RE.search(text):
                score += 3
            if source_type in {"marking_plan", "elevation", "section", "detail",
                               "schedule", "diaphragm_reinforcement",
                               "floor_plan_with_steel_marks"}:
                score += 5
            if score:
                level_hints = _source_level_hints(text, source_type)
                position_candidate = (
                    source_type in {"plan", "marking_plan", "floor_plan_with_steel_marks"}
                    or (_is_level_plan(text) and bool(mark_hits))
                    or (_is_level_plan(text) and bool(_STEELWORK_REFERENCE_RE.search(text)))
                )
                if source_type in {"schedule", "elevation", "section", "detail",
                                   "diaphragm_reinforcement", "reference_only"}:
                    position_status = "rejected"
                    position_reject_reason = (
                        "profile/schedule/reference source; not allowed to create x/y"
                    )
                elif position_candidate and source_type in {"plan", "marking_plan", "floor_plan_with_steel_marks"}:
                    position_status = "scanned"
                    position_reject_reason = ""
                elif position_candidate:
                    position_status = "candidate_rejected"
                    position_reject_reason = (
                        "level plan has steel evidence but classifier could not verify drawable steel anchors"
                    )
                else:
                    position_status = "not_candidate"
                    position_reject_reason = "no level plan steel-position evidence"
                source_pages.append({
                    "page": pi + 1,
                    "score": score,
                    "source_type": source_type,
                    "reason": "steel context text or steel-like symbols found",
                    "excerpt": hits[0] if hits else " ".join(text.split())[:260],
                    "symbols": sorted(set(mark_hits))[:20],
                    "level_hints": level_hints,
                    "position_candidate": position_candidate,
                    "position_candidate_status": position_status,
                    "position_reject_reason": position_reject_reason,
                })
                classification_rows.append({
                    "page": pi + 1,
                    "source_type": source_type,
                    "level_hints": level_hints,
                    "steel_symbol_count": len(set(mark_hits)),
                    "symbols": sorted(set(mark_hits))[:20],
                    "has_steelwork_reference": bool(_STEELWORK_REFERENCE_RE.search(text)),
                    "is_level_plan": _is_level_plan(text),
                    "position_candidate": position_candidate,
                    "position_candidate_status": position_status,
                    "position_reject_reason": position_reject_reason,
                    "reason": (
                        "level plan with steel marks promoted as position source"
                        if source_type == "floor_plan_with_steel_marks"
                        else "classified by steel source keywords"
                    ),
                    "excerpt": hits[0] if hits else " ".join(text.split())[:260],
                })
    finally:
        doc.close()

    source_pages = sorted(source_pages, key=lambda r: (-r["score"], r["page"]))
    position_types = {"plan", "marking_plan", "floor_plan_with_steel_marks"}
    profile_types = {"elevation", "section", "detail", "schedule"}
    reference_types = {"diaphragm_reinforcement", "reference_only"}
    position_sources = [
        r for r in source_pages
        if r.get("source_type") in position_types
    ][:20]
    profile_sources = [
        r for r in source_pages
        if r.get("source_type") in profile_types
    ][:20]
    reference_sources = [
        r for r in source_pages
        if r.get("source_type") in reference_types
    ][:20]
    position_pages = {int(r["page"]) for r in position_sources}
    profile_pages = {int(r["page"]) for r in profile_sources}
    reference_pages = {int(r["page"]) for r in reference_sources}
    source_pages_by_page = {int(r["page"]): r for r in source_pages}
    for row in classification_rows:
        page = int(row["page"])
        row["scanned_for_position"] = page in position_pages
        row["scanned_for_profile"] = page in profile_pages
        row["reference_only"] = page in reference_pages
        row["score"] = source_pages_by_page.get(page, {}).get("score", 0)
    symbol_rows = sorted(symbols.values(), key=lambda r: r["symbol"])
    symbol_families = _symbol_families(symbol_rows)
    steel_columns = [r for r in symbol_rows if r.get("member_type") == "COLUMN"]
    steel_beams = [r for r in symbol_rows if r.get("member_type") == "BEAM"]
    bracing = [r for r in symbol_rows if r.get("member_type") == "BRACING"]
    _write_source_page_images(pdf_path, source_pages, out_dir, cfg)

    steel_floor_regions = [
        {
            "page": row["page"],
            "source_type": row.get("source_type") or "steel_floor_context",
            "role": "DIAPHRAGM_REINFORCEMENT" if row.get("source_type") == "diaphragm_reinforcement" else "STEEL_FLOOR_DECK",
            "member_type": "FLOOR",
            "export_default": False if row.get("source_type") == "diaphragm_reinforcement" else True,
            "reason": "steel/composite/diaphragm floor wording found",
            "excerpt": row.get("excerpt", ""),
            "confidence": 0.62,
        }
        for row in source_pages
        if _FLOOR_CONTEXT_RE.search(row.get("excerpt", ""))
        or row.get("source_type") == "diaphragm_reinforcement"
    ]
    if symbol_rows:
        status = "steel_sources_planned"
        if not position_sources and (profile_sources or reference_sources or source_pages):
            zero_reason = (
                "steel_position_source_missing: steel evidence exists, but no "
                "floor/marking/framing/steelwork plan was verified as a position source."
            )
        else:
            zero_reason = ""
    elif source_pages:
        status = "steel_detected_but_unverified"
        zero_reason = "Steel source pages found, but no drawable steel symbols were extracted."
    else:
        status = "steel_source_missing"
        zero_reason = "No steel source pages, schedules, legend, or steel symbols found."

    census = {
        "status": status,
        "source": "deterministic_text_and_doc_analysis",
        "steel_source_views": source_pages[:20],
        "position_sources": position_sources,
        "profile_sources": profile_sources,
        "reference_sources": reference_sources,
        "source_classification_report": classification_rows,
        "symbol_families": symbol_families,
        "position_source_missing": bool(symbol_rows and not position_sources),
        "steel_position_source_pages": [r["page"] for r in position_sources],
        "steel_profile_source_pages": [r["page"] for r in profile_sources],
        "steel_detail_members": detail_members,
        "role_taxonomy": {
            "steel_column": len(steel_columns),
            "steel_beam": len(steel_beams),
            "steel_bracing": len(bracing),
            "steel_floor_or_diaphragm": len(steel_floor_regions),
            "detail_members": len(detail_members),
        },
        "steel_column_symbols": steel_columns,
        "steel_beam_symbols": steel_beams,
        "bracing_symbols": bracing,
        "steel_floor_regions": steel_floor_regions,
        "source_pages": source_pages[:20],
        "expected_symbols": [r["symbol"] for r in symbol_rows],
        "zero_steel_reason": zero_reason,
        "zero_or_low_steel_reason": (
            "position_source_missing"
            if symbol_rows and not position_sources else (
                "symbol_grammar_missing" if source_pages and not symbol_rows else ""
            )
        ),
        "warnings": warnings,
    }
    raw_text = json.dumps(census, indent=2, ensure_ascii=False)
    (out_dir / "steel_source_planner_raw.txt").write_text(raw_text, encoding="utf-8")
    (out_dir / "steel_source_planner.json").write_text(raw_text, encoding="utf-8")
    (out_dir / "steel_source_planner_parse_report.json").write_text(
        json.dumps({
            "parse_status": "ok",
            "source": "deterministic",
            "raw_response_length": len(raw_text),
            "warnings": warnings,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "steel_source_classification_report.json").write_text(
        json.dumps(classification_rows, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "steel_source_intelligence_raw.txt").write_text(raw_text, encoding="utf-8")
    (out_dir / "steel_source_intelligence.json").write_text(raw_text, encoding="utf-8")
    (out_dir / "steel_source_intelligence_parse_report.json").write_text(
        json.dumps({
            "parse_status": "ok",
            "source": "deterministic",
            "raw_response_length": len(raw_text),
            "warnings": warnings,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "steel_census.json").write_text(raw_text, encoding="utf-8")
    return census
