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


_STEEL_MARK_RE = re.compile(
    r"\b(?:SHS|CHS|RHS|UC|UB|SH|CH|SC)\s*[-/]?\s*[A-Z0-9]{0,8}\b", re.I)
_STEEL_PREFIX_RE = re.compile(r"^(SHS|CHS|RHS|UC|UB|SH|SC|CH)\w*", re.I)
_STEEL_CONTEXT_RE = re.compile(
    r"\b(STEEL|STEELWORK|STRUCTURAL STEEL|STEEL COLUMN|STEEL BEAM|"
    r"FRAMING|BRACING|TRUSS|UC|UB|SHS|CHS|RHS|NLB)\b", re.I)
_BEAM_CONTEXT_RE = re.compile(r"\b(BEAM|FRAMING|RAFTER|PURLIN|JOIST)\b", re.I)
_BRACE_CONTEXT_RE = re.compile(r"\b(BRAC|X[- ]?BRAC|DIAGONAL|TRUSS)\b", re.I)
_FLOOR_CONTEXT_RE = re.compile(r"\b(STEEL DECK|METAL DECK|COMPOSITE|BONDEK)\b", re.I)


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


def _symbol_kind(symbol: str, context: str = "") -> str:
    s = _normalize(symbol)
    if _BEAM_CONTEXT_RE.search(context):
        return "BEAM"
    if _BRACE_CONTEXT_RE.search(context):
        return "BRACING"
    if s.startswith(("UB",)):
        return "BEAM"
    return "COLUMN"


def _add_symbol(rows: dict, symbol: str, *, source: str, pages=None,
                context: str = "", confidence: float = 0.75) -> None:
    norm = _normalize(symbol)
    if not norm or not _STEEL_PREFIX_RE.match(norm):
        return
    item = rows.setdefault(norm, {
        "symbol": norm,
        "aliases": _aliases(norm),
        "member_type": _symbol_kind(norm, context),
        "section": norm,
        "expected_pages": [],
        "source": source,
        "confidence": confidence,
        "evidence": [],
    })
    if pages:
        existing = set(item.get("expected_pages") or [])
        item["expected_pages"] = sorted(existing | {int(p) for p in pages})
    if context:
        excerpt = " ".join(context.split())[:240]
        if excerpt not in item["evidence"]:
            item["evidence"].append(excerpt)
    item["confidence"] = max(float(item.get("confidence") or 0), confidence)


def _page_excerpt(text: str, match_start: int, width: int = 260) -> str:
    lo = max(0, match_start - width // 2)
    hi = min(len(text), match_start + width // 2)
    return " ".join(text[lo:hi].split())


def build_steel_census(pdf_path: str, analysis: DocAnalysis,
                       cfg: SlabV2Config, out_dir: Path) -> dict:
    """Build a compact steel census from document analysis and PDF text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols: dict[str, dict] = {}
    source_pages: list[dict] = []
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
            score = 0
            hits = []
            for m in _STEEL_CONTEXT_RE.finditer(text):
                score += 4
                hits.append(_page_excerpt(text, m.start()))
            mark_hits = []
            for m in _STEEL_MARK_RE.finditer(text):
                mark = _normalize(m.group(0))
                if len(mark) < 2:
                    continue
                mark_hits.append(mark)
                ctx = _page_excerpt(text, m.start())
                _add_symbol(symbols, mark, source="pdf_text_scan",
                            pages=[pi + 1], context=ctx, confidence=0.62)
            if mark_hits:
                score += min(18, len(mark_hits) * 2)
            if _BEAM_CONTEXT_RE.search(text):
                score += 3
            if _BRACE_CONTEXT_RE.search(text):
                score += 3
            if _FLOOR_CONTEXT_RE.search(text):
                score += 3
            if score:
                source_pages.append({
                    "page": pi + 1,
                    "score": score,
                    "source_type": "steel_context",
                    "reason": "steel context text or steel-like symbols found",
                    "excerpt": hits[0] if hits else " ".join(text.split())[:260],
                    "symbols": sorted(set(mark_hits))[:20],
                })
    finally:
        doc.close()

    source_pages = sorted(source_pages, key=lambda r: (-r["score"], r["page"]))
    symbol_rows = sorted(symbols.values(), key=lambda r: r["symbol"])
    steel_columns = [r for r in symbol_rows if r.get("member_type") == "COLUMN"]
    steel_beams = [r for r in symbol_rows if r.get("member_type") == "BEAM"]
    bracing = [r for r in symbol_rows if r.get("member_type") == "BRACING"]
    if symbol_rows:
        status = "steel_sources_planned"
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
        "steel_column_symbols": steel_columns,
        "steel_beam_symbols": steel_beams,
        "bracing_symbols": bracing,
        "steel_floor_regions": [],
        "source_pages": source_pages[:20],
        "expected_symbols": [r["symbol"] for r in symbol_rows],
        "zero_steel_reason": zero_reason,
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
    (out_dir / "steel_census.json").write_text(raw_text, encoding="utf-8")
    return census
