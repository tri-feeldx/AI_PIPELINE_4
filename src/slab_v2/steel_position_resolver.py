"""Document-level steel position resolver and detail linker.

The page-local steel detector only sees pages that are already extracted as
floor plans.  This resolver scans steel plan/marking-plan sources, links those
positions to detail/elevation/schedule evidence, and returns verified steel
members that can be injected into Ruby export without letting steel leak back
into the RC column detector.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import fitz
from shapely.geometry import box

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ColumnType, SlabV2Result, SteelMember
from src.slab_v2 import vector_extract
from src.slab_v2.steel_detector import (
    PT_TO_MM,
    _candidate_allowed_for,
    _candidate_polygons,
    _canonical_member_type,
    _collect_symbol_anchors,
    _merge_steel_types,
    _nearest_geometry,
    _normalize,
    _public_candidate,
    _render_overlay,
)


def _listify(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _source_pages(rows: list[dict]) -> list[int]:
    pages: list[int] = []
    for row in rows:
        try:
            page = int(row.get("page"))
        except Exception:
            continue
        if page not in pages:
            pages.append(page)
    return pages


def _simple_scale(page: fitz.Page) -> float | None:
    text = page.get_text("text") or ""
    m = re.search(r"\bSCALE\s*[:=]?\s*1\s*[:/]\s*(\d{2,4})\b", text, re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _profile_agreement(meta: dict, page_number: int) -> tuple[bool, list[str]]:
    evidence = []
    source_types = set(str(x) for x in meta.get("source_types") or [])
    detail_pages = set(int(x) for x in meta.get("detail_pages") or [])
    source_pages = set(int(x) for x in meta.get("source_pages") or [])
    expected_pages = set(int(x) for x in meta.get("expected_pages") or [])
    if detail_pages:
        evidence.append(f"profile/detail pages {sorted(detail_pages)}")
    if source_types & {"elevation", "section", "detail", "schedule"}:
        evidence.append(
            "profile source types "
            + ", ".join(sorted(source_types & {"elevation", "section", "detail", "schedule"})))
    if page_number in source_pages or page_number in expected_pages:
        evidence.append(f"position/source page agreement page {page_number}")
    return bool(evidence), evidence


_STEEL_LEVEL_TITLE_RE = re.compile(
    r"\bLEVEL\s*0?(\d+)\b[^\n\r]{0,120}?"
    r"\b(?:STEEL\s+MARKING|STEEL\s+FRAMING|STEELWORK|MARKING\s+PLAN|"
    r"FRAMING\s+PLAN|OUTLINE\s+PLAN|PLAN)\b",
    re.I,
)
_STEEL_ROOF_TITLE_RE = re.compile(
    r"\b(?:UPPER\s+|LOWER\s+)?ROOF\b[^\n\r]{0,120}?"
    r"\b(?:STEELWORK\s+MARKING|STEEL\s+MARKING|MARKING\s+PLAN|"
    r"STEELWORK|PLAN)\b",
    re.I,
)
_LEVEL_TOKEN_RE = re.compile(r"\bLEVEL\s*0?(\d+)\b|\bROOF\b", re.I)
_DETAIL_SOURCE_TYPES = {"elevation", "section", "detail", "schedule"}
_PROFILE_STEEL_MARK_RE = re.compile(
    r"\b(?:SHS|CHS|RHS|PFC|RFB|UC|UB|SH|CH|SC|PF|RB|BT|CT|TF|LA|EA|UA|PFB|D)\s*[-/]?\s*"
    r"(?:\d+[A-Z0-9*]*|[A-Z]\d+[A-Z0-9*]*|\*)\b",
    re.I,
)


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.upper())
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _level_name(raw_num: str) -> str:
    try:
        return f"LEVEL {int(raw_num):02d}"
    except Exception:
        return f"LEVEL {raw_num}"


def _level_hints_from_text(text: str, *, prefer_title: bool = True) -> list[str]:
    """Extract level hints from a steel source page without using geometry.

    Marking/framing plan titles are the strongest signal because detail sheets
    can mention many levels in one view.  The fallback is intentionally small:
    it keeps steel assignable while still leaving ambiguous multi-level pages
    in review when no title-like source exists.
    """
    text = str(text or "")
    hints: list[str] = []
    if prefer_title:
        if _STEEL_ROOF_TITLE_RE.search(text):
            hints.append("ROOF")
        for m in _STEEL_LEVEL_TITLE_RE.finditer(text):
            hints.append(_level_name(m.group(1)))
        if hints:
            return _dedupe_text(hints)

    compact = text[:2500]
    for m in _LEVEL_TOKEN_RE.finditer(compact):
        if m.group(1):
            hints.append(_level_name(m.group(1)))
        else:
            hints.append("ROOF")
    return _dedupe_text(hints)


def _source_row_for_page(rows: list[dict], page_number: int) -> dict:
    for row in rows:
        try:
            if int(row.get("page")) == int(page_number):
                return row
        except Exception:
            continue
    return {}


def _page_level_hints(page: fitz.Page, source_row: dict) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    hints: list[str] = []

    for key in ("level_hints", "levels", "level_range"):
        for value in _listify(source_row.get(key)):
            if value:
                hints.append(str(value))
                evidence.append(f"source row {key}: {value}")

    source_text = "\n".join(
        str(source_row.get(key) or "")
        for key in ("title", "excerpt", "reason", "source_type")
    )
    source_hints = _level_hints_from_text(source_text, prefer_title=False)
    if len(source_hints) == 1:
        hints.extend(source_hints)
        evidence.append("source row text level hint")
    elif source_hints:
        evidence.append(
            "ambiguous source row level hints: " + ", ".join(source_hints[:8]))

    page_text = page.get_text("text") or ""
    title_hints = _level_hints_from_text(page_text, prefer_title=True)
    if title_hints:
        hints.extend(title_hints)
        evidence.append("sheet title level hint")
    elif not hints:
        fallback_hints = _level_hints_from_text(page_text, prefer_title=False)
        if len(fallback_hints) == 1:
            hints.extend(fallback_hints)
            evidence.append("single page-level fallback hint")
        elif fallback_hints:
            evidence.append(
                "ambiguous page-level hints: " + ", ".join(fallback_hints[:8]))

    return _dedupe_text(hints), _dedupe_text(evidence)


def _anchor_placeholder(anchor: dict, scale: float, member_type: str):
    if member_type != "COLUMN":
        return None
    cx, cy = anchor["center"]
    side_mm = 180.0
    half_pt = side_mm / (PT_TO_MM * scale) / 2.0
    return box(cx - half_pt, cy - half_pt, cx + half_pt, cy + half_pt)


def _member_level_hints(meta: dict, page_hints: list[str] | None = None) -> list[str]:
    levels: list[str] = []
    for value in _listify(page_hints):
        if value:
            levels.append(str(value))
    if levels:
        # Plan/marking sheet title is the authority for x/y position level.
        # Detail/elevation pages often list many datums in one view; those
        # verify profile/range but must not spray one plan member across all
        # levels.
        return _dedupe_text(levels)
    for key in ("level_range", "levels"):
        for value in _listify(meta.get(key)):
            if value:
                levels.append(str(value))
    return _dedupe_text(levels)


def _normalize_level(value: str | None) -> str:
    text = str(value or "").upper()
    if re.search(r"\bROOF\b", text):
        return "ROOF"
    m = re.search(r"\b(?:LEVEL|L)\s*0?(\d{1,2})\b", text)
    if m:
        return _level_name(m.group(1))
    return str(value or "").strip().upper()


def _profile_level_range(meta: dict) -> list[str]:
    levels: list[str] = []
    for key in ("level_range", "levels"):
        for value in _listify(meta.get(key)):
            if value:
                levels.append(str(value))
    return _dedupe_text(levels)


def _resolve_level_assignment(position_hints: list[str],
                              profile_levels: list[str]) -> tuple[str, str, str]:
    """Return final_level, status, reason.

    Plan/marking page hints are x/y authority.  Profile/detail levels are only
    compatibility evidence; they must never override the plan level.
    """
    position_norm = [_normalize_level(v) for v in position_hints if v]
    position_norm = _dedupe_text([v for v in position_norm if v])
    profile_norm = [_normalize_level(v) for v in profile_levels if v]
    profile_norm = _dedupe_text([v for v in profile_norm if v])

    if len(position_norm) != 1:
        if not position_norm:
            return "", "review", "missing position level from plan or marking sheet"
        return "", "review", "ambiguous position levels: " + ", ".join(position_norm)

    final_level = position_norm[0]
    if profile_norm and final_level not in profile_norm:
        return "", "review", (
            f"position level {final_level} conflicts with profile/detail "
            f"range {', '.join(profile_norm)}"
        )
    if profile_norm:
        return final_level, "verified", (
            f"position level {final_level} verified against profile/detail range")
    return final_level, "verified", (
        f"position level {final_level} from plan/marking sheet; no profile "
        "level constraint")


def _role_key(member_type: str | None) -> str:
    value = str(member_type or "UNKNOWN").upper()
    if value == "COLUMN":
        return "steel_column"
    if value == "BEAM":
        return "steel_beam"
    if value == "BRACING":
        return "bracing"
    if value == "FLOOR":
        return "steel_floor_deck"
    return "unknown"


def _build_level_census(rows: list[dict], report: dict) -> tuple[dict, list[dict], dict]:
    levels: dict[str, dict] = {}
    compact: dict[tuple[str, str, str], dict] = {}
    prevented: list[dict] = []

    for row in rows:
        level = row.get("final_level") or row.get("position_level") or "UNASSIGNED"
        symbol = str(row.get("symbol") or "UNKNOWN")
        role = row.get("role") or _role_key(row.get("member_type"))
        status = str(row.get("status") or "review")
        exported = bool(row.get("exported"))
        source_page = row.get("source_page")

        level_bucket = levels.setdefault(level, {
            "totals": {"expected": 0, "detected": 0, "exported": 0, "review": 0},
            "symbols": {},
            "warnings": [],
        })
        sym_bucket = level_bucket["symbols"].setdefault(symbol, {
            "role": role,
            "expected": 0,
            "detected": 0,
            "exported": 0,
            "review": 0,
            "source_pages": [],
            "warnings": [],
        })
        key = (level, symbol, role)
        compact_row = compact.setdefault(key, {
            "level": level,
            "symbol": symbol,
            "role": role,
            "expected": 0,
            "detected": 0,
            "exported": 0,
            "review": 0,
            "source_pages": [],
            "warnings": [],
        })

        for bucket in (level_bucket["totals"], sym_bucket, compact_row):
            bucket["expected"] += 1
            bucket["detected"] += 1
            if exported:
                bucket["exported"] += 1
            if not exported or status != "verified":
                bucket["review"] += 1

        if source_page and source_page not in sym_bucket["source_pages"]:
            sym_bucket["source_pages"].append(source_page)
        if source_page and source_page not in compact_row["source_pages"]:
            compact_row["source_pages"].append(source_page)

        reason = (
            row.get("review_reason")
            or row.get("reject_reason")
            or row.get("level_assignment_reason")
            or ""
        )
        if not exported and reason:
            warning = f"{symbol}: {reason}"
            sym_bucket["warnings"].append(warning)
            compact_row["warnings"].append(warning)
            if "conflict" in reason.lower():
                prevented.append({
                    "id": row.get("id"),
                    "symbol": symbol,
                    "source_page": source_page,
                    "reason": reason,
                    "position_level": row.get("position_level"),
                    "profile_level_range": row.get("profile_level_range", []),
                })

    compact_rows = sorted(
        compact.values(),
        key=lambda r: (str(r["level"]), str(r["role"]), str(r["symbol"])),
    )
    expected_vs_detected = {
        level: {
            **data["totals"],
            "missing": max(0, data["totals"]["expected"] - data["totals"]["detected"]),
            "symbols": data["symbols"],
            "warnings": data["warnings"],
        }
        for level, data in sorted(levels.items(), key=lambda item: str(item[0]))
    }
    census = {
        "status": report.get("status", "not_run"),
        "expected_source": "steel_position_candidates",
        "levels": expected_vs_detected,
        "counts_by_level_and_symbol": compact_rows,
        "prevented_wrong_level_exports": prevented,
    }
    return census, compact_rows, expected_vs_detected


def _diagnose_zero_or_low_steel(report: dict, steel_census: dict | None) -> str:
    census_reason = str((steel_census or {}).get("zero_or_low_steel_reason") or "").strip()
    if census_reason:
        return census_reason
    if not (steel_census or {}).get("expected_symbols"):
        if (steel_census or {}).get("source_pages") or (steel_census or {}).get("steel_source_views"):
            return "symbol_grammar_missing"
        return "no_steel_expected"
    if not (steel_census or {}).get("position_sources"):
        return "position_source_missing"

    review_rows = report.get("review_candidates") or []
    rejected_rows = report.get("rejected_candidates") or []
    reason_text = " ".join(
        str(row.get("reason") or row.get("reject_reason") or row.get("review_reason") or "")
        for row in [*review_rows, *rejected_rows]
    ).lower()
    if "conflict" in reason_text:
        return "level_conflict"
    if "profile" in reason_text or "detail" in reason_text:
        return "profile_detail_missing"
    if "geometry" in reason_text or "nearby position" in reason_text:
        return "geometry_verification_failed"
    if "reference-only" in reason_text or "reference only" in reason_text:
        return "reference_only"
    return "geometry_verification_failed" if review_rows or rejected_rows else ""


def _profile_symbols_from_text(text: str) -> list[dict]:
    symbols: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for m in _PROFILE_STEEL_MARK_RE.finditer(text or ""):
        symbol = _normalize(m.group(0))
        if not symbol or (symbol, m.start()) in seen:
            continue
        seen.add((symbol, m.start()))
        lo = max(0, m.start() - 140)
        hi = min(len(text), m.end() + 140)
        context = " ".join(text[lo:hi].split())
        symbols.append({
            "symbol": symbol,
            "text": m.group(0),
            "char_start": m.start(),
            "context": context,
        })
    return symbols


def _word_level_anchors(page: fitz.Page) -> list[dict]:
    words = page.get_text("words") or []
    anchors: list[dict] = []
    for idx, word in enumerate(words):
        token = str(word[4] or "")
        if re.fullmatch(r"LEVEL", token, re.I) and idx + 1 < len(words):
            nxt = str(words[idx + 1][4] or "")
            if re.fullmatch(r"0?\d+", nxt):
                bbox = (
                    min(float(word[0]), float(words[idx + 1][0])),
                    min(float(word[1]), float(words[idx + 1][1])),
                    max(float(word[2]), float(words[idx + 1][2])),
                    max(float(word[3]), float(words[idx + 1][3])),
                )
                anchors.append({
                    "level": _level_name(nxt),
                    "bbox": bbox,
                    "source": f"{token} {nxt}",
                })
        elif re.fullmatch(r"ROOF", token, re.I):
            anchors.append({
                "level": "ROOF",
                "bbox": tuple(float(v) for v in word[:4]),
                "source": token,
            })
    return anchors


def _render_profile_page_preview(page: fitz.Page, out_path: Path) -> None:
    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False)
    pix.save(str(out_path))


def _extract_steel_elevation_profiles(
    doc: fitz.Document,
    profile_sources: list[dict],
    out_dir: Path,
    cfg: SlabV2Config,
) -> tuple[list[dict], list[dict]]:
    """Extract profile/detail evidence from elevation/detail pages.

    These rows are intentionally non-geometric for plan placement.  They only
    give the linker stronger role/level/profile evidence.
    """
    detail_members: list[dict] = []
    profile_link_candidates: list[dict] = []
    for row in profile_sources:
        source_type = str(row.get("source_type") or "")
        if source_type not in _DETAIL_SOURCE_TYPES:
            continue
        try:
            page_number = int(row.get("page"))
        except Exception:
            continue
        if page_number < 1 or page_number > len(doc):
            continue
        page = doc[page_number - 1]
        text = page.get_text("text") or ""
        levels = _level_hints_from_text(text, prefer_title=False)
        level_anchors = _word_level_anchors(page)
        symbols = _profile_symbols_from_text(text)
        view = {
            "page": page_number,
            "source_type": source_type,
            "bbox": [float(page.rect.x0), float(page.rect.y0),
                     float(page.rect.x1), float(page.rect.y1)],
            "levels": levels,
            "level_anchors": level_anchors[:40],
            "symbol_count": len(symbols),
            "symbols": sorted({s["symbol"] for s in symbols})[:80],
            "confidence": 0.72 if symbols and levels else 0.55,
            "note": "detail/elevation/profile source; not used as plan x/y position",
        }
        if getattr(cfg, "debug_images", False):
            _render_profile_page_preview(
                page, out_dir / f"steel_elevation_views_p{page_number:02d}.png")
        (out_dir / f"steel_elevation_views_p{page_number:02d}.json").write_text(
            json.dumps(view, indent=2, ensure_ascii=False), encoding="utf-8")
        for sym in symbols:
            member = {
                "symbol": sym["symbol"],
                "aliases": [sym["symbol"]],
                "role": "STEEL_PROFILE_SOURCE",
                "member_type": "UNKNOWN",
                "source_page": page_number,
                "source_type": source_type,
                "level_range": levels,
                "section": sym["symbol"],
                "source_view_bbox": view["bbox"],
                "confidence": 0.78 if levels else 0.64,
                "evidence": [sym["context"]],
            }
            detail_members.append(member)
            profile_link_candidates.append({
                "symbol": sym["symbol"],
                "source_page": page_number,
                "source_type": source_type,
                "levels": levels,
                "context": sym["context"],
                "link_use": "role/profile/level-range only",
            })
    return detail_members, profile_link_candidates


def resolve_steel_positions(
    pdf_path: str,
    steel_census: dict | None,
    cfg: SlabV2Config,
    out_dir: Path,
    column_types: dict[str, ColumnType] | None = None,
) -> dict:
    """Resolve document-level steel positions from plan/marking pages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    steel_census = steel_census or {}
    steel_types, alias_map, steel_meta = _merge_steel_types(
        column_types, steel_census)
    position_sources = steel_census.get("position_sources") or []
    profile_sources = steel_census.get("profile_sources") or []
    reference_sources = steel_census.get("reference_sources") or []
    position_pages = _source_pages(position_sources)

    report = {
        "status": "steel_source_missing",
        "position_sources": position_sources,
        "profile_sources": profile_sources,
        "reference_sources": reference_sources,
        "verified_members": [],
        "review_candidates": [],
        "rejected_candidates": [],
        "zero_steel_reason": "",
        "zero_or_low_steel_reason": "",
        "warnings": [],
        "counts_by_role": {},
    }
    members: list[SteelMember] = []
    census_rows: list[dict] = []

    if not steel_types:
        report["zero_steel_reason"] = (
            "No steel symbols were available for the position resolver.")
        report["zero_or_low_steel_reason"] = _diagnose_zero_or_low_steel(
            report, steel_census)
        census, compact_rows, expected_vs_detected = _build_level_census(
            census_rows, report)
        report["steel_level_census"] = census
        report["counts_by_level_and_symbol"] = compact_rows
        report["expected_vs_detected_by_level"] = expected_vs_detected
        for name, payload in (
            ("steel_level_census.json", census),
            ("steel_counts_by_level_and_symbol.json", compact_rows),
            ("steel_expected_vs_detected_by_level.json", expected_vs_detected),
        ):
            (out_dir / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8")
        (out_dir / "steel_link_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"members": members, "report": report}
    if not position_pages:
        doc = fitz.open(pdf_path)
        try:
            detail_rows, link_rows = _extract_steel_elevation_profiles(
                doc, profile_sources, out_dir, cfg)
        finally:
            doc.close()
        report["steel_detail_members_extracted"] = len(detail_rows)
        report["profile_link_candidates"] = link_rows[:200]
        (out_dir / "steel_detail_members.json").write_text(
            json.dumps(detail_rows, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out_dir / "steel_profile_link_candidates.json").write_text(
            json.dumps(link_rows, indent=2, ensure_ascii=False),
            encoding="utf-8")
        report["status"] = "steel_detected_but_unverified"
        report["zero_steel_reason"] = (
            "Steel profile/detail evidence exists, but no steel plan or "
            "marking plan page was classified as a position source.")
        report["zero_or_low_steel_reason"] = "position_source_missing"
        census, compact_rows, expected_vs_detected = _build_level_census(
            census_rows, report)
        report["steel_level_census"] = census
        report["counts_by_level_and_symbol"] = compact_rows
        report["expected_vs_detected_by_level"] = expected_vs_detected
        for name, payload in (
            ("steel_level_census.json", census),
            ("steel_counts_by_level_and_symbol.json", compact_rows),
            ("steel_expected_vs_detected_by_level.json", expected_vs_detected),
        ):
            (out_dir / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8")
        (out_dir / "steel_link_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"members": members, "report": report}

    doc = fitz.open(pdf_path)
    try:
        detail_rows, link_rows = _extract_steel_elevation_profiles(
            doc, profile_sources, out_dir, cfg)
        for detail in detail_rows:
            symbol = str(detail.get("symbol") or "")
            meta = steel_meta.setdefault(symbol, {
                "symbol": symbol,
                "aliases": [symbol],
                "member_type": detail.get("member_type") or "COLUMN",
                "role": detail.get("role") or "STEEL_PROFILE_SOURCE",
                "section": detail.get("section") or symbol,
                "expected_pages": [],
                "source_pages": [],
                "detail_pages": [],
                "source_types": [],
                "levels": [],
                "evidence": [],
                "export_default": True,
            })
            if detail.get("source_page") not in meta.get("detail_pages", []):
                meta.setdefault("detail_pages", []).append(detail.get("source_page"))
            if detail.get("source_type") not in meta.get("source_types", []):
                meta.setdefault("source_types", []).append(detail.get("source_type"))
            for level in _listify(detail.get("level_range")):
                if level and level not in meta.setdefault("levels", []):
                    meta["levels"].append(level)
            for ev in _listify(detail.get("evidence"))[:3]:
                if ev and ev not in meta.setdefault("evidence", []):
                    meta["evidence"].append(str(ev))
        report["steel_detail_members_extracted"] = len(detail_rows)
        report["profile_link_candidates"] = link_rows[:200]
        (out_dir / "steel_detail_members.json").write_text(
            json.dumps(detail_rows, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out_dir / "steel_profile_link_candidates.json").write_text(
            json.dumps(link_rows, indent=2, ensure_ascii=False),
            encoding="utf-8")
        for page_number in position_pages:
            if page_number < 1 or page_number > len(doc):
                continue
            page = doc[page_number - 1]
            source_row = _source_row_for_page(position_sources, page_number)
            page_level_hints, page_level_evidence = _page_level_hints(page, source_row)
            if page_level_hints or page_level_evidence:
                report.setdefault("level_hint_sources", {})[str(page_number)] = {
                    "hints": page_level_hints,
                    "evidence": page_level_evidence,
                    "source_type": source_row.get("source_type", ""),
                }
            scale = _simple_scale(page) or 100.0
            paths, classes = vector_extract.extract_paths(page, cfg, page.rect)
            anchors = _collect_symbol_anchors(page, steel_types, alias_map)
            geoms = _candidate_polygons(paths, scale)
            page_candidates = []
            render_candidates = []
            page_members: list[SteelMember] = []
            used_geom_ids: set[str] = set()
            radius_pt = max(24.0, 750.0 / (PT_TO_MM * scale))

            for anchor in anchors:
                meta = steel_meta.get(anchor["symbol"], {})
                position_hints = _member_level_hints(meta, page_level_hints)
                profile_levels = _profile_level_range(meta)
                final_level, level_status, level_reason = _resolve_level_assignment(
                    position_hints, profile_levels)
                position_level = final_level or (
                    _normalize_level(position_hints[0]) if len(position_hints) == 1 else "")
                member_type = _canonical_member_type(
                    meta.get("member_type"), meta.get("role"))
                if member_type == "REVIEW_ONLY":
                    row = {
                        "id": f"steel_pos_anchor_{len(page_candidates)+1:04d}",
                        "symbol": anchor["symbol"],
                        "member_type": member_type,
                        "role": _role_key(member_type),
                        "status": "review",
                        "exported": False,
                        "position_level": position_level,
                        "profile_level_range": profile_levels,
                        "final_level": "",
                        "level_assignment_status": level_status,
                        "level_assignment_reason": level_reason,
                        "anchor": anchor,
                        "reject_reason": "reference-only steel source",
                        "source_page": page_number,
                    }
                    page_candidates.append(row)
                    census_rows.append(row)
                    report["review_candidates"].append(row)
                    continue

                profile_ok, profile_evidence = _profile_agreement(
                    meta, page_number)
                candidate_pool = [
                    row for row in geoms if _candidate_allowed_for(member_type, row)
                ]
                geom, distance = _nearest_geometry(anchor, candidate_pool, radius_pt)
                source = "position_symbol_near_vector_geometry"
                confidence = 0.0
                polygon = None
                geom_id = None
                if geom is not None and geom["id"] not in used_geom_ids:
                    used_geom_ids.add(geom["id"])
                    polygon = geom["polygon"]
                    geom_id = geom["id"]
                    confidence = 0.90 if distance <= radius_pt * 0.50 else 0.84
                    if member_type in {"BEAM", "BRACING"} and geom.get("geometry_kind") == "linear_member":
                        confidence = max(confidence, 0.88)
                elif member_type == "COLUMN" and profile_ok:
                    polygon = _anchor_placeholder(anchor, scale, member_type)
                    source = "position_symbol_with_profile_placeholder"
                    distance = math.inf
                    confidence = 0.86

                if polygon is None:
                    row = {
                        "id": f"steel_pos_anchor_{len(page_candidates)+1:04d}",
                        "symbol": anchor["symbol"],
                        "member_type": member_type,
                        "role": _role_key(member_type),
                        "status": "review",
                        "exported": False,
                        "source_page": page_number,
                        "position_level": position_level,
                        "profile_level_range": profile_levels,
                        "final_level": "",
                        "level_assignment_status": level_status,
                        "level_assignment_reason": level_reason,
                        "anchor": anchor,
                        "nearest_distance_pt": (
                            None if distance is None or math.isinf(distance)
                            else distance),
                        "profile_agreement": profile_ok,
                        "reject_reason": "no nearby position geometry",
                    }
                    page_candidates.append(row)
                    census_rows.append(row)
                    report["review_candidates"].append(row)
                    continue

                status = "verified" if (
                    profile_ok and confidence >= 0.85
                    and level_status == "verified"
                    and meta.get("export_default", True)
                ) else "review"
                review_reasons = []
                if not profile_ok:
                    review_reasons.append("missing profile/detail agreement")
                if confidence < 0.85:
                    review_reasons.append("below verification threshold")
                if level_status != "verified":
                    review_reasons.append(level_reason)
                if not meta.get("export_default", True):
                    review_reasons.append("symbol/source not exported by default")
                member = SteelMember(
                    id=(
                        f"doc_steel_{member_type.lower()}_p{page_number:02d}_"
                        f"{len(members)+len(page_members)+1:04d}"
                    ),
                    symbol=anchor["symbol"],
                    member_type=member_type,
                    polygon=polygon,
                    section=str(meta.get("section") or anchor["symbol"]),
                    source=source,
                    confidence=confidence,
                    status=status,
                    nearby_text=[anchor.get("text", "")],
                    evidence=[
                        f"position source page {page_number}",
                        f"steel role {member_type}",
                        f"symbol anchor {anchor.get('text', '')}",
                        *profile_evidence,
                        *[str(e) for e in meta.get("evidence", [])[:3]],
                    ],
                )
                setattr(member, "source_page", page_number)
                setattr(member, "source_scale", scale)
                setattr(member, "level_hints", position_hints)
                setattr(member, "level_hint_evidence", page_level_evidence)
                setattr(member, "position_level", position_level)
                setattr(member, "profile_level_range", profile_levels)
                # Level comes from the plan/marking position source.  Missing
                # profile/detail agreement may keep the member in review, but
                # it should not erase a drawable candidate before contract
                # reconciliation can apply the expected count.
                setattr(member, "final_level", final_level if level_status == "verified" else "")
                setattr(member, "level_assignment_status", level_status)
                setattr(member, "level_assignment_reason", level_reason)
                setattr(member, "geometry_id", geom_id)
                census_row = {
                    "id": member.id,
                    "symbol": member.symbol,
                    "member_type": member.member_type,
                    "role": _role_key(member.member_type),
                    "source_page": page_number,
                    "status": status,
                    "exported": status == "verified",
                    "confidence": member.confidence,
                    "position_level": position_level,
                    "profile_level_range": profile_levels,
                    "final_level": getattr(member, "final_level", ""),
                    "level_assignment_status": level_status,
                    "level_assignment_reason": level_reason,
                    "review_reason": "; ".join(review_reasons),
                }
                census_rows.append(census_row)
                members.append(member)
                page_members.append(member)
                if status == "verified":
                    report["verified_members"].append({
                        "id": member.id,
                        "symbol": member.symbol,
                        "member_type": member.member_type,
                        "source_page": page_number,
                        "section": member.section,
                        "confidence": member.confidence,
                        "source": member.source,
                        "position_level": getattr(member, "position_level", ""),
                        "profile_level_range": getattr(member, "profile_level_range", []),
                        "final_level": getattr(member, "final_level", ""),
                        "level_assignment_status": getattr(
                            member, "level_assignment_status", ""),
                        "level_assignment_reason": getattr(
                            member, "level_assignment_reason", ""),
                        "level_hints": getattr(member, "level_hints", []),
                        "level_hint_evidence": getattr(member, "level_hint_evidence", []),
                        "evidence": member.evidence,
                    })
                    report["counts_by_role"][member.member_type] = (
                        report["counts_by_role"].get(member.member_type, 0) + 1)
                else:
                    report["review_candidates"].append({
                        "id": member.id,
                        "symbol": member.symbol,
                        "member_type": member.member_type,
                        "source_page": page_number,
                        "confidence": member.confidence,
                        "position_level": position_level,
                        "profile_level_range": profile_levels,
                        "final_level": getattr(member, "final_level", ""),
                        "level_assignment_status": level_status,
                        "level_assignment_reason": level_reason,
                        "reason": "; ".join(review_reasons)
                        or "missing profile agreement or below verification threshold",
                    })

                public_geom = _public_candidate({"polygon": polygon, "id": geom_id or member.id})
                page_candidates.append({
                    **public_geom,
                    "symbol": anchor["symbol"],
                    "member_type": member_type,
                    "status": status,
                    "confidence": confidence,
                    "source": source,
                    "source_page": page_number,
                    "anchor": anchor,
                    "profile_agreement": profile_ok,
                    "profile_evidence": profile_evidence,
                    "position_level": position_level,
                    "profile_level_range": profile_levels,
                    "final_level": getattr(member, "final_level", ""),
                    "level_assignment_status": level_status,
                    "level_assignment_reason": level_reason,
                    "level_hints": getattr(member, "level_hints", []),
                    "level_hint_evidence": getattr(member, "level_hint_evidence", []),
                })
                render_candidates.append({
                    "id": geom_id or member.id,
                    "polygon": polygon,
                    "symbol": anchor["symbol"],
                    "member_type": member_type,
                    "status": status,
                })

            tag = f"p{page_number:02d}"
            (out_dir / f"steel_position_candidates_{tag}.json").write_text(
                json.dumps(page_candidates, indent=2, ensure_ascii=False),
                encoding="utf-8")
            if getattr(cfg, "debug_images", False):
                _render_overlay(
                    page, out_dir, anchors,
                    render_candidates,
                    page_members, f"steel_position_candidates_{tag}.png")
    finally:
        doc.close()

    if members:
        report["status"] = "verified_steel"
        report["zero_steel_reason"] = ""
        report["zero_or_low_steel_reason"] = (
            _diagnose_zero_or_low_steel(report, steel_census)
            if report.get("review_candidates") or report.get("rejected_candidates")
            else ""
        )
    else:
        report["status"] = "steel_detected_but_unverified"
        report["zero_steel_reason"] = (
            "Steel position sources were scanned, but no candidate passed "
            "position + symbol + profile/detail verification.")
        report["zero_or_low_steel_reason"] = _diagnose_zero_or_low_steel(
            report, steel_census)
    level_hint_counts: dict[str, int] = {}
    final_level_counts: dict[str, int] = {}
    for member in members:
        hints = getattr(member, "level_hints", []) or ["unassigned"]
        for hint in hints:
            level_hint_counts[str(hint)] = level_hint_counts.get(str(hint), 0) + 1
        final_level = getattr(member, "final_level", "") or "unassigned"
        final_level_counts[str(final_level)] = (
            final_level_counts.get(str(final_level), 0) + 1)
    report["member_level_hint_counts"] = dict(sorted(level_hint_counts.items()))
    report["member_final_level_counts"] = dict(sorted(final_level_counts.items()))
    census, compact_rows, expected_vs_detected = _build_level_census(
        census_rows, report)
    report["steel_level_census"] = census
    report["counts_by_level_and_symbol"] = compact_rows
    report["expected_vs_detected_by_level"] = expected_vs_detected
    report["prevented_wrong_level_exports"] = census.get(
        "prevented_wrong_level_exports", [])
    for name, payload in (
        ("steel_level_census.json", census),
        ("steel_counts_by_level_and_symbol.json", compact_rows),
        ("steel_expected_vs_detected_by_level.json", expected_vs_detected),
    ):
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
    (out_dir / "steel_link_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"members": members, "report": report}


def steel_only_result(
    page_index: int,
    scale: float,
    members: list[SteelMember],
    debug_dir: Path,
    report: dict,
) -> SlabV2Result:
    result = SlabV2Result(page_index=page_index, debug_dir=str(debug_dir))
    result.scale = scale
    result.steel_members = list(members)
    result.steel_readiness = {
        "status": "verified_steel" if members else "steel_detected_but_unverified",
        "expected_symbols": sorted({m.symbol for m in members}),
        "expected_count": len({m.symbol for m in members}),
        "verified_count": len(members),
        "review_count": len(report.get("review_candidates", []) or []),
        "rejected_count": len(report.get("rejected_candidates", []) or []),
        "source_pages": sorted({int(getattr(m, "source_page", page_index + 1)) for m in members}),
        "counts_by_level": dict(report.get("member_final_level_counts", {}) or {}),
        "counts_by_level_and_symbol": report.get("counts_by_level_and_symbol", []),
        "expected_vs_detected_by_level": report.get(
            "expected_vs_detected_by_level", {}),
        "prevented_wrong_level_exports": report.get(
            "prevented_wrong_level_exports", []),
        "zero_steel_reason": report.get("zero_steel_reason", ""),
        "zero_or_low_steel_reason": report.get("zero_or_low_steel_reason", ""),
        "export_policy": "verified_only",
        "document_level_linker": True,
    }
    # A steel-only synthetic result has no slab geometry by design. Mark the
    # slab side as verified so model readiness is decided by the linked steel
    # evidence, not by the absence of concrete geometry on a marking/detail page.
    result.slab_readiness = {"status": "verified", "steel_only": True}
    result.wall_readiness = {"status": "not_required"}
    result.column_detection_report = {"status": "not_required"}
    result.opening_report = {}
    return result
