"""Drawing contract and reconciliation audit.

The contract is the semantic/count expectation for a drawing package.  It is
intentionally separate from geometry extraction: Gemini/document intelligence
can say what should exist, while the vector detectors still decide where valid
geometry exists.  This module turns those expectations into auditable counts so
bad pages do not fail silently.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_STEEL_RE = re.compile(
    r"^(?:UC|UB|SH|CH|SC|SHS|CHS|RHS|PFC|EA|UA|PF|RB|BT|CT|TF|LA|D)\b",
    re.I,
)

# Some detector outputs are plain dicts/dataclasses, but a few geometry-like
# objects cannot accept new attributes. Keep contract decisions in a sidecar for
# those objects so contract governance cannot crash the whole export.
_CONTRACT_META: dict[int, dict[str, Any]] = defaultdict(dict)


def _norm(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _level_key(value: Any) -> str:
    text = _norm(value).upper()
    if not text:
        return "UNKNOWN"
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"GROUND", "GROUND LEVEL", "GROUND FLOOR", "GF", "G/F"}:
        return "GROUND FLOOR"
    if text in {"FOUNDATION", "FOUNDATION LEVEL", "FOUNDATION FLOOR"}:
        return "FOUNDATION"
    m = re.search(r"\bLEVEL\s*0?(\d+)\b", text)
    if m:
        return f"LEVEL {int(m.group(1)):02d}"
    m = re.search(r"\bL\s*0?(\d+)\b", text)
    if m:
        return f"LEVEL {int(m.group(1)):02d}"
    if "ROOF" in text:
        return "ROOF"
    return text


def _symbol_key(value: Any) -> str:
    return _norm(value).upper().replace(" ", "")


def _obj_get(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    meta = _CONTRACT_META.get(id(obj), {})
    if key in meta:
        return meta[key]
    return getattr(obj, key, default)


def _obj_set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
        return
    try:
        setattr(obj, key, value)
    except Exception:
        _CONTRACT_META[id(obj)][key] = value


def _count(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except Exception:
        return default


def _is_steel_symbol(symbol: str, doc_analysis=None) -> bool:
    key = _symbol_key(symbol)
    if not key:
        return False
    try:
        ctype = (doc_analysis.column_types or {}).get(symbol)
        if not ctype:
            ctype = (doc_analysis.column_types or {}).get(key)
        if ctype and str(getattr(ctype, "material", "")).upper() == "STEEL":
            return True
    except Exception:
        pass
    return bool(_STEEL_RE.match(key))


def _result_level(storey: dict) -> str:
    return _level_key(
        storey.get("level_name")
        or storey.get("level_id")
        or getattr(storey.get("result"), "page_index", "")
    )


def _contract_row(
    subsystem: str,
    building: str,
    level: str,
    symbol: str = "",
    role: str = "",
    expected: int = 0,
    pages: list | None = None,
    source: str = "",
    evidence: list | None = None,
) -> dict:
    return {
        "subsystem": subsystem,
        "building": building or "UNKNOWN",
        "level": _level_key(level),
        "symbol": _symbol_key(symbol) if symbol else "",
        "role": _symbol_key(role) if role else "",
        "expected_count": _count(expected),
        "pages": sorted({int(p) for p in (pages or []) if str(p).strip().isdigit()}),
        "source": source,
        "evidence": list(evidence or []),
    }


_GEOMETRY_TITLE_RE = re.compile(
    r"\b(GENERAL\s+ARRANGEMENT|GA\s+PLAN|OUTLINE\s+PLAN|FLOOR\s+PLAN|"
    r"FRAMING\s+PLAN|STEELWORK\s+PLAN|STEEL\s+MARKING\s+PLAN|"
    r"MARKING\s+PLAN|STRUCTURAL\s+PLAN)\b",
    re.I,
)
_LOADING_TITLE_RE = re.compile(r"\b(LOADING\s+PLAN|LOAD\s+PLAN)\b", re.I)
_EVIDENCE_TITLE_RE = re.compile(
    r"\b(SCHEDULE|DETAIL|ELEVATION|SECTION|LEGEND|NOTES)\b",
    re.I,
)


def _storey_title(storey: dict) -> str:
    result = storey.get("result")
    role = getattr(result, "page_role_classification", {}) or {}
    return _norm(
        role.get("title")
        or storey.get("level_name")
        or storey.get("level_id")
        or ""
    )


def _storey_role(storey: dict) -> str:
    result = storey.get("result")
    role = getattr(result, "page_role_classification", {}) or {}
    return str(role.get("role") or "").lower()


def _authority_score(storey: dict, subsystem: str, contract: dict | None = None) -> float:
    """Rank pages within the same level for contract-count selection."""
    page = int(storey.get("page_idx", -999)) + 1
    title = _storey_title(storey).upper()
    role = _storey_role(storey)
    score = 0.0
    if role == "geometry_plan":
        score += 25.0
    if _GEOMETRY_TITLE_RE.search(title):
        score += 35.0
    if "GENERAL ARRANGEMENT" in title:
        score += 25.0
    if "OUTLINE PLAN" in title or "FLOOR PLAN" in title:
        score += 15.0
    if subsystem == "steel" and re.search(r"\b(STEEL|FRAMING|MARKING)\b", title):
        score += 20.0
    if _LOADING_TITLE_RE.search(title):
        score -= 35.0
    if role == "evidence_only" or _EVIDENCE_TITLE_RE.search(title):
        score -= 45.0
    page_auth = ((contract or {}).get("page_authority") or {}).get(str(page), {})
    if subsystem in set(page_auth.get("authority_for") or []):
        score += 60.0
    if subsystem in set(page_auth.get("evidence_for") or []):
        score -= 15.0
    return score


def _authority_pages_for_level(
    storeys_by_building: dict[str, list[dict]],
    building: str,
    level: str,
    subsystem: str,
    fallback_pages: list[int] | None = None,
) -> list[int]:
    rows = []
    for storey in (storeys_by_building or {}).get(building, []) or []:
        if _result_level(storey) != _level_key(level):
            continue
        page = int(storey.get("page_idx", -999)) + 1
        rows.append((page, _authority_score(storey, subsystem, None)))
    rows = [(p, s) for p, s in rows if s > -20.0]
    if not rows:
        return sorted({int(p) for p in (fallback_pages or [])})
    best = max(s for _, s in rows)
    # Keep pages close to the best authority. This supports split GA sheets
    # while preventing loading/evidence pages from suppressing richer GA pages.
    return sorted({p for p, s in rows if s >= best - 20.0})


def _build_level_page_sets(
    doc_analysis,
    storeys_by_building: dict[str, list[dict]],
    steel_census: dict | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    steel_census = steel_census or {}
    steel_position_pages = {
        int(r.get("page"))
        for r in steel_census.get("position_sources", []) or []
        if str(r.get("page", "")).isdigit()
    }
    page_authority: dict[str, dict] = {}
    level_sets: list[dict] = []
    by_page = {}
    for storeys in (storeys_by_building or {}).values():
        for storey in storeys or []:
            page = int(storey.get("page_idx", -999)) + 1
            by_page[page] = storey
    for building in getattr(doc_analysis, "buildings", []) or []:
        bname = getattr(building, "name", "") or "UNKNOWN"
        for floor in getattr(building, "floors", []) or []:
            level = _level_key(getattr(floor, "level_id", "") or getattr(floor, "level_name", ""))
            pages = [int(p) + 1 for p in (getattr(floor, "pages", []) or [])]
            page_rows = []
            for page in pages:
                storey = by_page.get(page, {})
                title = _storey_title(storey)
                role = _storey_role(storey)
                authority_for = []
                evidence_for = []
                for subsystem in ("slab", "rc_column", "wall", "opening"):
                    if page in _authority_pages_for_level(
                        storeys_by_building, bname, level, subsystem, pages):
                        authority_for.append(subsystem)
                    else:
                        evidence_for.append(subsystem)
                if page in steel_position_pages:
                    authority_for.append("steel")
                elif _authority_score(storey, "steel", None) > 10:
                    authority_for.append("steel")
                else:
                    evidence_for.append("steel")
                row = {
                    "page": page,
                    "title": title,
                    "role": role or "unknown",
                    "authority_for": sorted(set(authority_for)),
                    "evidence_for": sorted(set(evidence_for) - set(authority_for)),
                    "reason": (
                        "geometry/position authority"
                        if authority_for else "evidence/context only"
                    ),
                }
                page_authority[str(page)] = row
                page_rows.append(row)
            level_sets.append({
                "building": bname,
                "level": level,
                "pages": pages,
                "page_roles": page_rows,
                "authority_pages": {
                    subsystem: _authority_pages_for_level(
                        storeys_by_building, bname, level, subsystem, pages)
                    for subsystem in ("slab", "rc_column", "wall", "opening")
                } | {
                    "steel": sorted([
                        r["page"] for r in page_rows
                        if "steel" in r.get("authority_for", [])
                    ]),
                },
            })
    return level_sets, page_authority


def _append_counter_rows(rows: list[dict], *, subsystem: str, building: str,
                         level: str, counts: dict, pages: list | None,
                         source: str, role: str = "") -> None:
    for symbol, n in sorted((counts or {}).items()):
        n_int = _count(n)
        if n_int <= 0:
            continue
        rows.append(_contract_row(
            subsystem, building, level, symbol=symbol, role=role,
            expected=n_int, pages=pages, source=source))


def _steel_level_rows_from_storeys(
    storeys_by_building: dict[str, list[dict]],
) -> list[dict]:
    """Collect steel count rows that were produced by the position resolver.

    Steel source intelligence may contain a large grammar/schedule vocabulary
    with no level or position.  Those rows are useful for parsing, but they are
    not a drawing contract.  The per-result steel readiness rows are already
    tied to extracted storeys and therefore make a safer contract-count source.
    """
    rows: list[dict] = []
    for building, storeys in (storeys_by_building or {}).items():
        for storey in storeys or []:
            result = storey.get("result")
            if result is None:
                continue
            default_level = _result_level(storey)
            page = _page_number(result)
            readiness = getattr(result, "steel_readiness", {}) or {}
            for row in readiness.get("counts_by_level_and_symbol", []) or []:
                symbol = row.get("symbol") or row.get("mark") or ""
                if not symbol:
                    continue
                level = (
                    row.get("level")
                    or row.get("final_level")
                    or row.get("position_level")
                    or default_level
                )
                expected = (
                    row.get("expected")
                    or row.get("expected_count")
                    or row.get("detected")
                    or row.get("detected_count")
                    or row.get("exported")
                    or row.get("exported_count")
                    or 1
                )
                rows.append({
                    "building": building,
                    "level": level,
                    "symbol": symbol,
                    "role": row.get("role") or row.get("member_type") or "unknown",
                    "expected": expected,
                    "pages": row.get("pages") or row.get("source_pages") or [page],
                    "source": "steel_level_census_result",
                })
            for member in getattr(result, "steel_members", []) or []:
                symbol = _obj_get(member, "symbol", "")
                if not symbol:
                    continue
                rows.append({
                    "building": building,
                    "level": (
                        _obj_get(member, "final_level", "")
                        or _obj_get(member, "position_level", "")
                        or default_level
                    ),
                    "symbol": symbol,
                    "role": _obj_get(member, "member_type", "") or "unknown",
                    "expected": 1,
                    "pages": [_page_number(result)],
                    "source": "steel_member_result",
                })
    # Collapse duplicate per-member rows only after preserving aggregate rows.
    collapsed: dict[tuple[str, str, str, str, str], dict] = {}
    for row in rows:
        key = (
            row.get("source", ""),
            row.get("building", ""),
            _level_key(row.get("level")),
            _symbol_key(row.get("symbol")),
            _symbol_key(row.get("role")),
        )
        current = collapsed.setdefault(key, {
            **row,
            "expected": 0,
            "pages": [],
        })
        current["expected"] += _count(row.get("expected"), 1)
        current["pages"] = sorted({
            *[int(p) for p in current.get("pages", []) if str(p).isdigit()],
            *[int(p) for p in (row.get("pages") or []) if str(p).isdigit()],
        })
    return list(collapsed.values())


def build_drawing_contract(
    doc_analysis,
    storeys_by_building: dict[str, list[dict]],
    steel_census: dict | None = None,
    pdf_path: str = "",
) -> dict:
    """Build the expected drawing contract from existing semantic evidence.

    This first version reuses the already parsed Gemini/document intelligence
    and steel source intelligence.  It writes the same shape expected from a
    future dedicated Gemini "contract" prompt, so the reconciler and UI are
    stable while the semantic source improves.
    """
    rows: list[dict] = []
    levels: list[dict] = []
    steel_census = steel_census or {}
    level_page_sets, page_authority = _build_level_page_sets(
        doc_analysis, storeys_by_building, steel_census)
    authority_lookup = {
        (row.get("building"), row.get("level")): row.get("authority_pages", {})
        for row in level_page_sets
    }

    for building in getattr(doc_analysis, "buildings", []) or []:
        bname = getattr(building, "name", "") or "UNKNOWN"
        for floor in getattr(building, "floors", []) or []:
            level = _level_key(getattr(floor, "level_id", "") or getattr(floor, "level_name", ""))
            pages = [int(p) + 1 for p in (getattr(floor, "pages", []) or [])]
            auth = authority_lookup.get((bname, level), {})
            slab_pages = auth.get("slab") or pages
            rc_pages = auth.get("rc_column") or pages
            wall_pages = auth.get("wall") or pages
            levels.append({
                "building": bname,
                "level": level,
                "pages": pages,
                "authority_pages": auth,
                "titles": list(getattr(floor, "titles", []) or []),
                "ffl_m": getattr(floor, "ffl_m", None),
                "storey_height_mm": getattr(floor, "storey_height_mm", 0.0),
            })
            if pages:
                rows.append(_contract_row(
                    "slab", bname, level, role="floor_slab",
                    expected=1, pages=slab_pages,
                    source="document_floor_page_mapping"))

            rc_counts = {}
            for symbol, n in (getattr(floor, "columns", {}) or {}).items():
                if _is_steel_symbol(symbol, doc_analysis):
                    continue
                rc_counts[symbol] = _count(n)
            _append_counter_rows(
                rows, subsystem="rc_column", building=bname, level=level,
                counts=rc_counts, pages=rc_pages, source="document_column_census")

            _append_counter_rows(
                rows, subsystem="wall", building=bname, level=level,
                counts=getattr(floor, "walls", {}) or {}, pages=wall_pages,
                source="document_wall_census")

    # Per-floor column rows often contain recovered/backfilled symbols not
    # present on FloorInfo.  Merge them without letting steel leak into RC.
    seen = {
        (r["subsystem"], r["building"], r["level"], r["symbol"], r["role"])
        for r in rows
    }
    for entry in getattr(doc_analysis, "columns_per_floor", []) or []:
        bname = entry.get("building") or "UNKNOWN"
        level = _level_key(entry.get("level_id") or entry.get("level_name"))
        counts = {}
        for symbol, n in (entry.get("counts") or {}).items():
            if not _is_steel_symbol(symbol, doc_analysis):
                counts[symbol] = _count(n)
        for symbol, n in counts.items():
            key = ("rc_column", bname, level, _symbol_key(symbol), "")
            if key in seen:
                continue
            rows.append(_contract_row(
                "rc_column", bname, level, symbol=symbol, expected=n,
                source="document_columns_per_floor"))
            seen.add(key)

    # Steel contract comes from actual level-position resolver counts first.
    # Source-intelligence symbol grammars may include schedule/detail-only marks
    # without a drawable level/position; they must not become UNKNOWN-level
    # contract items when a per-level steel census is available.
    steel_rows = steel_census.get("steel_level_census") or {}
    compact = (
        _steel_level_rows_from_storeys(storeys_by_building)
        or steel_census.get("counts_by_level_and_symbol")
        or steel_census.get("steel_counts_by_level_and_symbol")
        or []
    )
    if isinstance(steel_rows, dict):
        compact = compact or steel_rows.get("counts_by_level_and_symbol", [])
    for row in compact or []:
        level = row.get("level") or row.get("final_level") or row.get("position_level") or "UNKNOWN"
        symbol = row.get("symbol") or row.get("mark") or ""
        role = row.get("role") or row.get("member_type") or "unknown"
        expected = (
            row.get("expected")
            or row.get("expected_count")
            or row.get("detected")
            or row.get("detected_count")
            or row.get("exported")
            or row.get("exported_count")
            or 0
        )
        if _count(expected) <= 0 and symbol:
            expected = 1
        if symbol:
            rows.append(_contract_row(
                "steel", row.get("building") or "UNKNOWN", level,
                symbol=symbol, role=role, expected=expected,
                pages=row.get("pages") or row.get("source_pages") or [],
                source="steel_level_census"))

    if not any(r["subsystem"] == "steel" for r in rows):
        for symbol in steel_census.get("expected_symbols", []) or []:
            rows.append(_contract_row(
                "steel", "UNKNOWN", "UNKNOWN", symbol=symbol,
                role="unknown", expected=1,
                source="steel_expected_symbols_no_level"))

    return {
        "schema_version": "drawing_contract_v3",
        "policy": "contract_count_v3_level_page_set",
        "source": "doc_analysis_plus_steel_intelligence",
        "pdf": str(pdf_path or ""),
        "levels": levels,
        "level_page_sets": level_page_sets,
        "page_authority": page_authority,
        "contract_items": rows,
        "exclusions": [
            {
                "rule": "steel_symbols_excluded_from_rc",
                "reason": "Steel marks must be reconciled by the steel subsystem, not RC columns.",
            },
            {
                "rule": "stair_only_not_opening",
                "reason": "Stair objects are context unless independent penetration/void/shaft evidence exists.",
            },
            {
                "rule": "dashed_or_reference_only_blocks_export",
                "reason": "Reference-only or dashed candidates require explicit drawable evidence.",
            },
        ],
    }


def _add_counter(counter: Counter, key: tuple, amount: int = 1) -> None:
    if len(key) == 4:
        building, level, symbol, role = key
        key = (
            building,
            _level_key(level),
            _symbol_key(symbol),
            _symbol_key(role),
        )
    counter[key] += int(amount or 0)


def _page_number(result: Any) -> int:
    return int(getattr(result, "page_index", -1) or -1) + 1


def _bounds(obj: Any) -> tuple | None:
    poly = _obj_get(obj, "polygon", None)
    if poly is None and isinstance(obj, dict):
        poly = obj.get("polygon_pdf") or obj.get("polygon")
    try:
        if poly is not None and not poly.is_empty:
            return tuple(float(v) for v in poly.bounds)
    except Exception:
        pass
    bbox = _obj_get(obj, "bbox", None)
    if bbox is None:
        bbox = _obj_get(obj, "anchor_bbox", None)
    if bbox and len(bbox) >= 4:
        return tuple(float(v) for v in bbox[:4])
    return None


def _center(bounds: tuple | None) -> tuple[float, float] | None:
    if not bounds:
        return None
    x0, y0, x1, y1 = bounds
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _anchor_center(obj: Any) -> tuple[float, float] | None:
    bbox = _obj_get(obj, "anchor_bbox", None)
    if bbox and len(bbox) >= 4:
        return _center(tuple(float(v) for v in bbox[:4]))
    return None


def _distance_score(obj: Any) -> float:
    anchor = _anchor_center(obj)
    centroid = _center(_bounds(obj))
    if not anchor or not centroid:
        return 0.0
    distance = math.hypot(anchor[0] - centroid[0], anchor[1] - centroid[1])
    return 1.0 / (1.0 + distance)


def _geometry_measure(obj: Any) -> float:
    poly = _obj_get(obj, "polygon", None)
    if poly is None and isinstance(obj, dict):
        poly = obj.get("polygon_pdf") or obj.get("polygon")
    try:
        if poly is not None and not poly.is_empty:
            return float(poly.area)
    except Exception:
        pass
    line = _obj_get(obj, "line", None) or _obj_get(obj, "centerline", None) or []
    total = 0.0
    try:
        for a, b in zip(line, line[1:]):
            total += math.hypot(float(a[0]) - float(b[0]),
                                float(a[1]) - float(b[1]))
    except Exception:
        pass
    return total


def _has_geometry(obj: Any) -> bool:
    return _geometry_measure(obj) > 0.0


def _candidate_text(obj: Any) -> str:
    chunks = [
        _obj_get(obj, "source", ""),
        _obj_get(obj, "status", ""),
        _obj_get(obj, "reject_reason", ""),
        _obj_get(obj, "mapping_status", ""),
        _obj_get(obj, "label", ""),
        _obj_get(obj, "symbol", ""),
        _obj_get(obj, "type", ""),
        _obj_get(obj, "opening_intent", ""),
    ]
    if isinstance(obj, dict):
        chunks.extend(str(v) for v in obj.values()
                      if isinstance(v, (str, int, float)))
    chunks.extend(str(v) for v in _obj_get(obj, "evidence", []) or [])
    chunks.extend(str(v) for v in _obj_get(obj, "evidence_ids", []) or [])
    chunks.extend(str(v) for v in _obj_get(obj, "object_roles", []) or [])
    return " ".join(str(c or "") for c in chunks).upper()


def _block_reason(obj: Any) -> str:
    if bool(_obj_get(obj, "is_dashed", False)):
        return "blocked_dashed"
    if bool(_obj_get(obj, "is_reference_only", False)):
        return "blocked_reference"
    text = _candidate_text(obj)
    if "DASH" in text or "HIDDEN" in text:
        return "blocked_dashed"
    if "REFERENCE" in text or "SCHEDULE_ONLY" in text:
        return "blocked_reference"
    if "REJECT" in text:
        return "blocked_rejected"
    if not _has_geometry(obj):
        return "missing_geometry"
    return ""


def _candidate_symbol(obj: Any, subsystem: str) -> str:
    if subsystem == "wall":
        return _symbol_key(_obj_get(obj, "label", ""))
    if subsystem == "steel":
        return _symbol_key(_obj_get(obj, "symbol", ""))
    if subsystem == "rc_column":
        return _symbol_key(_obj_get(obj, "symbol", ""))
    if subsystem == "opening":
        return _symbol_key(_obj_get(obj, "opening_intent", "")
                           or _obj_get(obj, "type", ""))
    return ""


def _candidate_role(obj: Any, subsystem: str) -> str:
    if subsystem == "steel":
        return _symbol_key(_obj_get(obj, "member_type", "") or "unknown")
    if subsystem == "opening":
        return "CUT_OPENING"
    if subsystem == "slab":
        return "FLOOR_SLAB"
    return ""


def _role_aliases(role: str) -> set[str]:
    key = _symbol_key(role)
    aliases = {key}
    if key.startswith("STEEL_"):
        aliases.add(key.replace("STEEL_", "", 1))
    if key in {"COLUMN", "BEAM", "BRACING", "FLOOR"}:
        aliases.add(f"STEEL_{key}")
    if key in {"CUT_OPENING", "OPENING"}:
        aliases.update({"VOID", "LIFT_SHAFT", "SLAB_PENETRATION"})
    return {a for a in aliases if a}


def _candidate_score(obj: Any, subsystem: str) -> float:
    score = 0.0
    score += float(_obj_get(obj, "confidence", 0.0) or 0.0) * 10.0
    score += _distance_score(obj) * 5.0
    geom = _geometry_measure(obj)
    if geom > 0:
        # Geometry size/length is useful as a tie-breaker, but cap it so a
        # giant false positive cannot beat symbol/level evidence by size alone.
        score += min(math.log1p(geom), 20.0)
    if _obj_get(obj, "labeled", False):
        score += 8.0
    if subsystem == "wall":
        score += float(_obj_get(obj, "l_mm", 0.0) or 0.0) / 1000.0
        if str(_obj_get(obj, "mapping_status", "")).lower() == "verified":
            score += 8.0
    if subsystem == "steel":
        if str(_obj_get(obj, "status", "")).lower() == "verified":
            score += 10.0
        if _obj_get(obj, "section", ""):
            score += 3.0
    return score


def _candidate_id(obj: Any, subsystem: str, index: int) -> str:
    for attr in ("candidate_id", "id", "label", "symbol"):
        value = _obj_get(obj, attr, "")
        if value:
            return str(value)
    return f"{subsystem}_{index:04d}"


def _candidate_registry_row(
    obj: Any,
    subsystem: str,
    index: int,
    *,
    building: str,
    level: str,
    page: int,
) -> dict:
    text = _candidate_text(obj)
    decision = _obj_get(obj, "contract_export_decision", "") or "undecided"
    return {
        "id": _candidate_id(obj, subsystem, index),
        "subsystem": subsystem,
        "building": building,
        "level": level,
        "page": page,
        "symbol": _candidate_symbol(obj, subsystem),
        "role": _candidate_role(obj, subsystem),
        "source": _obj_get(obj, "source", "") or _obj_get(obj, "status", ""),
        "nearest_text": _obj_get(obj, "label", "") or _obj_get(obj, "symbol", ""),
        "bounds": _bounds(obj),
        "has_geometry": _has_geometry(obj),
        "geometry_measure": _geometry_measure(obj),
        "score": _candidate_score(obj, subsystem),
        "is_dashed": ("DASH" in text or "HIDDEN" in text),
        "is_reference_only": (
            "REFERENCE" in text or "SCHEDULE_ONLY" in text
            or "REFERENCE_ONLY" in text
        ),
        "export_decision": decision,
        "reject_reason": (
            _obj_get(obj, "contract_export_reason", "")
            or _obj_get(obj, "reject_reason", "")
            or _block_reason(obj)
        ),
    }


def _result_candidates(result: Any, subsystem: str) -> list:
    if subsystem == "slab":
        return list(getattr(result, "slabs", []) or [])
    if subsystem == "rc_column":
        return list(getattr(result, "columns", []) or [])
    if subsystem == "wall":
        return list(getattr(result, "walls", []) or [])
    if subsystem == "steel":
        return list(getattr(result, "steel_members", []) or [])
    if subsystem == "opening":
        return list(getattr(result, "verified_cut_openings", []) or [])
    return []


def _set_result_candidates(result: Any, subsystem: str, items: list) -> None:
    if subsystem == "slab":
        result.slabs = items
    elif subsystem == "rc_column":
        result.columns = items
    elif subsystem == "wall":
        result.walls = items
    elif subsystem == "steel":
        result.steel_members = items
    elif subsystem == "opening":
        result.verified_cut_openings = items
        result.resolved_openings = items


def _contract_items_for_result(contract: dict, building: str, level: str,
                               page: int, subsystem: str) -> list[dict]:
    out = []
    target_level = _level_key(level)
    for item in contract.get("contract_items", []) or []:
        if item.get("subsystem") != subsystem:
            continue
        item_building = item.get("building") or "UNKNOWN"
        item_level = _level_key(item.get("level") or "UNKNOWN")
        pages = item.get("pages") or []
        if item_building not in {"UNKNOWN", building}:
            continue
        if item_level not in {"UNKNOWN", target_level}:
            continue
        if pages and page not in pages:
            continue
        out.append(item)
    return out


def _contract_items_for_level(contract: dict, building: str, level: str,
                              pages: set[int], subsystem: str) -> list[dict]:
    out = []
    target_level = _level_key(level)
    for item in contract.get("contract_items", []) or []:
        if item.get("subsystem") != subsystem:
            continue
        item_building = item.get("building") or "UNKNOWN"
        item_level = _level_key(item.get("level") or "UNKNOWN")
        item_pages = set(int(p) for p in item.get("pages") or [])
        if item_building not in {"UNKNOWN", building}:
            continue
        if item_level not in {"UNKNOWN", target_level}:
            continue
        if item_pages and pages and not (item_pages & pages):
            continue
        out.append(item)
    return out


def _item_matches_candidate(item: dict, obj: Any, subsystem: str) -> bool:
    symbol = item.get("symbol") or ""
    role = item.get("role") or ""
    cand_symbol = _candidate_symbol(obj, subsystem)
    cand_role = _candidate_role(obj, subsystem)
    if symbol and cand_symbol and symbol != cand_symbol:
        return False
    if symbol and not cand_symbol and subsystem not in {"slab", "opening"}:
        return False
    if role and cand_role and not (_role_aliases(role) & _role_aliases(cand_role)):
        return False
    return True


def _contract_page_allows(item: dict, page: int) -> bool:
    pages = set(int(p) for p in item.get("pages") or [])
    return not pages or int(page) in pages


def apply_contract_export_policy(
    contract: dict,
    storeys_by_building: dict[str, list[dict]],
) -> dict:
    """Select exactly the best contract candidates before Ruby export.

    This is intentionally count-driven, not verification-driven.  If the
    contract expects N items, we export up to N non-dashed/reference candidates
    with local geometry and hide extras.  Missing or blocked items stay visible
    in audit and keep the model in DEBUG via reconciliation.
    """
    decisions = {
        "schema_version": "contract_export_policy_v3",
        "policy": "contract_count_v3_level_page_set",
        "rows": [],
        "summary": defaultdict(lambda: Counter()),
    }
    subsystems = ("slab", "rc_column", "wall", "steel", "opening")

    def _export_all_detected_steel(records: list[dict]) -> bool:
        for rec in records or []:
            readiness = getattr(rec.get("result"), "steel_readiness", {}) or {}
            if bool(readiness.get("export_all_detected_steel")):
                return True
        return False

    for building, storeys in (storeys_by_building or {}).items():
        level_groups: dict[str, list[dict]] = defaultdict(list)
        for storey in storeys or []:
            result = storey.get("result")
            if result is None:
                continue
            result.candidate_registry = []
            result.contract_export_decisions = {
                "policy": "contract_count_v3_level_page_set",
                "building": building,
                "level": _result_level(storey),
                "page": _page_number(result),
                "subsystems": [],
            }
            level_groups[_result_level(storey)].append(storey)

        for level, level_storeys in level_groups.items():
            pages = {
                _page_number(storey.get("result"))
                for storey in level_storeys
                if storey.get("result") is not None
            }
            for subsystem in subsystems:
                items = _contract_items_for_level(
                    contract, building, level, pages, subsystem)
                records = []
                for storey in level_storeys:
                    result = storey.get("result")
                    if result is None:
                        continue
                    page = _page_number(result)
                    for idx, obj in enumerate(_result_candidates(result, subsystem)):
                        records.append({
                            "storey": storey,
                            "result": result,
                            "page": page,
                            "idx": idx,
                            "obj": obj,
                        })

                if not items:
                    if subsystem == "steel" and _export_all_detected_steel(records):
                        selected_by_result: dict[int, set[int]] = defaultdict(set)
                        blocked = []
                        exported = []
                        for rec in records:
                            reason = _block_reason(rec["obj"])
                            if reason:
                                blocked.append((rec, reason))
                                _obj_set(rec["obj"], "contract_export_decision", reason)
                                _obj_set(rec["obj"], "contract_export_reason",
                                         "debug steel export blocked by dashed/reference/rejected/no-geometry rule")
                            else:
                                exported.append(rec)
                                selected_by_result[id(rec["result"])].add(rec["idx"])
                                _obj_set(rec["obj"], "contract_export_decision", "exported")
                                _obj_set(rec["obj"], "contract_export_reason",
                                         "debug_export_all_detected_steel_no_contract_item")
                        row = {
                            "building": building,
                            "level": level,
                            "pages": sorted(pages),
                            "subsystem": subsystem,
                            "symbol": "*",
                            "role": "detected_steel_debug",
                            "expected_count": 0,
                            "eligible_count": len(exported),
                            "exported_count": len(exported),
                            "blocked_count": len(blocked),
                            "missing_count": 0,
                            "extra_hidden_count": 0,
                            "debug_export_all_detected": True,
                            "exported_candidate_ids": [
                                {
                                    "id": _candidate_id(rec["obj"], subsystem, rec["idx"]),
                                    "page": rec["page"],
                                }
                                for rec in exported
                            ],
                            "blocked_candidate_ids": [
                                {
                                    "id": _candidate_id(rec["obj"], subsystem, rec["idx"]),
                                    "page": rec["page"],
                                    "reason": reason,
                                }
                                for rec, reason in blocked
                            ],
                            "extra_hidden_candidate_ids": [],
                        }
                        decisions["rows"].append(row)
                        decisions["summary"][subsystem]["exported"] += len(exported)
                        decisions["summary"][subsystem]["blocked"] += len(blocked)
                        for storey in level_storeys:
                            result = storey.get("result")
                            if result is None:
                                continue
                            original = _result_candidates(result, subsystem)
                            selected_indices = selected_by_result.get(id(result), set())
                            _set_result_candidates(
                                result, subsystem,
                                [obj for idx, obj in enumerate(original)
                                 if idx in selected_indices],
                            )
                            result.candidate_registry.extend([
                                _candidate_registry_row(
                                    obj, subsystem, idx, building=building,
                                    level=level, page=_page_number(result))
                                for idx, obj in enumerate(original)
                            ])
                            if original:
                                result.contract_export_decisions["subsystems"].append({
                                    "subsystem": subsystem,
                                    "items": [row],
                                    "unmatched_hidden_count": 0,
                                    "unmatched_hidden_ids": [],
                                    "level_page_set_pages": sorted(pages),
                                    "reason": "debug_export_all_detected_steel_no_contract_item",
                                })
                        continue

                    for rec in records:
                        obj = rec["obj"]
                        _obj_set(obj, "contract_export_decision",
                                 "no_contract_hidden")
                        _obj_set(obj, "contract_export_reason",
                                 "no_contract_item_for_subsystem_level_page")
                    for storey in level_storeys:
                        result = storey.get("result")
                        if result is None:
                            continue
                        original = _result_candidates(result, subsystem)
                        result.candidate_registry.extend([
                            _candidate_registry_row(
                                obj, subsystem, idx, building=building,
                                level=level, page=_page_number(result))
                            for idx, obj in enumerate(original)
                        ])
                        _set_result_candidates(result, subsystem, [])
                        if original:
                            result.contract_export_decisions["subsystems"].append({
                                "subsystem": subsystem,
                                "items": [],
                                "unmatched_hidden_count": len(original),
                                "unmatched_hidden_ids": [
                                    _candidate_id(obj, subsystem, idx)
                                    for idx, obj in enumerate(original)
                                ],
                                "reason": "no_contract_item_for_subsystem_level_page",
                            })
                    continue

                used: set[tuple[int, int]] = set()
                selected_by_result: dict[int, set[int]] = defaultdict(set)
                rows_by_result: dict[int, list[dict]] = defaultdict(list)
                export_all_steel = (
                    subsystem == "steel" and _export_all_detected_steel(records)
                )

                for item in items:
                    expected = max(_count(item.get("expected_count")), 0)
                    candidates = [
                        rec for rec in records
                        if (id(rec["result"]), rec["idx"]) not in used
                        and _contract_page_allows(item, rec["page"])
                        and _item_matches_candidate(item, rec["obj"], subsystem)
                    ]
                    blocked = []
                    eligible = []
                    for rec in candidates:
                        reason = _block_reason(rec["obj"])
                        if reason:
                            blocked.append((rec, reason))
                        else:
                            eligible.append(rec)

                    eligible.sort(
                        key=lambda rec: (
                            _candidate_score(rec["obj"], subsystem)
                            + _authority_score(rec["storey"], subsystem, contract)
                        ),
                        reverse=True,
                    )
                    chosen = eligible if export_all_steel else eligible[:expected]
                    for rec in chosen:
                        used.add((id(rec["result"]), rec["idx"]))
                        selected_by_result[id(rec["result"])].add(rec["idx"])
                        _obj_set(rec["obj"], "contract_export_decision", "exported")
                        if export_all_steel:
                            _obj_set(rec["obj"], "contract_export_reason",
                                     "debug_export_all_detected_steel_geometry_candidate")
                        else:
                            _obj_set(rec["obj"], "contract_export_reason",
                                     "contract_selected_best_candidate_across_level_page_set")
                    extra = [] if export_all_steel else eligible[expected:]
                    for rec in extra:
                        _obj_set(rec["obj"], "contract_export_decision",
                                 "extra_hidden")
                        _obj_set(rec["obj"], "contract_export_reason",
                                 "more_candidates_than_contract_count_across_level")
                    for rec, reason in blocked:
                        _obj_set(rec["obj"], "contract_export_decision", reason)
                        _obj_set(rec["obj"], "contract_export_reason",
                                 "candidate blocked by dashed/reference/rejected/no-geometry rule")

                    row = {
                        "building": building,
                        "level": level,
                        "pages": sorted(pages),
                        "subsystem": subsystem,
                        "symbol": item.get("symbol", ""),
                        "role": item.get("role", ""),
                        "expected_count": expected,
                        "eligible_count": len(eligible),
                        "exported_count": len(chosen),
                        "blocked_count": len(blocked),
                        "missing_count": max(expected - len(chosen), 0),
                        "extra_hidden_count": len(extra),
                        "debug_export_all_detected": bool(export_all_steel),
                        "exported_candidate_ids": [
                            {
                                "id": _candidate_id(rec["obj"], subsystem, rec["idx"]),
                                "page": rec["page"],
                            }
                            for rec in chosen
                        ],
                        "blocked_candidate_ids": [
                            {
                                "id": _candidate_id(rec["obj"], subsystem, rec["idx"]),
                                "page": rec["page"],
                                "reason": reason,
                            }
                            for rec, reason in blocked
                        ],
                        "extra_hidden_candidate_ids": [
                            {
                                "id": _candidate_id(rec["obj"], subsystem, rec["idx"]),
                                "page": rec["page"],
                            }
                            for rec in extra
                        ],
                    }
                    decisions["rows"].append(row)
                    decisions["summary"][subsystem]["expected"] += expected
                    decisions["summary"][subsystem]["exported"] += len(chosen)
                    decisions["summary"][subsystem]["blocked"] += len(blocked)
                    decisions["summary"][subsystem]["missing"] += row["missing_count"]
                    decisions["summary"][subsystem]["extra_hidden"] += len(extra)
                    for rec in chosen + extra:
                        rows_by_result[id(rec["result"])].append(row)

                unmatched_debug_exported = []
                unmatched_debug_blocked = []
                for rec in records:
                    key = (id(rec["result"]), rec["idx"])
                    obj = rec["obj"]
                    if key not in used and not _obj_get(obj, "contract_export_decision", ""):
                        if export_all_steel:
                            reason = _block_reason(obj)
                            if reason:
                                unmatched_debug_blocked.append((rec, reason))
                                _obj_set(obj, "contract_export_decision", reason)
                                _obj_set(obj, "contract_export_reason",
                                         "debug steel export blocked by dashed/reference/rejected/no-geometry rule")
                            else:
                                unmatched_debug_exported.append(rec)
                                used.add(key)
                                selected_by_result[id(rec["result"])].add(rec["idx"])
                                _obj_set(obj, "contract_export_decision", "exported")
                                _obj_set(obj, "contract_export_reason",
                                         "debug_export_all_detected_steel_unmatched_contract")
                        else:
                            _obj_set(obj, "contract_export_decision", "extra_hidden")
                            _obj_set(obj, "contract_export_reason",
                                     "no_matching_contract_item_for_level_page_set")

                if unmatched_debug_exported or unmatched_debug_blocked:
                    row = {
                        "building": building,
                        "level": level,
                        "pages": sorted(pages),
                        "subsystem": subsystem,
                        "symbol": "*",
                        "role": "detected_steel_unmatched_contract_debug",
                        "expected_count": 0,
                        "eligible_count": len(unmatched_debug_exported),
                        "exported_count": len(unmatched_debug_exported),
                        "blocked_count": len(unmatched_debug_blocked),
                        "missing_count": 0,
                        "extra_hidden_count": 0,
                        "debug_export_all_detected": True,
                        "exported_candidate_ids": [
                            {
                                "id": _candidate_id(rec["obj"], subsystem, rec["idx"]),
                                "page": rec["page"],
                            }
                            for rec in unmatched_debug_exported
                        ],
                        "blocked_candidate_ids": [
                            {
                                "id": _candidate_id(rec["obj"], subsystem, rec["idx"]),
                                "page": rec["page"],
                                "reason": reason,
                            }
                            for rec, reason in unmatched_debug_blocked
                        ],
                        "extra_hidden_candidate_ids": [],
                    }
                    decisions["rows"].append(row)
                    decisions["summary"][subsystem]["exported"] += len(unmatched_debug_exported)
                    decisions["summary"][subsystem]["blocked"] += len(unmatched_debug_blocked)

                for storey in level_storeys:
                    result = storey.get("result")
                    if result is None:
                        continue
                    original = _result_candidates(result, subsystem)
                    selected_indices = selected_by_result.get(id(result), set())
                    _set_result_candidates(
                        result, subsystem,
                        [obj for idx, obj in enumerate(original)
                         if idx in selected_indices],
                    )
                    result.candidate_registry.extend([
                        _candidate_registry_row(
                            obj, subsystem, idx, building=building,
                            level=level, page=_page_number(result))
                        for idx, obj in enumerate(original)
                    ])
                    unmatched = [
                        (idx, obj) for idx, obj in enumerate(original)
                        if idx not in selected_indices
                    ]
                    subsystem_rows = [
                        row for row in decisions["rows"]
                        if row["subsystem"] == subsystem
                        and row["building"] == building
                        and row["level"] == level
                    ]
                    if subsystem_rows or unmatched:
                        result.contract_export_decisions["subsystems"].append({
                            "subsystem": subsystem,
                            "items": subsystem_rows,
                            "unmatched_hidden_count": len(unmatched),
                            "unmatched_hidden_ids": [
                                _candidate_id(obj, subsystem, idx)
                                for idx, obj in unmatched
                            ],
                            "level_page_set_pages": sorted(pages),
                        })

    decisions["summary"] = {
        key: dict(counter) for key, counter in decisions["summary"].items()
    }
    return decisions


def _actual_counts(storeys_by_building: dict[str, list[dict]]) -> dict[str, Counter]:
    detected = defaultdict(Counter)
    exported = defaultdict(Counter)
    blocked = defaultdict(Counter)

    for bname, storeys in (storeys_by_building or {}).items():
        for storey in storeys or []:
            result = storey.get("result")
            if result is None:
                continue
            level = _result_level(storey)
            page = getattr(result, "page_index", -1) + 1

            slab_count = len(getattr(result, "slabs", []) or [])
            if slab_count:
                key = (bname, level, "", "floor_slab")
                _add_counter(detected["slab"], key, slab_count)
                _add_counter(exported["slab"], key, slab_count)

            for col in getattr(result, "columns", []) or []:
                symbol = _symbol_key(getattr(col, "symbol", ""))
                if not symbol:
                    continue
                key = (bname, level, symbol, "")
                _add_counter(detected["rc_column"], key)
                status = _norm(getattr(col, "source", ""))
                confidence = float(getattr(col, "confidence", 0.0) or 0.0)
                if symbol == "C?" or status.lower() in {"review", "rejected"}:
                    _add_counter(blocked["rc_column"], key)
                else:
                    _add_counter(exported["rc_column"], key)

            col_report = getattr(result, "column_detection_report", {}) or {}
            for symbol, n in (col_report.get("missing") or {}).items():
                _add_counter(blocked["rc_column"], (bname, level, _symbol_key(symbol), ""), _count(n, 1))

            for wall in getattr(result, "walls", []) or []:
                symbol = _symbol_key(getattr(wall, "label", ""))
                if not symbol:
                    continue
                key = (bname, level, symbol, "")
                _add_counter(detected["wall"], key)
                if _norm(_obj_get(wall, "mapping_status", "")).lower() in {"review", "rejected"}:
                    _add_counter(blocked["wall"], key)
                else:
                    _add_counter(exported["wall"], key)

            wall_ready = getattr(result, "wall_readiness", {}) or {}
            for symbol, n in (wall_ready.get("missing") or {}).items():
                _add_counter(blocked["wall"], (bname, level, _symbol_key(symbol), ""), _count(n, 1))

            for steel in getattr(result, "steel_members", []) or []:
                symbol = _symbol_key(_obj_get(steel, "symbol", ""))
                role = _symbol_key(_obj_get(steel, "member_type", "") or "unknown")
                final_level = _level_key(_obj_get(steel, "final_level", "") or level)
                key = (bname, final_level, symbol, role)
                _add_counter(detected["steel"], key)
                if (_norm(_obj_get(steel, "status", "")).lower() == "verified"
                        or _obj_get(steel, "contract_export_decision", "") == "exported"):
                    _add_counter(exported["steel"], key)
                else:
                    _add_counter(blocked["steel"], key)

            steel_ready = getattr(result, "steel_readiness", {}) or {}
            for row in steel_ready.get("counts_by_level_and_symbol", []) or []:
                symbol = _symbol_key(row.get("symbol", ""))
                role = _symbol_key(row.get("role") or row.get("member_type") or "unknown")
                row_level = _level_key(row.get("level") or row.get("final_level") or level)
                key = (bname, row_level, symbol, role)
                _add_counter(detected["steel"], key, _count(row.get("detected") or row.get("detected_count")))
                _add_counter(exported["steel"], key, _count(row.get("exported") or row.get("exported_count")))
                review_n = _count(row.get("review") or row.get("review_count"))
                if review_n:
                    _add_counter(blocked["steel"], key, review_n)

            for opening in getattr(result, "verified_cut_openings", []) or []:
                intent = _symbol_key(_obj_get(opening, "opening_intent", "") or _obj_get(opening, "type", ""))
                key = (bname, level, intent, "cut_opening")
                _add_counter(detected["opening"], key)
                _add_counter(exported["opening"], key)

            for review in getattr(result, "opening_review_candidates", []) or []:
                intent = _symbol_key(_obj_get(review, "opening_intent", "") or _obj_get(review, "type", ""))
                _add_counter(blocked["opening"], (bname, level, intent, "cut_opening"))

            contract_decisions = getattr(result, "contract_export_decisions", {}) or {}
            for section in contract_decisions.get("subsystems", []) or []:
                subsystem = section.get("subsystem", "")
                for row in section.get("items", []) or []:
                    symbol = _symbol_key(row.get("symbol", ""))
                    role = _symbol_key(row.get("role", ""))
                    row_level = _level_key(row.get("level") or level)
                    row_building = row.get("building") or bname
                    key = (row_building, row_level, symbol, role)
                    blocked_n = _count(row.get("blocked_count"))
                    if blocked_n:
                        _add_counter(blocked[subsystem], key, blocked_n)

    return {"detected": detected, "exported": exported, "blocked": blocked}


def _matching_count(counter: Counter, subsystem: str, building: str, level: str,
                    symbol: str, role: str) -> int:
    level = _level_key(level)
    exact = counter[subsystem].get((building, level, symbol, role), 0)
    if exact:
        return exact
    # Allow unknown contract building to match any actual building.
    total = 0
    for (b, lv, sym, rl), n in counter[subsystem].items():
        if building not in {"UNKNOWN", b}:
            continue
        if level not in {"UNKNOWN", _level_key(lv)}:
            continue
        if symbol and symbol != sym:
            continue
        if role and not (_role_aliases(role) & _role_aliases(rl)):
            continue
        total += n
    return total


def reconcile_drawing_contract(
    contract: dict,
    storeys_by_building: dict[str, list[dict]],
) -> dict:
    counts = _actual_counts(storeys_by_building)
    rows: list[dict] = []
    by_subsystem = defaultdict(lambda: {
        "expected": 0, "detected": 0, "exported": 0,
        "missing": 0, "extra": 0, "blocked": 0,
        "status": "unknown",
    })

    seen_actual_keys: set[tuple] = set()
    for item in contract.get("contract_items", []) or []:
        subsystem = item.get("subsystem", "unknown")
        building = item.get("building") or "UNKNOWN"
        level = _level_key(item.get("level") or "UNKNOWN")
        symbol = item.get("symbol") or ""
        role = item.get("role") or ""
        expected = _count(item.get("expected_count"))
        detected = _matching_count(counts["detected"], subsystem, building, level, symbol, role)
        exported = _matching_count(counts["exported"], subsystem, building, level, symbol, role)
        blocked = _matching_count(counts["blocked"], subsystem, building, level, symbol, role)
        # Blocked/review/dashed candidates explain why we did not draw, but do
        # not fulfil the requested drawing count.
        missing = max(expected - exported, 0)
        extra = max(exported - expected, 0) if expected else 0
        if expected <= 0:
            status = "unknown"
        elif missing == 0 and extra == 0 and exported >= expected:
            status = "fulfilled"
        elif blocked:
            status = "blocked"
        elif exported > 0 or detected > 0:
            status = "partial"
        else:
            status = "missing"
        row = {
            **item,
            "level": level,
            "detected_count": detected,
            "exported_count": exported,
            "missing_count": missing,
            "extra_count": extra,
            "blocked_count": blocked,
            "status": status,
            "reason": _contract_reason(status, expected, detected, exported, blocked, missing, extra),
        }
        rows.append(row)
        seen_actual_keys.add((subsystem, building, level, symbol, role))
        agg = by_subsystem[subsystem]
        agg["expected"] += expected
        agg["detected"] += detected
        agg["exported"] += exported
        agg["missing"] += missing
        agg["extra"] += extra
        agg["blocked"] += blocked

    # Actual geometry that was exported but absent from contract is extra audit.
    for subsystem, counter in counts["exported"].items():
        for (building, level, symbol, role), exported in counter.items():
            if any(
                r["subsystem"] == subsystem
                and r["building"] in {building, "UNKNOWN"}
                and _level_key(r["level"]) in {_level_key(level), "UNKNOWN"}
                and (not r["symbol"] or r["symbol"] == symbol)
                and (not r["role"] or r["role"] == role)
                for r in rows
            ):
                continue
            row = {
                "subsystem": subsystem,
                "building": building,
                "level": level,
                "symbol": symbol,
                "role": role,
                "expected_count": 0,
                "detected_count": exported,
                "exported_count": exported,
                "missing_count": 0,
                "extra_count": exported,
                "blocked_count": 0,
                "status": "extra",
                "source": "geometry_without_contract",
                "reason": "Geometry was exported but no contract item expected it.",
            }
            rows.append(row)
            agg = by_subsystem[subsystem]
            agg["detected"] += exported
            agg["exported"] += exported
            agg["extra"] += exported

    critical = [
        r for r in rows
        if r.get("status") in {"missing", "partial", "blocked", "extra", "conflict"}
        and (r.get("expected_count", 0) or r.get("exported_count", 0))
    ]
    for subsystem, agg in by_subsystem.items():
        if agg["missing"] or agg["extra"]:
            agg["status"] = "partial"
        elif agg["blocked"]:
            agg["status"] = "blocked"
        elif agg["expected"] and agg["exported"] >= agg["expected"]:
            agg["status"] = "fulfilled"
        elif agg["expected"]:
            agg["status"] = "missing"
        else:
            agg["status"] = "unknown"

    status = "fulfilled" if not critical else "partial"
    if any(r.get("status") == "blocked" for r in critical):
        status = "blocked"
    if any(r.get("status") == "missing" for r in critical):
        status = "partial"

    return {
        "schema_version": "contract_reconciliation_v3",
        "policy": "contract_count_v3_level_page_set",
        "contract_status": status,
        "critical_unfulfilled_count": len(critical),
        "by_subsystem": dict(by_subsystem),
        "counts_by_level": rows,
        "missing_extra_blocked": critical,
        "reasons": [
            f"{r['subsystem']} {r.get('level')} {r.get('symbol') or r.get('role')}: {r['reason']}"
            for r in critical[:50]
        ],
    }


def _contract_reason(status: str, expected: int, detected: int, exported: int,
                     blocked: int, missing: int, extra: int) -> str:
    if status == "fulfilled":
        return "Expected count fulfilled by exported verified geometry."
    if status == "blocked":
        return (
            f"Expected {expected}; exported {exported}; blocked/review {blocked}. "
            "Geometry exists but failed export gates or is dashed/reference/review."
        )
    if status == "partial":
        return (
            f"Expected {expected}; detected {detected}; exported {exported}; "
            f"missing {missing}; blocked {blocked}."
        )
    if status == "missing":
        return f"Expected {expected}, but no exportable local geometry was found."
    if status == "extra":
        return f"Exported {extra} geometry item(s) not present in the contract."
    return "No reliable contract count was available."


def attach_contract_to_storeys(
    contract: dict,
    reconciliation: dict,
    storeys_by_building: dict[str, list[dict]],
) -> None:
    for storeys in (storeys_by_building or {}).values():
        for storey in storeys or []:
            result = storey.get("result")
            if result is None:
                continue
            result.drawing_contract = contract
            result.contract_reconciliation = reconciliation


def build_missing_contract(reason: str) -> tuple[dict, dict]:
    """Create an explicit missing-contract audit instead of failing silently."""
    contract = {
        "schema_version": "drawing_contract_v3",
        "policy": "contract_count_v3_level_page_set",
        "contract_status": "missing_contract",
        "contract_items": [],
        "levels": [],
        "warnings": [reason],
    }
    reconciliation = {
        "schema_version": "contract_reconciliation_v3",
        "policy": "contract_count_v3_level_page_set",
        "contract_status": "missing_contract",
        "critical_unfulfilled_count": 1,
        "by_subsystem": {},
        "counts_by_level": [],
        "missing_extra_blocked": [{
            "subsystem": "drawing_contract",
            "status": "missing_contract",
            "expected_count": 1,
            "detected_count": 0,
            "exported_count": 0,
            "missing_count": 1,
            "blocked_count": 0,
            "reason": reason,
        }],
        "reasons": [reason],
    }
    return contract, reconciliation


def write_contract_outputs(run_root: Path | str, contract: dict,
                           reconciliation: dict) -> dict:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    raw_text = (
        "Drawing Contract v3 / contract_count_v3_level_page_set\n"
        "Source: existing Gemini document analysis + steel source intelligence.\n"
        "Gemini is the semantic/count authority; vector detectors remain the geometry authority.\n"
        "Export invariant: expected N exports at most N best local non-dashed candidates across all authority pages in the same level page set; extras are hidden; missing/blocked keep DEBUG.\n"
    )
    outputs = {
        "drawing_contract_raw": root / "drawing_contract_raw.txt",
        "drawing_contract": root / "drawing_contract.json",
        "level_page_set_report": root / "level_page_set_report.json",
        "page_authority_report": root / "page_authority_report.json",
        "subsystem_authority_by_level": root / "subsystem_authority_by_level.json",
        "contract_reconciliation_report": root / "contract_reconciliation_report.json",
        "contract_counts_by_level": root / "contract_counts_by_level.json",
        "contract_missing_extra_blocked": root / "contract_missing_extra_blocked.json",
        "contract_export_decisions": root / "contract_export_decisions.json",
    }
    outputs["drawing_contract_raw"].write_text(raw_text, encoding="utf-8")
    outputs["drawing_contract"].write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    level_sets = contract.get("level_page_sets", []) or []
    outputs["level_page_set_report"].write_text(
        json.dumps(level_sets, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    outputs["page_authority_report"].write_text(
        json.dumps(contract.get("page_authority", {}) or {},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    authority_by_level = [
        {
            "building": row.get("building", "UNKNOWN"),
            "level": row.get("level", "UNKNOWN"),
            "pages": row.get("pages", []),
            "authority_pages": row.get("authority_pages", {}),
        }
        for row in level_sets
    ]
    outputs["subsystem_authority_by_level"].write_text(
        json.dumps(authority_by_level, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    outputs["contract_reconciliation_report"].write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    outputs["contract_counts_by_level"].write_text(
        json.dumps(reconciliation.get("counts_by_level", []),
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    outputs["contract_missing_extra_blocked"].write_text(
        json.dumps(reconciliation.get("missing_extra_blocked", []),
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    outputs["contract_export_decisions"].write_text(
        json.dumps(reconciliation.get("export_decisions", {}),
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return {key: str(path) for key, path in outputs.items()}


def write_candidate_registry_outputs(
    run_root: Path | str,
    storeys_by_building: dict[str, list[dict]],
) -> dict:
    """Write per-page candidate registries after contract decisions.

    These files are intentionally verbose: they are the audit trail for why a
    geometry object was exported, hidden as extra, blocked as dashed/reference,
    or excluded because no Gemini contract item existed for that subsystem.
    """
    root = Path(run_root)
    written: dict[str, str] = {}
    index_rows: list[dict] = []
    merged_by_level: dict[tuple[str, str], list[dict]] = defaultdict(list)
    pages_by_level: dict[tuple[str, str], set[int]] = defaultdict(set)
    for bname, storeys in (storeys_by_building or {}).items():
        for storey in storeys or []:
            result = storey.get("result")
            if result is None:
                continue
            page = _page_number(result)
            level = _result_level(storey)
            rows = list(getattr(result, "candidate_registry", []) or [])
            if not rows:
                continue
            merged_by_level[(bname, level)].extend(rows)
            pages_by_level[(bname, level)].add(page)
            page_dir = root / f"page_{page}"
            page_dir.mkdir(parents=True, exist_ok=True)
            out = page_dir / f"candidate_registry_p{page:02d}.json"
            payload = {
                "schema_version": "candidate_registry_v1",
                "policy": "contract_count_v3_level_page_set",
                "building": bname,
                "level": level,
                "page": page,
                "rows": rows,
            }
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
            written[f"page_{page}"] = str(out)
            index_rows.append({
                "building": bname,
                "level": level,
                "page": page,
                "candidate_count": len(rows),
                "path": str(out),
            })
    merged_dir = root / "merged_candidate_registry"
    merged_dir.mkdir(parents=True, exist_ok=True)
    for (bname, level), rows in sorted(merged_by_level.items()):
        safe_level = re.sub(r"[^A-Za-z0-9_]+", "_", level).strip("_") or "UNKNOWN"
        safe_building = re.sub(r"[^A-Za-z0-9_]+", "_", bname).strip("_") or "UNKNOWN"
        out = merged_dir / f"merged_candidate_registry_{safe_building}_{safe_level}.json"
        payload = {
            "schema_version": "merged_candidate_registry_v1",
            "policy": "contract_count_v3_level_page_set",
            "building": bname,
            "level": level,
            "pages": sorted(pages_by_level[(bname, level)]),
            "candidate_count": len(rows),
            "rows": rows,
        }
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        key = f"merged_{safe_building}_{safe_level}"
        written[key] = str(out)
        index_rows.append({
            "building": bname,
            "level": level,
            "page": "merged",
            "candidate_count": len(rows),
            "path": str(out),
        })
    index = root / "candidate_registry_index.json"
    index.write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    written["index"] = str(index)
    return written
