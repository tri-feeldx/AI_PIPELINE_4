"""Conservative steel detector for slab_v2.

Steel is intentionally separate from RC columns.  This module uses the
Gemini/doc census only as semantic input (which symbols are steel) and the
PDF vector/text geometry as authority.  Ambiguous items stay in review and
are not exported by default.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon, box

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ColumnType, SteelMember

PT_TO_MM = 25.4 / 72.0
_STEEL_PREFIX_RE = re.compile(
    r"^(SHS|CHS|RHS|PFC|RFB|UC|UB|SH|SC|CH|PF|RB|BT|CT|TF|LA|EA|UA|PFB|D)\w*",
    re.I,
)


@dataclass
class SteelDetectionResult:
    members: list[SteelMember] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    assignment: dict = field(default_factory=dict)
    readiness: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text or "").upper())


def _aliases(symbol: str) -> set[str]:
    norm = _normalize(symbol)
    out = {norm} if norm else set()
    m = re.match(r"^([A-Z]+)([0-9].*)$", norm)
    if m:
        out.add(m.group(1) + m.group(2))
    return out


def _is_steel_type(symbol: str, column_type: ColumnType) -> bool:
    material = str(getattr(column_type, "material", "") or "").upper()
    if material == "STEEL":
        return True
    return bool(_STEEL_PREFIX_RE.match(_normalize(symbol)))


def _steel_types(column_types: dict[str, ColumnType] | None) -> dict[str, ColumnType]:
    return {
        str(sym): ct for sym, ct in (column_types or {}).items()
        if _is_steel_type(str(sym), ct)
    }


def _listify(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _canonical_member_type(value, role: str = "") -> str:
    raw = _normalize(value or role or "COLUMN")
    role_raw = _normalize(role or "")
    combined = raw or role_raw
    if combined in {"BRACING", "STEELBRACING"}:
        return "BRACING"
    if combined in {"BEAM", "STEELBEAM", "PURLINGIRT", "FACADESTEEL"}:
        return "BEAM"
    if combined in {"FLOOR", "STEELFLOORDECK", "DIAPHRAGMREINFORCEMENT"}:
        return "FLOOR"
    if combined in {"REFERENCEONLY", "REVIEWONLY"}:
        return "REVIEW_ONLY"
    return "COLUMN"


def _census_items(steel_census: dict | None) -> list[dict]:
    if not steel_census:
        return []
    rows: list[dict] = []
    for key in ("steel_column_symbols", "steel_beam_symbols",
                "bracing_symbols", "steel_detail_members",
                "expected_symbols"):
        for item in _listify(steel_census.get(key)):
            if isinstance(item, dict):
                rows.append(item)
            elif item:
                rows.append({"symbol": str(item), "member_type": "COLUMN"})
    seen = set()
    out = []
    for item in rows:
        sym = _normalize(item.get("symbol") or item.get("mark") or item.get("id"))
        # Rows have already passed the steel source/census planner.  Keep the
        # known-prefix guard for fallback symbols, but do not discard
        # consultant-specific families learned from the PDF itself.
        source_ok = bool(
            item.get("source")
            or item.get("source_type")
            or item.get("source_types")
            or item.get("evidence")
        )
        if not sym or sym in seen or (not _STEEL_PREFIX_RE.match(sym) and not source_ok):
            continue
        seen.add(sym)
        member_type = _canonical_member_type(item.get("member_type"), item.get("role"))
        out.append({**item, "symbol": sym, "member_type": member_type})
    return out


def _steel_types_from_census(steel_census: dict | None) -> dict[str, ColumnType]:
    rows: dict[str, ColumnType] = {}
    for item in _census_items(steel_census):
        sym = item["symbol"]
        rows[sym] = ColumnType(
            symbol=sym,
            width_mm=0.0,
            depth_mm=0.0,
            count_total=0,
            material="STEEL",
        )
        setattr(rows[sym], "section", item.get("section") or sym)
    return rows


def _steel_meta_from_census(steel_census: dict | None) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for item in _census_items(steel_census):
        sym = item["symbol"]
        row = meta.setdefault(sym, {
            "member_type": _canonical_member_type(item.get("member_type"), item.get("role")),
            "section": item.get("section") or sym,
            "source": item.get("source") or "steel_census",
            "evidence": [],
            "expected_pages": item.get("expected_pages") or [],
            "source_pages": [],
            "source_types": [],
            "detail_pages": [],
            "role": item.get("role") or "",
            "export_default": item.get("export_default", True),
        })
        mt = _canonical_member_type(item.get("member_type"), item.get("role"))
        if row.get("member_type") == "COLUMN" and mt not in {"COLUMN", "REVIEW_ONLY"}:
            row["member_type"] = mt
        row["section"] = row.get("section") or item.get("section") or sym
        row["role"] = row.get("role") or item.get("role") or ""
        row["export_default"] = bool(row.get("export_default", True)) and bool(
            item.get("export_default", True))
        row["evidence"].extend(str(e) for e in _listify(item.get("evidence")) if e)
        for key in ("expected_pages", "source_pages", "detail_pages"):
            values = set(row.get(key) or [])
            raw = item.get(key) or ([item.get("source_page")] if key == "detail_pages" else [])
            for value in _listify(raw):
                try:
                    values.add(int(value))
                except (TypeError, ValueError):
                    pass
            row[key] = sorted(values)
        for value in _listify(item.get("source_types") or item.get("source_type")):
            if value and value not in row["source_types"]:
                row["source_types"].append(str(value))
    if steel_census:
        for source in _listify(steel_census.get("steel_source_views") or steel_census.get("source_pages")):
            if not isinstance(source, dict):
                continue
            source_page = source.get("page")
            source_type = source.get("source_type")
            for sym in _listify(source.get("symbols")):
                ns = _normalize(sym)
                if ns in meta:
                    try:
                        meta[ns].setdefault("source_pages", [])
                        if int(source_page) not in meta[ns]["source_pages"]:
                            meta[ns]["source_pages"].append(int(source_page))
                    except (TypeError, ValueError):
                        pass
                    if source_type and source_type not in meta[ns].setdefault("source_types", []):
                        meta[ns]["source_types"].append(str(source_type))
    return meta


def _alias_map_from_census(steel_census: dict | None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for item in _census_items(steel_census):
        sym = item["symbol"]
        aliases = set(_aliases(sym))
        for alias in _listify(item.get("aliases")):
            norm = _normalize(alias)
            if norm:
                aliases.add(norm)
        out.setdefault(sym, set()).update(aliases)
    return out


def _merge_steel_types(column_types: dict[str, ColumnType] | None,
                       steel_census: dict | None) -> tuple[
                           dict[str, ColumnType], dict[str, set[str]], dict[str, dict]]:
    merged = _steel_types(column_types)
    from_census = _steel_types_from_census(steel_census)
    for sym, ct in from_census.items():
        merged.setdefault(sym, ct)
    meta = _steel_meta_from_census(steel_census)
    for sym, ct in merged.items():
        meta.setdefault(sym, {
            "member_type": "COLUMN",
            "section": getattr(ct, "section", "") or sym,
            "source": "document_column_census",
            "evidence": [],
            "expected_pages": [],
        })
    alias_map = {sym: set(_aliases(sym)) for sym in merged}
    for sym, aliases in _alias_map_from_census(steel_census).items():
        alias_map.setdefault(sym, set(_aliases(sym))).update(aliases)
    return merged, alias_map, meta


def _expected_source_pages(steel_census: dict | None) -> set[int]:
    pages: set[int] = set()
    if not steel_census:
        return pages
    for source in _listify(steel_census.get("source_pages")):
        if isinstance(source, dict):
            value = source.get("page")
        else:
            value = source
        try:
            pages.add(int(value))
        except (TypeError, ValueError):
            pass
    for item in _census_items(steel_census):
        for value in _listify(item.get("expected_pages")):
            try:
                pages.add(int(value))
            except (TypeError, ValueError):
                pass
        for value in _listify(item.get("source_pages")):
            try:
                pages.add(int(value))
            except (TypeError, ValueError):
                pass
        for value in _listify(item.get("detail_pages") or item.get("source_page")):
            try:
                pages.add(int(value))
            except (TypeError, ValueError):
                pass
    return pages


def _path_polygon(path, max_closed_segments: int = 12) -> Polygon | None:
    if getattr(path, "fill_polygon", None) is not None:
        poly = path.fill_polygon
        return poly if poly.is_valid and poly.area > 0 else None
    if (not getattr(path, "is_closed", False)
            or not 3 <= len(path.segments) <= max_closed_segments):
        return None
    pts = [seg[0] for seg in path.segments]
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    return poly if poly.is_valid and poly.area > 0 else None


def _rect_like_score(poly: Polygon) -> float:
    mrr = poly.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon" or mrr.area <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, poly.area / mrr.area))
    return ratio


def _collect_symbol_anchors(page: fitz.Page, steel_symbols: dict[str, ColumnType],
                            alias_map: dict[str, set[str]] | None = None) -> list[dict]:
    words = page.get_text("words")
    anchors: list[dict] = []
    wanted = {sym: {_normalize(sym), *_aliases(sym)} for sym in steel_symbols}
    for sym, aliases in (alias_map or {}).items():
        wanted.setdefault(sym, set()).update(_normalize(a) for a in aliases)
    norm_to_symbol = {}
    for sym, aliases in wanted.items():
        for alias in aliases:
            if alias:
                norm_to_symbol[alias] = sym

    for i, word in enumerate(words):
        if len(word) < 5:
            continue
        text = str(word[4])
        norm = _normalize(text)
        candidates = []
        if norm in norm_to_symbol:
            candidates.append((norm_to_symbol[norm], [word]))
        for j in (i + 1, i + 2):
            if j >= len(words):
                continue
            nxt = words[j]
            if abs(float(nxt[1]) - float(word[1])) > 10:
                continue
            combo = _normalize(text + str(nxt[4]))
            if combo in norm_to_symbol:
                candidates.append((norm_to_symbol[combo], [word, nxt]))
        for sym, used_words in candidates:
            x0 = min(float(w[0]) for w in used_words)
            y0 = min(float(w[1]) for w in used_words)
            x1 = max(float(w[2]) for w in used_words)
            y1 = max(float(w[3]) for w in used_words)
            anchors.append({
                "symbol": sym,
                "text": " ".join(str(w[4]) for w in used_words),
                "bbox": [x0, y0, x1, y1],
                "center": [(x0 + x1) / 2, (y0 + y1) / 2],
            })
            break
    return anchors


def _write_audit(result: SteelDetectionResult, audit_out_dir: Path,
                 page: fitz.Page, anchors: list[dict], geoms: list[dict],
                 cfg: SlabV2Config) -> None:
    audit_out_dir.mkdir(parents=True, exist_ok=True)
    page_tag = f"p{page.number + 1:02d}"
    (audit_out_dir / f"steel_candidates_{page_tag}.json").write_text(
        json.dumps(result.candidates, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (audit_out_dir / f"steel_assignment_{page_tag}.json").write_text(
        json.dumps(result.assignment, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (audit_out_dir / "steel_readiness_report.json").write_text(
        json.dumps(result.readiness, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (audit_out_dir / "steel_reconciliation.json").write_text(
        json.dumps({
            "status": "page_local_only",
            "reason": "V1 exports only page-verified steel geometry; cross-floor steel reconciliation is deferred.",
            "page": page.number + 1,
            "readiness": result.readiness,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8")
    if getattr(cfg, "debug_images", False):
        _render_overlay(page, audit_out_dir, anchors, geoms, result.members,
                        f"steel_candidates_{page_tag}.png")
        _render_overlay(page, audit_out_dir, anchors, geoms, result.members,
                        f"steel_assignment_{page_tag}.png")


def _candidate_polygons(paths: list, scale: float) -> list[dict]:
    rows = []
    mm_per_pt = PT_TO_MM * scale
    for p in paths:
        if getattr(p, "outside_content", False):
            continue
        # Steel columns are often drawn as tiny filled/outlined markers rather
        # than RC-style rectangles.  Allow higher-segment closed paths for
        # small circles/markers, but keep large circular text callouts out.
        poly = _path_polygon(p, max_closed_segments=80)
        if poly is None:
            continue
        minx, miny, maxx, maxy = poly.bounds
        w_mm = (maxx - minx) * mm_per_pt
        h_mm = (maxy - miny) * mm_per_pt
        area_m2 = poly.area * mm_per_pt * mm_per_pt / 1_000_000.0
        short_mm = max(1.0, min(w_mm, h_mm))
        long_mm = max(w_mm, h_mm)
        aspect = long_mm / short_mm
        rect_score = _rect_like_score(poly)
        seg_count = len(getattr(p, "segments", []) or [])
        is_roundish_outline = (
            seg_count > 12
            and not bool(getattr(p, "is_filled", False))
            and abs(w_mm - h_mm) <= max(40.0, long_mm * 0.25)
        )
        is_point_marker = (
            0.0005 <= area_m2 <= 0.20
            and 20 <= short_mm <= 500
            and 20 <= long_mm <= 650
        )
        # Round callout bubbles often contain the text anchor itself.  They are
        # useful annotation, not drawable steel.  Keep only truly tiny markers.
        if is_roundish_outline and not is_point_marker:
            continue
        is_small_symbol = area_m2 <= 2.5 and long_mm <= 2500
        is_linear_member = (
            0.02 <= area_m2 <= 8.0
            and 30 <= short_mm <= 900
            and 900 <= long_mm <= 18000
            and aspect >= 3.5
            and rect_score >= 0.55
        )
        # Keep only member-like shapes. Huge floor/slab regions are not steel
        # members even when the page has steel context.
        if not (is_point_marker or is_small_symbol or is_linear_member):
            continue
        if is_linear_member:
            geometry_kind = "linear_member"
        elif is_point_marker:
            geometry_kind = "point_marker"
        else:
            geometry_kind = "small_symbol"
        rows.append({
            "id": f"steel_geom_{len(rows)+1:04d}",
            "polygon": poly,
            "bbox": [minx, miny, maxx, maxy],
            "area_pt2": poly.area,
            "area_m2": area_m2,
            "width_mm": w_mm,
            "height_mm": h_mm,
            "aspect": aspect,
            "geometry_kind": geometry_kind,
            "rect_score": rect_score,
            "segment_count": seg_count,
            "source_path_id": getattr(p, "id", None),
            "style_id": getattr(p, "style_id", None),
            "is_filled": bool(getattr(p, "is_filled", False)),
        })
    return rows


def _candidate_allowed_for(member_type: str, row: dict) -> bool:
    kind = row.get("geometry_kind")
    member_type = str(member_type or "COLUMN").upper()
    if member_type == "COLUMN":
        return kind in {"point_marker", "small_symbol"} or row.get("aspect", 1) <= 4.5
    if member_type in {"BEAM", "BRACING"}:
        return kind == "linear_member"
    if member_type == "FLOOR":
        return row.get("area_m2", 0) >= 0.2
    return True


def _nearest_geometry(anchor: dict, geoms: list[dict], radius_pt: float) -> tuple[dict | None, float]:
    point = Point(anchor["center"])
    best, best_dist = None, float("inf")
    for row in geoms:
        poly = row["polygon"]
        dist = 0.0 if poly.contains(point) else poly.distance(point)
        if dist < best_dist:
            best, best_dist = row, dist
    if best is not None and best_dist <= radius_pt:
        return best, best_dist
    return None, best_dist


def _public_candidate(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k != "polygon"}
    poly = row.get("polygon")
    if poly is not None:
        out["polygon_pdf_pts"] = [list(c) for c in poly.exterior.coords]
    return out


def _render_overlay(page: fitz.Page, out_dir: Path, anchors: list[dict],
                    candidates: list[dict], members: list[SteelMember],
                    name: str) -> str:
    scale = 144 / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    def tx(p):
        return (p[0] * scale, p[1] * scale)

    for row in candidates:
        poly = row.get("polygon")
        if poly is None:
            continue
        pts = [tx(c) for c in poly.exterior.coords]
        dr.line(pts, fill=(255, 120, 0, 180), width=2)
    for anchor in anchors:
        x0, y0, x1, y1 = anchor["bbox"]
        dr.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                     outline=(255, 220, 0, 255), width=2)
        dr.text((x0 * scale, y0 * scale - 20), anchor["symbol"],
                fill=(60, 60, 0, 255), font=font)
    for member in members:
        if member.polygon is None:
            continue
        pts = [tx(c) for c in member.polygon.exterior.coords]
        dr.polygon(pts, fill=(0, 150, 255, 90), outline=(0, 80, 220, 255))
        cx, cy = member.polygon.centroid.coords[0]
        dr.text(tx((cx, cy)), member.symbol, fill=(0, 0, 160, 255), font=font)
    img = Image.alpha_composite(img, ov).convert("RGB")
    path = out_dir / name
    img.save(path)
    return str(path)


def detect_steel(
    page: fitz.Page,
    paths: list,
    classes: list,
    slab_union,
    scale: float,
    column_types: dict[str, ColumnType] | None,
    cfg: SlabV2Config,
    audit_out_dir: Path,
    renderer=None,
    steel_census: dict | None = None,
) -> SteelDetectionResult:
    result = SteelDetectionResult()
    steel_types, alias_map, steel_meta = _merge_steel_types(
        column_types, steel_census)
    source_pages = sorted(_expected_source_pages(steel_census))
    page_number = page.number + 1
    census_status = (steel_census or {}).get("status", "")
    zero_reason = (steel_census or {}).get("zero_steel_reason", "")

    if not steel_types:
        status = "steel_source_missing"
        if source_pages:
            status = "steel_detected_but_unverified"
            zero_reason = zero_reason or (
                "Steel source pages were found, but no drawable steel symbols "
                "were extracted into the steel census.")
        else:
            zero_reason = zero_reason or (
                "No steel symbols, schedules, or steel source pages were found.")
        result.readiness = {
            "status": status,
            "expected_symbols": [],
            "expected_count": 0,
            "verified_count": 0,
            "review_count": 0,
            "rejected_count": 0,
            "source_pages": source_pages,
            "steel_census_status": census_status,
            "zero_steel_reason": zero_reason,
            "export_policy": "verified_only",
            "warnings": [],
        }
        result.assignment = {
            "page_number": page_number,
            "expected_symbols": [],
            "source_pages": source_pages,
            "detected": {},
            "review_count": 0,
            "zero_steel_reason": zero_reason,
        }
        _write_audit(result, audit_out_dir, page, [], [], cfg)
        return result

    if not scale:
        result.warnings.append("steel detection skipped: no verified page scale")
        result.readiness = {
            "status": "steel_detected_but_unverified",
            "expected_symbols": sorted(steel_types),
            "expected_count": len(steel_types),
            "verified_count": 0,
            "review_count": 0,
            "rejected_count": 0,
            "source_pages": source_pages,
            "steel_census_status": census_status,
            "zero_steel_reason": "Steel expected, but page scale is not verified.",
            "export_policy": "verified_only",
            "warnings": result.warnings,
        }
        result.assignment = {
            "page_number": page_number,
            "expected_symbols": sorted(steel_types),
            "source_pages": source_pages,
            "detected": {},
            "review_count": 0,
        }
        _write_audit(result, audit_out_dir, page, [], [], cfg)
        return result

    anchors = _collect_symbol_anchors(page, steel_types, alias_map)
    geoms = _candidate_polygons(paths, scale)

    if source_pages and page_number not in source_pages and not anchors:
        result.readiness = {
            "status": "no_steel_expected",
            "expected_symbols": sorted(steel_types),
            "expected_count": len(steel_types),
            "verified_count": 0,
            "review_count": 0,
            "rejected_count": 0,
            "source_pages": source_pages,
            "steel_census_status": census_status,
            "zero_steel_reason": (
                "Steel census does not list this page as a steel source, "
                "and no steel anchors were found on the page."),
            "export_policy": "verified_only",
            "warnings": [],
        }
        result.assignment = {
            "page_number": page_number,
            "expected_symbols": sorted(steel_types),
            "source_pages": source_pages,
            "anchors": [],
            "assignments": [],
            "detected": {},
            "review_count": 0,
        }
        _write_audit(result, audit_out_dir, page, anchors, geoms, cfg)
        return result

    radius_pt = max(24.0, 650.0 / (PT_TO_MM * scale))
    used_geom_ids: set[str] = set()
    assignments = []

    for anchor in anchors:
        meta = steel_meta.get(anchor["symbol"], {})
        member_type = _canonical_member_type(meta.get("member_type"), meta.get("role"))
        if member_type == "REVIEW_ONLY":
            result.candidates.append({
                "id": f"steel_anchor_{len(result.candidates)+1:04d}",
                "symbol": anchor["symbol"],
                "member_type": member_type,
                "status": "review",
                "source": "text_anchor",
                "anchor": anchor,
                "reject_reason": "steel source is review-only/reference-only",
            })
            continue
        candidate_pool = [
            row for row in geoms if _candidate_allowed_for(member_type, row)
        ]
        geom, distance = _nearest_geometry(anchor, candidate_pool, radius_pt)
        if geom is None or geom["id"] in used_geom_ids:
            result.candidates.append({
                "id": f"steel_anchor_{len(result.candidates)+1:04d}",
                "symbol": anchor["symbol"],
                "member_type": member_type,
                "status": "review",
                "source": "text_anchor",
                "anchor": anchor,
                "nearest_distance_pt": None if math.isinf(distance) else distance,
                "reject_reason": "no unique nearby steel geometry",
            })
            continue
        used_geom_ids.add(geom["id"])
        confidence = 0.92 if distance <= radius_pt * 0.35 else 0.82
        if member_type in {"BEAM", "BRACING"} and geom.get("geometry_kind") == "linear_member":
            confidence = max(confidence, 0.88)
        source_pages_for_symbol = set(meta.get("expected_pages") or []) | set(
            meta.get("source_pages") or []) | set(meta.get("detail_pages") or [])
        source_agreement = (not source_pages_for_symbol
                            or page_number in source_pages_for_symbol
                            or bool(meta.get("detail_pages")))
        if (member_type == "COLUMN"
                and geom.get("geometry_kind") in {"point_marker", "small_symbol"}
                and source_agreement
                and meta.get("export_default", True)):
            confidence = max(confidence, 0.87)
        status = "verified" if confidence >= 0.85 and meta.get("export_default", True) else "review"
        member = SteelMember(
            id=f"steel_{member_type.lower()}_{len(result.members)+1:04d}",
            symbol=anchor["symbol"],
            member_type=member_type,
            polygon=geom["polygon"],
            section=str(meta.get("section") or getattr(
                steel_types[anchor["symbol"]], "section", "") or anchor["symbol"]),
            source="text_anchor_near_vector_geometry",
            confidence=confidence,
            status=status,
            nearby_text=[anchor["text"]],
            evidence=[
                f"steel census symbol {anchor['symbol']}",
                f"steel member type {member_type}",
                "nearby vector geometry",
                f"anchor distance {distance:.2f} pt",
                f"steel source agreement {source_agreement}",
                f"source types {', '.join(meta.get('source_types') or [])}",
                *[str(e) for e in meta.get("evidence", [])[:3]],
            ],
        )
        if status == "verified":
            result.members.append(member)
        assignments.append({
            "member_id": member.id,
            "symbol": member.symbol,
            "status": member.status,
            "geometry_id": geom["id"],
            "distance_pt": distance,
            "confidence": confidence,
            "anchor": anchor,
        })
        result.candidates.append({
            **_public_candidate(geom),
            "symbol": anchor["symbol"],
            "member_type": member_type,
            "status": status,
            "confidence": confidence,
            "anchor": anchor,
            "source": "text_anchor_near_vector_geometry",
        })

    expected_by_symbol = {sym: 1 for sym in steel_types}
    detected_by_symbol: dict[str, int] = {}
    for member in result.members:
        detected_by_symbol[member.symbol] = detected_by_symbol.get(member.symbol, 0) + 1
    review_count = sum(1 for c in result.candidates if c.get("status") == "review")
    result.assignment = {
        "page_number": page_number,
        "expected_symbols": sorted(steel_types),
        "source_pages": source_pages,
        "steel_census_status": census_status,
        "anchors": anchors,
        "assignments": assignments,
        "detected": detected_by_symbol,
        "review_count": review_count,
    }
    if result.members:
        status = "verified_steel"
    elif anchors or (source_pages and page_number in source_pages):
        status = "steel_detected_but_unverified"
        zero_reason = zero_reason or (
            "Steel evidence exists on this page, but no candidate passed "
            "symbol + geometry verification.")
    elif source_pages:
        status = "no_steel_expected"
        zero_reason = zero_reason or (
            "Steel is expected elsewhere in the document, not on this page.")
    else:
        status = "steel_source_missing"
        zero_reason = zero_reason or "No steel source evidence found."
    result.readiness = {
        "status": status,
        "expected_symbols": sorted(steel_types),
        "expected_count": len(steel_types),
        "verified_count": len(result.members),
        "review_count": review_count,
        "rejected_count": 0,
        "source_pages": source_pages,
        "steel_census_status": census_status,
        "zero_steel_reason": zero_reason if not result.members else "",
        "export_policy": "verified_only",
        "warnings": result.warnings,
    }

    _write_audit(result, audit_out_dir, page, anchors, geoms, cfg)
    return result
