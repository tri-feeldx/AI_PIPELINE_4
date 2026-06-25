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
_STEEL_PREFIX_RE = re.compile(r"^(UC|UB|SH|SC|SHS|CHS|RHS|CH)\w*", re.I)


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


def _path_polygon(path) -> Polygon | None:
    if getattr(path, "fill_polygon", None) is not None:
        poly = path.fill_polygon
        return poly if poly.is_valid and poly.area > 0 else None
    if not getattr(path, "is_closed", False) or not 3 <= len(path.segments) <= 12:
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


def _collect_symbol_anchors(page: fitz.Page, steel_symbols: dict[str, ColumnType]) -> list[dict]:
    words = page.get_text("words")
    anchors: list[dict] = []
    wanted = {sym: {_normalize(sym), *_aliases(sym)} for sym in steel_symbols}
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


def _candidate_polygons(paths: list, scale: float) -> list[dict]:
    rows = []
    mm_per_pt = PT_TO_MM * scale
    for p in paths:
        if getattr(p, "outside_content", False):
            continue
        poly = _path_polygon(p)
        if poly is None:
            continue
        minx, miny, maxx, maxy = poly.bounds
        w_mm = (maxx - minx) * mm_per_pt
        h_mm = (maxy - miny) * mm_per_pt
        area_m2 = poly.area * mm_per_pt * mm_per_pt / 1_000_000.0
        # Steel plan symbols are usually small.  Keep this loose because some
        # SH/CH symbols are outlined with detail furniture around them.
        if area_m2 > 2.5 or max(w_mm, h_mm) > 2500:
            continue
        rows.append({
            "id": f"steel_geom_{len(rows)+1:04d}",
            "polygon": poly,
            "bbox": [minx, miny, maxx, maxy],
            "area_pt2": poly.area,
            "area_m2": area_m2,
            "rect_score": _rect_like_score(poly),
            "source_path_id": getattr(p, "id", None),
            "style_id": getattr(p, "style_id", None),
            "is_filled": bool(getattr(p, "is_filled", False)),
        })
    return rows


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
) -> SteelDetectionResult:
    result = SteelDetectionResult()
    steel_types = _steel_types(column_types)
    if not steel_types:
        result.readiness = {
            "status": "not_required",
            "expected_symbols": [],
            "verified_count": 0,
            "review_count": 0,
            "warnings": [],
        }
        return result
    if not scale:
        result.warnings.append("steel detection skipped: no verified page scale")
        result.readiness = {
            "status": "review",
            "expected_symbols": sorted(steel_types),
            "verified_count": 0,
            "review_count": 0,
            "warnings": result.warnings,
        }
        return result

    anchors = _collect_symbol_anchors(page, steel_types)
    geoms = _candidate_polygons(paths, scale)
    radius_pt = max(24.0, 650.0 / (PT_TO_MM * scale))
    used_geom_ids: set[str] = set()
    assignments = []

    for anchor in anchors:
        geom, distance = _nearest_geometry(anchor, geoms, radius_pt)
        if geom is None or geom["id"] in used_geom_ids:
            result.candidates.append({
                "id": f"steel_anchor_{len(result.candidates)+1:04d}",
                "symbol": anchor["symbol"],
                "member_type": "COLUMN",
                "status": "review",
                "source": "text_anchor",
                "anchor": anchor,
                "nearest_distance_pt": None if math.isinf(distance) else distance,
                "reject_reason": "no unique nearby steel geometry",
            })
            continue
        used_geom_ids.add(geom["id"])
        confidence = 0.92 if distance <= radius_pt * 0.35 else 0.82
        status = "verified" if confidence >= 0.85 else "review"
        member = SteelMember(
            id=f"steel_col_{len(result.members)+1:04d}",
            symbol=anchor["symbol"],
            member_type="COLUMN",
            polygon=geom["polygon"],
            section=str(getattr(steel_types[anchor["symbol"]], "section", "") or ""),
            source="text_anchor_near_vector_geometry",
            confidence=confidence,
            status=status,
            nearby_text=[anchor["text"]],
            evidence=[
                f"steel census symbol {anchor['symbol']}",
                "nearby vector geometry",
                f"anchor distance {distance:.2f} pt",
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
            "member_type": "COLUMN",
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
        "page_number": page.number + 1,
        "expected_symbols": sorted(steel_types),
        "anchors": anchors,
        "assignments": assignments,
        "detected": detected_by_symbol,
        "review_count": review_count,
    }
    status = "verified" if result.members else ("review" if anchors else "not_found")
    result.readiness = {
        "status": status,
        "expected_symbols": sorted(steel_types),
        "verified_count": len(result.members),
        "review_count": review_count,
        "export_policy": "verified_only",
        "warnings": result.warnings,
    }

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
        }, indent=2, ensure_ascii=False),
        encoding="utf-8")
    if getattr(cfg, "debug_images", False):
        _render_overlay(page, audit_out_dir, anchors, geoms, result.members,
                        f"steel_candidates_{page_tag}.png")
        _render_overlay(page, audit_out_dir, anchors, geoms, result.members,
                        f"steel_assignment_{page_tag}.png")
    return result
