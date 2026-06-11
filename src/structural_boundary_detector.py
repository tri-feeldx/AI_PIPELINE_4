"""
Wall/core/opening evidence for wall-guided slab extraction.

This module does not create final wall geometry. It classifies vector/text
evidence so slab extraction can pick better boundaries on no-fill drawings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import fitz
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


@dataclass
class BoundaryObject:
    kind: str
    polygon: Polygon
    label: str = ""
    confidence: float = 0.0
    source: str = "geometry"
    auto_cut: bool = False


@dataclass
class StructuralBoundaryResult:
    walls: list[BoundaryObject] = field(default_factory=list)
    cores: list[BoundaryObject] = field(default_factory=list)
    load_bearing_elements: list[BoundaryObject] = field(default_factory=list)
    columns_or_piles: list[BoundaryObject] = field(default_factory=list)
    footings: list[BoundaryObject] = field(default_factory=list)
    stairs: list[BoundaryObject] = field(default_factory=list)
    openings: list[BoundaryObject] = field(default_factory=list)
    penetrations: list[BoundaryObject] = field(default_factory=list)
    uncertain_regions: list[BoundaryObject] = field(default_factory=list)
    ignored_regions: list[BoundaryObject] = field(default_factory=list)
    debug: dict = field(default_factory=dict)

    @property
    def cut_candidates(self) -> list[BoundaryObject]:
        return [
            *[o for o in self.cores if o.auto_cut],
            *[o for o in self.stairs if o.auto_cut],
            *[o for o in self.openings if o.auto_cut],
            *[o for o in self.penetrations if o.auto_cut],
        ]


_STAIR_RE = re.compile(r"\b(STAIR|STAIRS|ST)\s*\d*\b", re.I)
_CORE_RE = re.compile(r"\b(LIFT|CORE|SHAFT|RISER|ELEVATOR)\b", re.I)
_OPENING_RE = re.compile(r"\b(OPENING|VOID|PENETRATION|PEN|HATCH|HOLE)\b", re.I)
_IGNORE_RE = re.compile(r"\b(LEGEND|NOTES?|SCHEDULE|TITLE|REVISION|KEYPLAN|SCALE)\b", re.I)
_COLUMN_PILE_RE = re.compile(
    r"\b([A-Z]-)?C{1,2}\d{1,3}\b|\bPC\d+\b|\bPILE\s*CAP\b|\bCONCRETE\s+COLUMN\b",
    re.I,
)
_FOOTING_RE = re.compile(r"\bPF\d+\b|\bPAD\s+FOOTING\b|\bSTRIP\s+FOOTING\b|\bFOOTING\b", re.I)
_LOAD_BEARING_RE = re.compile(r"\bLOAD[-\s]*BEARING\b|\bCONCRETE\s+ELEMENT\b", re.I)
_WALL_LABEL_RE = re.compile(
    r"\b((R\.?\s*C\.?|RC|CONCRETE|MASONRY|PRECAST|BLOCKWORK|LOAD\s*BEARING|LOAD-BEARING|RETAINING)\s+WALL|WALL\s+(W\d+|BW\d+))\b",
    re.I,
)


def _rect_from_item(item):
    kind = item[0]
    if kind == "re":
        return item[1]
    if kind == "qu":
        q = item[1]
        return fitz.Rect(
            min(q.ul.x, q.ur.x, q.ll.x, q.lr.x),
            min(q.ul.y, q.ur.y, q.ll.y, q.lr.y),
            max(q.ul.x, q.ur.x, q.ll.x, q.lr.x),
            max(q.ul.y, q.ur.y, q.ll.y, q.lr.y),
        )
    return None


def _page_drawing_area(page: fitz.Page) -> Polygon:
    w, h = page.rect.width, page.rect.height
    return box(0, 0, w * 0.84, h * 0.90)


def _text_blocks(page: fitz.Page) -> list[dict]:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    out = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    out.append({"text": text, "bbox": span["bbox"]})
    return out


def _bbox_poly(bbox) -> Polygon:
    return box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _nearest_rect_poly(cx: float, cy: float, rects: list[Polygon], radius: float = 90.0) -> Polygon | None:
    best = None
    best_dist = float("inf")
    for poly in rects:
        c = poly.centroid
        dist = math.hypot(c.x - cx, c.y - cy)
        if dist <= radius and dist < best_dist:
            best = poly
            best_dist = dist
    return best


def _semantic_keywords(legend_semantics: dict | None, key: str) -> list[str]:
    if not legend_semantics:
        return []
    rules = legend_semantics.get("rules_for_code", {}) or {}
    values = rules.get(key, []) or []
    out = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned.upper() not in {v.upper() for v in out}:
            out.append(cleaned)
    return out


def _keyword_hit(text: str, keywords: list[str]) -> str | None:
    upper = text.upper()
    for keyword in keywords:
        if keyword.upper() in upper:
            return keyword
    return None


def _looks_like_note_text(text: str) -> bool:
    upper = text.upper()
    if len(text) > 90:
        return True
    return any(token in upper for token in ("REFER TO", "DRAWING", "ARCHITECT", "NOTES", "SCHEDULE"))


def _looks_like_round_or_square_symbol(poly: Polygon) -> bool:
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return False
    ratio = max(w, h) / max(min(w, h), 1)
    return 0.55 <= min(w, h) <= 45 and ratio <= 1.45


def _extract_vector_evidence(drawings: list[dict], drawing_area: Polygon) -> tuple[list[Polygon], list[Polygon]]:
    walls: list[Polygon] = []
    rects: list[Polygon] = []
    for d in drawings:
        width = float(d.get("width") or 0)
        rect = d.get("rect")
        if rect:
            try:
                poly = box(rect.x0, rect.y0, rect.x1, rect.y1)
            except Exception:
                poly = None
            if poly is not None and not poly.is_empty and drawing_area.intersects(poly.centroid):
                rw = abs(rect.x1 - rect.x0)
                rh = abs(rect.y1 - rect.y0)
                if min(rw, rh) >= 2:
                    rects.append(poly)
                aspect = max(rw, rh) / max(min(rw, rh), 1)
                if min(rw, rh) >= 2 and aspect >= 3:
                    walls.append(poly)

        if width >= 1.2 and d.get("items"):
            pts = []
            for item in d.get("items", []):
                if item[0] == "l":
                    pts.extend([(float(item[1].x), float(item[1].y)), (float(item[2].x), float(item[2].y))])
                else:
                    r = _rect_from_item(item)
                    if r:
                        pts.extend([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
            if len(pts) >= 2:
                try:
                    poly = LineString(pts).buffer(max(width, 1.2), cap_style=2, join_style=2)
                    if not poly.is_empty and drawing_area.intersects(poly.centroid):
                        walls.append(poly)
                except Exception:
                    pass
    return walls, rects


def detect_structural_boundary_objects(
    page: fitz.Page,
    drawings: list[dict],
    text_blocks: list[dict] | None = None,
    auto_cut_voids: bool = True,
    legend_semantics: dict | None = None,
) -> StructuralBoundaryResult:
    """Classify wall/core/opening evidence on one page."""
    drawing_area = _page_drawing_area(page)
    blocks = text_blocks if text_blocks is not None else _text_blocks(page)
    wall_polys, rects = _extract_vector_evidence(drawings, drawing_area)
    wall_keywords = _semantic_keywords(legend_semantics, "wall_keywords")
    slab_boundary_keywords = _semantic_keywords(legend_semantics, "slab_boundary_keywords")
    cut_keywords = _semantic_keywords(legend_semantics, "net_slab_cut_keywords")
    never_cut_keywords = _semantic_keywords(legend_semantics, "never_cut_keywords")

    result = StructuralBoundaryResult()
    for poly in wall_polys:
        if _looks_like_round_or_square_symbol(poly):
            result.columns_or_piles.append(BoundaryObject("column_or_pile", poly, confidence=0.58, source="compact_symbol"))
        else:
            result.walls.append(BoundaryObject("wall", poly, confidence=0.65, source="thick_line_or_rect"))

    for block in blocks:
        text = block.get("text", "")
        bbox = block.get("bbox")
        if not bbox:
            continue
        label_poly = _bbox_poly(bbox).buffer(12)
        center = label_poly.centroid
        target = _nearest_rect_poly(center.x, center.y, rects) or label_poly
        wall_hit = _keyword_hit(text, wall_keywords)
        cut_hit = _keyword_hit(text, cut_keywords)
        never_cut_hit = _keyword_hit(text, never_cut_keywords)
        slab_hit = _keyword_hit(text, slab_boundary_keywords)
        generic_wall_hit = _WALL_LABEL_RE.search(text)
        column_hit = _COLUMN_PILE_RE.search(text)
        footing_hit = _FOOTING_RE.search(text)
        load_bearing_hit = _LOAD_BEARING_RE.search(text)
        in_drawing_area = drawing_area.buffer(5).contains(center)
        if footing_hit and in_drawing_area:
            result.footings.append(BoundaryObject(
                "footing",
                target,
                text,
                0.82,
                "footing_label",
            ))
        elif column_hit and in_drawing_area:
            result.columns_or_piles.append(BoundaryObject(
                "column_or_pile",
                target,
                text,
                0.84,
                "column_or_pile_label",
            ))
        elif (wall_hit or generic_wall_hit) and in_drawing_area and not _looks_like_note_text(text):
            result.walls.append(BoundaryObject(
                "wall",
                target,
                text,
                0.88 if wall_hit else 0.82,
                f"legend_semantic:{wall_hit}" if wall_hit else "wall_label",
            ))
        elif load_bearing_hit and in_drawing_area and not _looks_like_note_text(text):
            result.load_bearing_elements.append(BoundaryObject(
                "load_bearing_element",
                target,
                text,
                0.78,
                "load_bearing_label",
            ))
        elif cut_hit and not never_cut_hit:
            result.penetrations.append(BoundaryObject(
                "penetration", target, text, 0.86, f"legend_semantic:{cut_hit}",
                auto_cut=bool(auto_cut_voids),
            ))
        elif slab_hit:
            result.uncertain_regions.append(BoundaryObject(
                "slab_boundary_evidence", label_poly, text, 0.72, f"legend_semantic:{slab_hit}",
                auto_cut=False,
            ))
        elif _IGNORE_RE.search(text):
            result.ignored_regions.append(BoundaryObject("ignored", label_poly, text, 0.60, "text"))
        elif _STAIR_RE.search(text):
            result.stairs.append(BoundaryObject(
                "stair", target, text, 0.82, "text+nearby_rect",
                auto_cut=bool(auto_cut_voids),
            ))
        elif _CORE_RE.search(text):
            result.cores.append(BoundaryObject(
                "core", target, text, 0.78, "text+nearby_rect",
                auto_cut=bool(auto_cut_voids),
            ))
        elif _OPENING_RE.search(text):
            result.openings.append(BoundaryObject(
                "opening", target, text, 0.76, "text+nearby_rect",
                auto_cut=bool(auto_cut_voids),
            ))

    wall_union = unary_union([w.polygon for w in result.walls]) if result.walls else None
    result.debug = {
        "walls": len(result.walls),
        "load_bearing_elements": len(result.load_bearing_elements),
        "columns_or_piles": len(result.columns_or_piles),
        "footings": len(result.footings),
        "cores": len(result.cores),
        "stairs": len(result.stairs),
        "openings": len(result.openings),
        "penetrations": len(result.penetrations),
        "cut_candidates": len(result.cut_candidates),
        "semantic_wall_keywords": wall_keywords,
        "semantic_slab_boundary_keywords": slab_boundary_keywords,
        "semantic_cut_keywords": cut_keywords,
        "semantic_walls": sum(
            1 for w in result.walls
            if str(w.source).startswith("legend_semantic") or str(w.source) == "wall_label"
        ),
        "semantic_slab_boundary_evidence": sum(
            1 for o in result.uncertain_regions if o.kind == "slab_boundary_evidence"
        ),
        "semantic_cut_candidates": sum(
            1 for o in result.cut_candidates if str(o.source).startswith("legend_semantic")
        ),
        "wall_area_pdf": float(wall_union.area) if wall_union and not wall_union.is_empty else 0.0,
    }
    return result
