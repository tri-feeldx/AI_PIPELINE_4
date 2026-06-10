"""
Legend side-strip locator for structural PDF review.

This module is intentionally diagnostic-first: it finds likely legend crops,
builds consensus across pages, and leaves slab/wall geometry untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import fitz


LEGEND_KEYWORDS = (
    "LEGEND",
    "PLAN LEGEND",
    "LOAD BEARING",
    "LOAD-BEARING",
    "STEELWORK LEGEND",
)


@dataclass
class LegendCandidate:
    page_index: int
    side: str
    bbox: list[float]
    source: str
    confidence: float
    text_count: int
    vector_count: int
    text_preview: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rect_list(rect: fitz.Rect) -> list[float]:
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


def _clamp_rect(rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(
        max(page_rect.x0, rect.x0),
        max(page_rect.y0, rect.y0),
        min(page_rect.x1, rect.x1),
        min(page_rect.y1, rect.y1),
    )


def _intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    try:
        overlap = fitz.Rect(a)
        overlap.intersect(fitz.Rect(b))
        return not overlap.is_empty
    except Exception:
        return False


def _text_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    blocks = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = block.get("lines", [])
        text_parts = []
        for line in lines:
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    text_parts.append(t)
        text = " ".join(text_parts).strip()
        if text:
            blocks.append({"bbox": fitz.Rect(block["bbox"]), "text": text})
    return blocks


def _drawing_rects(page: fitz.Page) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        r = drawing.get("rect")
        if r:
            rects.append(fitz.Rect(r))
            continue
        points = []
        for item in drawing.get("items", []):
            if item[0] == "l":
                points.extend([item[1], item[2]])
            elif item[0] == "re":
                rects.append(fitz.Rect(item[1]))
            elif item[0] in {"c", "qu"}:
                points.extend(item[1:])
        if points:
            xs = [p.x for p in points if hasattr(p, "x")]
            ys = [p.y for p in points if hasattr(p, "y")]
            if xs and ys:
                rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
    return rects


def _count_evidence(
    rect: fitz.Rect,
    texts: list[dict[str, Any]],
    vectors: list[fitz.Rect],
) -> tuple[int, int, str]:
    matching_text = [t["text"] for t in texts if _intersects(rect, t["bbox"])]
    vector_count = sum(1 for r in vectors if _intersects(rect, r))
    preview = " | ".join(matching_text[:12])
    return len(matching_text), vector_count, preview[:700]


def _has_new_evidence(
    rect: fitz.Rect,
    previous: fitz.Rect,
    texts: list[dict[str, Any]],
    vectors: list[fitz.Rect],
) -> bool:
    # Check the newly exposed bottom band plus newly exposed left/right strips.
    bands = []
    if rect.y1 > previous.y1:
        bands.append(fitz.Rect(rect.x0, previous.y1, rect.x1, rect.y1))
    if rect.x0 < previous.x0:
        bands.append(fitz.Rect(rect.x0, rect.y0, previous.x0, rect.y1))
    if rect.x1 > previous.x1:
        bands.append(fitz.Rect(previous.x1, rect.y0, rect.x1, rect.y1))
    for band in bands:
        text_count, vector_count, _ = _count_evidence(band, texts, vectors)
        if text_count > 0:
            return True
    return False


def _find_anchor(texts: list[dict[str, Any]], side_rect: fitz.Rect) -> dict[str, Any] | None:
    best = None
    best_score = -1
    for item in texts:
        if not _intersects(side_rect, item["bbox"]):
            continue
        upper = item["text"].upper()
        score = 0
        for kw in LEGEND_KEYWORDS:
            if kw in upper:
                score += 3 if "LEGEND" in kw else 1
        if score > best_score:
            best = item
            best_score = score
    return best if best_score > 0 else None


def _fallback_anchor(texts: list[dict[str, Any]], side_rect: fitz.Rect) -> dict[str, Any] | None:
    side_texts = [t for t in texts if _intersects(side_rect, t["bbox"])]
    legend_like = [
        t for t in side_texts
        if any(k in t["text"].upper() for k in ("DENOTES", "WALL", "COLUMN", "SLAB", "CONCRETE", "STEEL"))
    ]
    if len(legend_like) < 3:
        return None
    # Legends are usually dense side stacks. Start at the top-most dense text.
    return sorted(legend_like, key=lambda t: (t["bbox"].y0, t["bbox"].x0))[0]


def _expand_candidate(
    page: fitz.Page,
    side: str,
    anchor: dict[str, Any],
    texts: list[dict[str, Any]],
    vectors: list[fitz.Rect],
    dpi: int,
    source: str,
) -> LegendCandidate:
    page_rect = page.rect
    px = 72.0 / float(dpi)
    down_first = 100 * px
    down_step = 50 * px
    x_step = 10 * px
    backtrack = 250 * px
    anchor_box = fitz.Rect(anchor["bbox"])
    crop_width = max(page_rect.width * 0.16, anchor_box.width + 170 * px)
    if side == "left":
        x0 = max(page_rect.x0, anchor_box.x0 - 80 * px)
        x1 = min(page_rect.x1, x0 + crop_width)
        min_x0 = max(page_rect.x0, anchor_box.x0 - 220 * px)
        max_x1 = min(page_rect.x1, anchor_box.x1 + 420 * px)
    else:
        x1 = min(page_rect.x1, anchor_box.x1 + 180 * px)
        x0 = max(page_rect.x0, min(anchor_box.x0 - 120 * px, x1 - crop_width))
        min_x0 = max(page_rect.x0, anchor_box.x0 - 260 * px)
        max_x1 = min(page_rect.x1, anchor_box.x1 + 520 * px)

    rect = _clamp_rect(fitz.Rect(x0, anchor_box.y0 - 25 * px, x1, anchor_box.y1 + down_first), page_rect)
    accepted = rect
    empty_streak = 0
    iterations = 0
    while empty_streak < 5 and iterations < 80:
        iterations += 1
        next_rect = _clamp_rect(
            fitz.Rect(
                max(min_x0, rect.x0 - x_step),
                rect.y0,
                min(max_x1, rect.x1 + x_step),
                rect.y1 + down_step,
            ),
            page_rect,
        )
        if next_rect == rect:
            break
        if _has_new_evidence(next_rect, rect, texts, vectors):
            accepted = next_rect
            empty_streak = 0
        else:
            empty_streak += 1
        rect = next_rect

    if empty_streak >= 5:
        accepted = _clamp_rect(
            fitz.Rect(accepted.x0, accepted.y0, accepted.x1, max(accepted.y0 + 20 * px, accepted.y1 - backtrack)),
            page_rect,
        )

    text_count, vector_count, preview = _count_evidence(accepted, texts, vectors)
    anchor_bonus = 0.32 if source == "keyword" else 0.12
    density_bonus = min(0.42, text_count * 0.025 + vector_count * 0.004)
    height_bonus = 0.12 if accepted.height > page_rect.height * 0.12 else 0.0
    confidence = min(0.96, anchor_bonus + density_bonus + height_bonus)
    warnings = []
    if source != "keyword":
        warnings.append("no explicit LEGEND keyword; side-density fallback")
    if text_count < 4:
        warnings.append("low text evidence")

    return LegendCandidate(
        page_index=page.number,
        side=side,
        bbox=_rect_list(accepted),
        source=source,
        confidence=round(confidence, 3),
        text_count=text_count,
        vector_count=vector_count,
        text_preview=preview,
        warnings=warnings,
    )


def locate_legends_on_page(page: fitz.Page, dpi: int = 144) -> list[LegendCandidate]:
    texts = _text_blocks(page)
    vectors = _drawing_rects(page)
    page_rect = page.rect
    side_width = page_rect.width * 0.34
    side_rects = {
        "left": fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x0 + side_width, page_rect.y1),
        "right": fitz.Rect(page_rect.x1 - side_width, page_rect.y0, page_rect.x1, page_rect.y1),
    }
    candidates: list[LegendCandidate] = []
    for side, side_rect in side_rects.items():
        anchor = _find_anchor(texts, side_rect)
        if anchor is not None:
            cand = _expand_candidate(page, side, anchor, texts, vectors, dpi, "keyword")
            if cand.text_count >= 2 or cand.vector_count >= 6:
                candidates.append(cand)
    if candidates:
        return candidates

    for side, side_rect in side_rects.items():
        anchor = _fallback_anchor(texts, side_rect)
        if anchor is not None:
            cand = _expand_candidate(page, side, anchor, texts, vectors, dpi, "side_density")
            if cand.text_count >= 3:
                candidates.append(cand)
    return candidates


def _consensus(candidates: list[LegendCandidate], page_count: int) -> dict[str, Any]:
    by_side: dict[str, list[LegendCandidate]] = {"left": [], "right": []}
    for cand in candidates:
        by_side.setdefault(cand.side, []).append(cand)
    best_side = None
    best_items: list[LegendCandidate] = []
    for side, items in by_side.items():
        if len(items) > len(best_items):
            best_side = side
            best_items = items
    if not best_items or not best_side:
        return {"side": None, "coverage": 0.0, "bbox": None, "status": "missing"}
    coverage = len({c.page_index for c in best_items}) / max(1, page_count)
    bboxes = [c.bbox for c in best_items]
    bbox = [float(median([b[i] for b in bboxes])) for i in range(4)]
    return {
        "side": best_side,
        "coverage": round(coverage, 3),
        "bbox": bbox,
        "status": "accepted" if coverage > 0.5 else "weak",
        "page_count": page_count,
        "candidate_page_count": len({c.page_index for c in best_items}),
    }


def locate_legends_for_pages(pdf_path: str, page_indices: list[int], dpi: int = 144) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    candidates: list[LegendCandidate] = []
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= doc.page_count:
                continue
            candidates.extend(locate_legends_on_page(doc[page_index], dpi=dpi))
    finally:
        doc.close()

    rows = []
    for cand in candidates:
        x0, y0, x1, y1 = cand.bbox
        rows.append({
            "Page": cand.page_index + 1,
            "Side": cand.side,
            "Source": cand.source,
            "Confidence": cand.confidence,
            "Text blocks": cand.text_count,
            "Vector items": cand.vector_count,
            "BBox": f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}",
            "Warnings": "; ".join(cand.warnings),
            "Preview": cand.text_preview,
        })
    return {
        "candidates": [c.to_dict() for c in candidates],
        "rows": rows,
        "consensus": _consensus(candidates, len(page_indices)),
    }
