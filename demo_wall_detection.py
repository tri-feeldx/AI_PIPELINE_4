"""
Standalone wall detection demo — processes a PDF page-by-page, showing
and saving each intermediate step (images + JSON).

Usage:
    python demo_wall_detection.py <pdf_path> [page_numbers...]

    # all pages:
    python demo_wall_detection.py "combine strc.pdf"

    # specific pages (1-indexed):
    python demo_wall_detection.py "combine strc.pdf" 10 11 62

Output:
    demo_wall_output/<pdf_stem>/page_<N>/
        step_A_page_raster.png        — raw page raster
        step_B_wall_classes.png       — paths colored by WALL class
        step_C_wall_faces_raw.png     — all faces from WALL classes
        step_D_wall_faces_filtered.png — after area + aspect ratio filter
        step_E_wall_merged.png        — after merge (core wall clusters)
        step_F_wall_labeled.png       — final walls with text labels
        wall_result.json              — structured output
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point
from shapely.ops import unary_union

from src.slab_v2.config import SlabV2Config
from src.slab_v2 import vector_extract, planarize
from src.slab_v2 import ai_select
from src.slab_v2.wall_extract import (
    _WALL_LABEL_RE, _MIN_AREA_FRAC, _MAX_AREA_FRAC, _MIN_ASPECT_RATIO,
    _MERGE_BUFFER_PT, _MERGE_DEBUFFER_PT, _LABEL_RADIUS_PT,
    _mrr_sides, _classify_wall_type,
)
from src.slab_v2.models import WallFootprint

WALL_COLOR = (142, 36, 170)
DPI = 150


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _content_rect(page):
    from src.vision_refiner import find_legend_rect, find_drawing_content_rect
    legend = find_legend_rect(page)
    return find_drawing_content_rect(page, legend)


def _detect_scale(doc, page_index):
    from src.pdf_processor import extract_text_blocks, detect_scale_from_blocks
    blocks = extract_text_blocks(doc[page_index])
    return detect_scale_from_blocks(blocks)


def _make_base(page, dpi=DPI):
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    raster = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    arr = np.asarray(raster, dtype=np.float32)
    faded = (arr * 0.35 + 255 * 0.65).astype(np.uint8)
    return raster, Image.fromarray(faded), scale


def _tx(p, s):
    return (p[0] * s, p[1] * s)


def _save(img, path):
    img.save(path)
    print(f"  saved: {path}")
    return str(path)


def process_page(pdf_path: str, page_index: int, out_dir: Path):
    """Run wall detection step-by-step on one page, saving everything."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SlabV2Config()
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"PAGE {page_index + 1}")
    print(f"{'='*60}")

    raster, faded, sc = _make_base(page)
    content = _content_rect(page)
    content_area = content.width * content.height

    # ── Step A: page raster ──────────────────────────────────────────────
    print("\n[Step A] Page raster")
    _save(raster, out_dir / "step_A_page_raster.png")

    # ── Vector extraction + planarization ────────────────────────────────
    paths, classes = vector_extract.extract_paths(page, cfg, content)
    all_ids = {c.id for c in classes if c.role != "FRAME"}
    fg_all = planarize.build_face_graph(paths, all_ids, cfg, content_area)

    # ── Gemini Round 1 election ──────────────────────────────────────────
    from src.slab_v2.debug_render import PageRenderer
    rend = PageRenderer(page, cfg, str(out_dir))

    # Gemini needs these prompt images BEFORE election
    rend.step01_paths_by_style(paths, classes)
    rend.step02_style_legend_sheet(classes)

    text_scale = _detect_scale(doc, page_index)
    words = page.get_text("words")
    title_words = [w[4] for w in words
                   if w[0] > page.rect.width * 0.78
                   or w[1] > page.rect.height * 0.88]

    print("\n[Gemini] Running class election...")
    ctx = ai_select.SelectionContext(
        page=page, paths=paths, classes=classes, cfg=cfg,
        content_rect=content, content_area_pt2=content_area,
        renderer=rend, fg_all=fg_all, scale=text_scale,
        title_text=" ".join(title_words))
    election = None
    try:
        election = ai_select.elect_classes(ctx, [])
    except ai_select.AIError as e:
        print(f"  Election raised: {e}")
        print("  (roles were still assigned to classes — checking for WALL)")
    except Exception as e:
        print(f"  Election failed: {e}")
        doc.close()
        return None

    # build roles dict from classes (roles are set on classes even when
    # elect_classes raises AIError for missing slab_edge_classes)
    roles = {}
    if election:
        roles = election.roles
    else:
        for c in classes:
            if c.role and c.role != "UNKNOWN":
                roles[c.id] = c.role

    # ── Step B: identify WALL classes ────────────────────────────────────
    wall_class_ids = {
        cid for cid, role in roles.items() if role == "WALL"
    }
    print(f"\n[Step B] WALL classes identified: {sorted(wall_class_ids)}")

    if not wall_class_ids:
        print("  No WALL classes found — skipping page")
        _save_result(out_dir, page_index, [], wall_class_ids, text_scale)
        doc.close()
        return []

    # draw WALL class paths highlighted
    img = faded.copy()
    dr = ImageDraw.Draw(img)
    for p in paths:
        if p.outside_content:
            continue
        if p.style_id in wall_class_ids:
            color, w = WALL_COLOR, 4
        else:
            color, w = (200, 200, 200), 1
        for (a, b) in p.segments:
            dr.line([_tx(a, sc), _tx(b, sc)], fill=color, width=w)
    font = _font(22)
    dr.text((10, 10), f"WALL classes (purple): {sorted(wall_class_ids)}",
            fill=(0, 0, 0), font=font)
    for cid in wall_class_ids:
        cls = next((c for c in classes if c.id == cid), None)
        if cls:
            print(f"  CLASS {cid}: {cls.key.describe()} | "
                  f"{cls.n_paths} paths, {cls.n_segments} segs")
    _save(img, out_dir / "step_B_wall_classes.png")

    # ── Step C: raw wall faces ───────────────────────────────────────────
    min_area = _MIN_AREA_FRAC * content_area
    max_area = _MAX_AREA_FRAC * content_area
    wall_faces_raw = [
        f for f in fg_all.faces
        if (f.style_ids & wall_class_ids)
        and min_area <= f.area_pt2 <= max_area
    ]
    print(f"\n[Step C] Raw wall faces (area filtered): {len(wall_faces_raw)}")

    img = faded.copy().convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for f in wall_faces_raw:
        ext = [_tx(p, sc) for p in f.polygon.exterior.coords]
        dr.polygon(ext, fill=WALL_COLOR + (100,), outline=WALL_COLOR + (255,))
    img = Image.alpha_composite(img, ov).convert("RGB")
    dr = ImageDraw.Draw(img)
    dr.text((10, 10), f"Raw WALL faces: {len(wall_faces_raw)} "
            f"(area {_MIN_AREA_FRAC*100:.2f}%-{_MAX_AREA_FRAC*100:.0f}% "
            f"of content)", fill=(0, 0, 0), font=_font(22))
    _save(img, out_dir / "step_C_wall_faces_raw.png")

    # ── Step D: aspect ratio filter ──────────────────────────────────────
    filtered = []
    for f in wall_faces_raw:
        short, long = _mrr_sides(f.polygon)
        if short < 0.1:
            continue
        ratio = long / short
        if ratio >= _MIN_ASPECT_RATIO:
            filtered.append((f, short, long, ratio))
    print(f"\n[Step D] After aspect ratio filter (>= {_MIN_ASPECT_RATIO}:1): "
          f"{len(filtered)} faces")
    for f, s, l, r in filtered:
        print(f"  face {f.id}: {s:.1f} x {l:.1f} pt, ratio {r:.1f}:1")

    img = faded.copy().convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for f, s, l, r in filtered:
        ext = [_tx(p, sc) for p in f.polygon.exterior.coords]
        dr.polygon(ext, fill=WALL_COLOR + (130,), outline=WALL_COLOR + (255,))
    img = Image.alpha_composite(img, ov).convert("RGB")
    dr = ImageDraw.Draw(img)
    font14 = _font(14)
    for f, s, l, r in filtered:
        rp = f.polygon.representative_point()
        x, y = _tx((rp.x, rp.y), sc)
        txt = f"{r:.0f}:1"
        bbox = dr.textbbox((x, y), txt, font=font14, anchor="mm")
        dr.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1],
                     fill=(255, 255, 255))
        dr.text((x, y), txt, fill=(100, 0, 120), font=font14, anchor="mm")
    dr.text((10, 10), f"Aspect-filtered: {len(filtered)} faces "
            f"(ratio >= {_MIN_ASPECT_RATIO}:1)",
            fill=(0, 0, 0), font=_font(22))
    _save(img, out_dir / "step_D_wall_faces_filtered.png")

    if not filtered:
        print("  No faces pass aspect ratio filter — skipping")
        _save_result(out_dir, page_index, [], wall_class_ids, text_scale)
        doc.close()
        return []

    # ── Step E: merge adjacent faces ─────────────────────────────────────
    merged = unary_union(
        [f.polygon.buffer(_MERGE_BUFFER_PT) for f, *_ in filtered]
    ).buffer(-_MERGE_DEBUFFER_PT)
    parts = []
    for g in getattr(merged, "geoms", [merged]):
        if g.is_empty or g.area < min_area:
            continue
        short, long = _mrr_sides(g)
        if short < 0.1:
            continue
        ratio = long / short
        if ratio >= _MIN_ASPECT_RATIO:
            parts.append(g)
    print(f"\n[Step E] After merge: {len(parts)} wall polygons")

    img = faded.copy().convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for p in parts:
        ext = [_tx(pt, sc) for pt in p.exterior.coords]
        dr.polygon(ext, fill=WALL_COLOR + (130,), outline=WALL_COLOR + (255,))
    img = Image.alpha_composite(img, ov).convert("RGB")
    dr = ImageDraw.Draw(img)
    dr.text((10, 10), f"Merged walls: {len(parts)} "
            f"(buffer={_MERGE_BUFFER_PT}pt, debuffer={_MERGE_DEBUFFER_PT}pt)",
            fill=(0, 0, 0), font=_font(22))
    _save(img, out_dir / "step_E_wall_merged.png")

    # ── Step F: assign text labels + dimensions ──────────────────────────
    label_candidates = []
    for w in words:
        text = w[4].strip()
        if _WALL_LABEL_RE.match(text):
            cx = (w[0] + w[2]) / 2
            cy = (w[1] + w[3]) / 2
            label_candidates.append((text, Point(cx, cy)))
    print(f"\n[Step F] Wall text labels found: "
          f"{[t for t, _ in label_candidates]}")

    scale_factor = (text_scale or 100) * 25.4 / 72.0
    walls: list[WallFootprint] = []
    unlabeled_idx = 0

    for poly in parts:
        best_label = ""
        best_dist = _LABEL_RADIUS_PT
        for text, pt in label_candidates:
            d = poly.distance(pt)
            if d < best_dist:
                best_dist = d
                best_label = text
        if not best_label:
            unlabeled_idx += 1
            best_label = f"WALL_{unlabeled_idx}"

        short_pt, long_pt = _mrr_sides(poly)
        w_mm = short_pt * scale_factor
        l_mm = long_pt * scale_factor
        wall_type = _classify_wall_type(best_label)

        walls.append(WallFootprint(
            label=best_label, polygon=poly,
            w_mm=round(w_mm, 1), l_mm=round(l_mm, 1),
            wall_type=wall_type,
        ))

    for w in walls:
        print(f"  {w.label}: {w.w_mm:.0f} x {w.l_mm:.0f} mm ({w.wall_type})")

    # final labeled image
    img = faded.copy().convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for w in walls:
        ext = [_tx(p, sc) for p in w.polygon.exterior.coords]
        dr.polygon(ext, fill=WALL_COLOR + (130,), outline=WALL_COLOR + (255,))
    img = Image.alpha_composite(img, ov).convert("RGB")
    dr = ImageDraw.Draw(img)
    font14 = _font(14)
    for w in walls:
        rp = w.polygon.representative_point()
        x, y = _tx((rp.x, rp.y), sc)
        txt = f"{w.label} ({w.w_mm:.0f}x{w.l_mm:.0f})"
        bbox = dr.textbbox((x, y), txt, font=font14, anchor="mm")
        dr.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1],
                     fill=(255, 255, 255))
        dr.text((x, y), txt, fill=(100, 0, 120), font=font14, anchor="mm")
    types_count = {}
    for w in walls:
        types_count[w.wall_type] = types_count.get(w.wall_type, 0) + 1
    summary = "  ".join(f"{t}:{n}" for t, n in sorted(types_count.items()))
    dr.text((10, 10), f"Final: {len(walls)} wall(s)   {summary}",
            fill=(0, 0, 0), font=_font(22))
    _save(img, out_dir / "step_F_wall_labeled.png")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.2f}s")

    _save_result(out_dir, page_index, walls, wall_class_ids, text_scale)
    doc.close()
    return walls


def _save_result(out_dir, page_index, walls, wall_class_ids, scale):
    result = {
        "page_number": page_index + 1,
        "wall_class_ids": sorted(wall_class_ids),
        "scale": scale,
        "walls": [
            {
                "label": w.label,
                "w_mm": w.w_mm,
                "l_mm": w.l_mm,
                "wall_type": w.wall_type,
                "polygon_coords": [list(c) for c in w.polygon.exterior.coords],
            }
            for w in walls
        ],
    }
    path = out_dir / "wall_result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  saved: {path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"ERROR: file not found: {pdf_path}")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()

    if len(sys.argv) > 2:
        pages = [int(p) - 1 for p in sys.argv[2:]]
    else:
        pages = list(range(n_pages))

    stem = Path(pdf_path).stem.replace(" ", "_")
    out_root = Path("demo_wall_output") / stem

    print(f"Wall Detection Demo")
    print(f"PDF: {pdf_path} ({n_pages} pages)")
    print(f"Processing pages: {[p + 1 for p in pages]}")
    print(f"Output: {out_root}")

    total_walls = 0
    for pi in pages:
        if pi < 0 or pi >= n_pages:
            print(f"\nSkipping page {pi + 1} — out of range")
            continue
        page_dir = out_root / f"page_{pi + 1}"
        walls = process_page(pdf_path, pi, page_dir)
        if walls:
            total_walls += len(walls)

    print(f"\n{'='*60}")
    print(f"DONE — {total_walls} wall(s) detected across {len(pages)} page(s)")
    print(f"Output saved to: {out_root}")


if __name__ == "__main__":
    main()
