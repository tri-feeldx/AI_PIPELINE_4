"""
Demo: Column & Foundation Census + Location Detection

Step 1 — Gemini text census: scan all pages → column/foundation types, counts, schedules
Step 2 — Vector detection: find column/foundation positions on each plan page
Step 3 — Visualize: save PNG showing detected elements on each page with hits

Saves:
  columns_<pdf_stem>.json   — full Gemini census
  columns_<pdf_stem>_p<N>.png — visualizations per page

Usage:
  python demo_columns.py
  python demo_columns.py --pdf "path/to/file.pdf"
"""

import argparse
import json
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent

PDFS = [
    r"C:\Users\LENOVO\Downloads\combine strc.pdf",
    r"C:\Users\LENOVO\Downloads\2019-St Carloa Moama-STRUC-Combine1.pdf",
    r"C:\Users\LENOVO\Downloads\Structural.pdf",
]


def draw_elements(page: fitz.Page, columns: list, foundations: list,
                  dpi: int = 120) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    pw, ph = page.rect.width, page.rect.height

    def p2px(x, y):
        return x / pw * iw, y / ph * ih

    for col in columns:
        coords = list(col.polygon.exterior.coords)
        pts = [p2px(x, y) for x, y in coords]
        draw.polygon(pts, outline=(220, 30, 30), width=3)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        draw.text((cx + 3, cy - 8), f"{col.symbol}", fill=(220, 30, 30))

    for fdn in foundations:
        coords = list(fdn.polygon.exterior.coords)
        pts = [p2px(x, y) for x, y in coords]
        draw.polygon(pts, outline=(0, 80, 200), width=3)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        draw.text((cx + 3, cy - 8), f"{fdn.symbol}", fill=(0, 80, 200))

    # Legend
    draw.rectangle([(8, 8), (230, 52)], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.line([(14, 22), (40, 22)], fill=(220, 30, 30), width=3)
    draw.text((46, 15), "Column (vector)", fill=(0, 0, 0))
    draw.line([(14, 42), (40, 42)], fill=(0, 80, 200), width=3)
    draw.text((46, 35), "Foundation (vector)", fill=(0, 0, 0))
    return img


def process_pdf(pdf_path: str):
    stem = Path(pdf_path).stem[:30]
    print(f"\n{'='*70}")
    print(f"PDF: {Path(pdf_path).name}")

    if not Path(pdf_path).exists():
        print(f"  ERROR: not found — skip"); return

    doc = fitz.open(pdf_path)
    n   = doc.page_count
    page_indices = list(range(n))
    print(f"  Pages: {n}")

    # ── Step 1: Gemini Census ────────────────────────────────────────────────
    print(f"\n[Step 1] Gemini Column & Foundation Census...")
    from src.column_analyzer import analyze_columns_and_foundations
    census = analyze_columns_and_foundations(pdf_path, page_indices)

    json_path = BASE / f"columns_{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(census, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved -> {json_path}")
    print(f"  Column types   : {list(census['column_types'].keys())}")
    print(f"  Foundation types: {list(census['foundation_types'].keys())}")
    print(f"  Detail pages   : {census['detail_pages']}")
    print(f"  Footing pages  : {census['footing_plan_pages']}")
    print(f"  Confidence     : {census['detection_confidence']}")
    for bldg in census.get("buildings", []):
        print(f"  Building: {bldg['name']}")
        for fl in bldg.get("floors", []):
            print(f"    {fl['level_name']}: {fl.get('total_columns',0)} cols "
                  f"pages={fl.get('slab_plan_pages',[])} "
                  f"types={fl.get('columns',{})}")
    if census.get("orphan_columns"):
        print(f"  Orphan columns : {census['orphan_columns']}")

    # ── Step 2: Vector Detection ─────────────────────────────────────────────
    print(f"\n[Step 2] Vector path detection on all pages...")
    from src.column_detector import (
        detect_columns_on_page, detect_foundations_on_page,
        assign_columns_to_regions,
    )

    col_types  = census.get("column_types", {})
    fdn_types  = census.get("foundation_types", {})
    footing_1idx = set(census.get("footing_plan_pages", []))

    # Build page → (building, level, relevant col_types) map from census hierarchy.
    # Only scan pages that Gemini identified as slab plan pages for a building/floor.
    page_job_map: dict = {}
    for bldg in census.get("buildings", []):
        for floor in bldg.get("floors", []):
            for pg_1idx in floor.get("slab_plan_pages", []):
                idx = pg_1idx - 1
                types_on_page = floor.get("columns", {})
                relevant = {
                    sym: col_types[sym]
                    for sym in types_on_page
                    if sym in col_types and col_types[sym].get("width_mm") is not None
                }
                if idx not in page_job_map:
                    page_job_map[idx] = {
                        "building": bldg["name"],
                        "level": floor["level_name"],
                        "col_types": {},
                    }
                page_job_map[idx]["col_types"].update(relevant)

    pages_with_hits = []
    all_cols, all_fdns = [], []

    for idx, job in sorted(page_job_map.items()):
        page = doc[idx]
        cols = detect_columns_on_page(
            page, job["col_types"], scale=100, page_index=idx,
            building=job["building"], level=job["level"],
        )
        is_footing = (idx + 1) in footing_1idx
        fdns = detect_foundations_on_page(page, fdn_types, scale=100,
                                          page_index=idx) if is_footing else []
        all_cols.extend(cols)
        all_fdns.extend(fdns)
        if cols or fdns:
            pages_with_hits.append((idx, page, cols, fdns))

    all_cols = assign_columns_to_regions(all_cols, census)
    print(f"  Columns detected   : {len(all_cols)}")
    print(f"  Foundations detected: {len(all_fdns)}")

    # ── Step 3: Visualize ────────────────────────────────────────────────────
    print(f"\n[Step 3] Saving visualization images...")
    saved = []
    for idx, page, cols, fdns in pages_with_hits:
        img = draw_elements(page, cols, fdns)
        out = BASE / f"columns_{stem}_p{idx+1}.png"
        img.save(str(out))
        saved.append(out)
        print(f"  Saved: {out}  ({len(cols)} cols, {len(fdns)} fdns)")

    if not pages_with_hits:
        print(f"  No elements detected via vector paths on any page.")
        print(f"  (Census found types: cols={list(col_types.keys())} "
              f"fdns={list(fdn_types.keys())})")
        print(f"  -> Column symbols may use outline (unfilled) rectangles not matched by vector filter.")

    doc.close()
    return census, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=None)
    args = parser.parse_args()
    pdfs = [args.pdf] if args.pdf else PDFS
    for pdf in pdfs:
        try:
            process_pdf(pdf)
        except Exception as e:
            import traceback
            print(f"\nERROR on {pdf}: {e}")
            traceback.print_exc()
    print(f"\nDone. JSON + PNG files saved in: {BASE}")


if __name__ == "__main__":
    main()
