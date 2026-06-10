"""
Demo Stage 4: Gemini/GPT-4o Vision + Vector Snap + Shapely Subtract

Workflow:
  1. Detect legend rect and drawing content rect from PDF
  2. Render 2 crops: content (200 DPI) + legend (150 DPI)
  3. Send BOTH to Vision API -> no_go_zones + exterior
  4. Snap all vertices to nearest vector endpoint (with Bezier curve sampling)
  5. Subtract no-go zones via Shapely
  6. Draw overlay: orange=no-go, green=snapped exterior, cyan=final polygon

Usage:
  python demo_stage4.py
  python demo_stage4.py --pdf "C:/path/to/file.pdf" --page 61
  python demo_stage4.py --pages 62 63 64 65
  python demo_stage4.py --pdf "path.pdf" --pages 18 --backend openai
"""

import argparse
import math
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
from dotenv import load_dotenv

from src.vision_refiner import (
    DPI_CONTENT, DPI_LEGEND, SNAP_R,
    get_vision_client,
    find_legend_rect, find_drawing_content_rect,
    render_crop, collect_endpoints,
    call_gemini_2images, call_openai_2images,
    snap_all, build_final_polygon,
)

load_dotenv()

PDF_DEFAULT  = r"C:\Users\LENOVO\Downloads\combine strc.pdf"
PAGE_DEFAULT = 61    # 1-indexed


# ── Rendering (full page for overlay background) ───────────────────────────────

def render_full_page(page: fitz.Page, dpi: int) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ── Overlay drawing ────────────────────────────────────────────────────────────

def pdf_to_px(x_pt, y_pt, page, iw, ih):
    return x_pt / page.rect.width * iw, y_pt / page.rect.height * ih


def _draw_dashed(draw, pts_closed, color, width=3, dash=14):
    for i in range(len(pts_closed) - 1):
        x0, y0 = pts_closed[i]; x1, y1 = pts_closed[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1:
            continue
        steps = max(1, int(seg / dash))
        for s in range(steps):
            t0 = s / steps; t1 = min((s + 0.5) / steps, 1.0)
            draw.line([(x0 + t0*(x1-x0), y0 + t0*(y1-y0)),
                       (x0 + t1*(x1-x0), y0 + t1*(y1-y0))],
                      fill=color, width=width)


def draw_overlay(full_img, page, exterior_gem, content_rect,
                 snapped_ext, snap_info, no_go_zones, final_poly) -> Image.Image:
    out  = full_img.copy()
    draw = ImageDraw.Draw(out)
    iw, ih = out.size

    def g2px(xn, yn):
        xp = content_rect.x0 + xn * content_rect.width
        yp = content_rect.y0 + yn * content_rect.height
        return pdf_to_px(xp, yp, page, iw, ih)

    def p2px(x, y):
        return pdf_to_px(x, y, page, iw, ih)

    # Red dashed — Gemini raw exterior
    if exterior_gem:
        gem_px = [g2px(x, y) for x, y in exterior_gem]
        _draw_dashed(draw, gem_px + [gem_px[0]], (220, 40, 40), width=3)
        for xn, yn in exterior_gem:
            px, py = g2px(xn, yn)
            draw.ellipse([(px-5, py-5), (px+5, py+5)], fill=(220, 40, 40))

    # Green solid — snapped exterior (orange dot = unsnapped)
    if snapped_ext:
        sp = [p2px(x, y) for x, y in snapped_ext]
        for i in range(len(sp)):
            draw.line([sp[i], sp[(i+1) % len(sp)]], fill=(0, 200, 80), width=3)
        for i, (x, y) in enumerate(snapped_ext):
            px, py = p2px(x, y)
            c = (0, 200, 80) if snap_info[i]["snapped"] else (255, 180, 0)
            draw.ellipse([(px-6, py-6), (px+6, py+6)], fill=c, outline=(0, 0, 0))
            draw.text((px+8, py-6), str(i), fill=(0, 80, 30))

    # Orange dashed — no-go zones
    for zi, (nx0, ny0, nx1, ny1) in enumerate(no_go_zones or []):
        corners = [
            p2px(content_rect.x0 + nx0*content_rect.width,
                 content_rect.y0 + ny0*content_rect.height),
            p2px(content_rect.x0 + nx1*content_rect.width,
                 content_rect.y0 + ny0*content_rect.height),
            p2px(content_rect.x0 + nx1*content_rect.width,
                 content_rect.y0 + ny1*content_rect.height),
            p2px(content_rect.x0 + nx0*content_rect.width,
                 content_rect.y0 + ny1*content_rect.height),
        ]
        _draw_dashed(draw, corners + [corners[0]], (255, 140, 0), width=4)
        draw.text((corners[0][0]+4, corners[0][1]+4), f"NO-GO {zi}", fill=(255, 140, 0))

    # Cyan thick — final polygon after Shapely subtract
    if final_poly and not final_poly.is_empty:
        fp = [p2px(x, y) for x, y in final_poly.exterior.coords]
        for i in range(len(fp) - 1):
            draw.line([fp[i], fp[i+1]], fill=(0, 220, 220), width=5)

    # Blue — content crop boundary
    cr = [p2px(content_rect.x0, content_rect.y0),
          p2px(content_rect.x1, content_rect.y0),
          p2px(content_rect.x1, content_rect.y1),
          p2px(content_rect.x0, content_rect.y1)]
    for i in range(4):
        draw.line([cr[i], cr[(i+1) % 4]], fill=(0, 120, 255), width=2)

    # Legend box
    lx, ly = 20, 20
    draw.rectangle([(lx, ly), (lx+310, ly+108)], fill=(255, 255, 255), outline=(0, 0, 0))
    entries = [
        ((220, 40, 40),  "Gemini raw exterior"),
        ((0, 200, 80),   "Snapped exterior (orange = unsnapped)"),
        ((255, 140, 0),  "No-go zones"),
        ((0, 220, 220),  "Final polygon (after subtract)"),
        ((0, 120, 255),  "Content crop boundary"),
    ]
    for ei, (col, lbl) in enumerate(entries):
        draw.line([(lx+10, ly+18+ei*18), (lx+40, ly+18+ei*18)], fill=col, width=3)
        draw.text((lx+48, ly+11+ei*18), lbl, fill=(0, 0, 0))

    return out


# ── Per-page pipeline ──────────────────────────────────────────────────────────

def process_page(pdf_path: str, page_number: int, snap_r: float,
                 client, model: str, backend: str, base: Path,
                 save_crops: bool = False, no_overlay: bool = False):
    page_idx = page_number - 1
    doc  = fitz.open(pdf_path)
    page = doc[page_idx]
    print(f"\n{'='*60}")
    print(f"PDF: {Path(pdf_path).name}  Page: {page_number}  "
          f"({page.rect.width:.0f}x{page.rect.height:.0f}pt)  backend={backend}")

    legend_rect  = find_legend_rect(page)
    content_rect = find_drawing_content_rect(page, legend_rect)
    print(f"  Content: ({content_rect.x0:.0f},{content_rect.y0:.0f})-"
          f"({content_rect.x1:.0f},{content_rect.y1:.0f})")

    img_content, bytes_content = render_crop(page, content_rect, DPI_CONTENT)
    img_legend,  bytes_legend  = render_crop(page, legend_rect,  DPI_LEGEND)
    print(f"  Content img: {img_content.size[0]}x{img_content.size[1]}px  "
          f"Legend img: {img_legend.size[0]}x{img_legend.size[1]}px")

    if save_crops:
        img_content.save(str(base / f"crop_content_p{page_number}.png"))
        img_legend.save( str(base / f"crop_legend_p{page_number}.png"))

    if backend == "openai":
        exterior, no_go_zones = call_openai_2images(
            client, model, bytes_content, bytes_legend)
    else:
        exterior, no_go_zones = call_gemini_2images(
            client, model, bytes_content, bytes_legend)

    snapped_ext, snap_info = snap_all(exterior, content_rect, page, snap_r)
    final_poly = build_final_polygon(snapped_ext, no_go_zones, content_rect)

    full_img = render_full_page(page, DPI_CONTENT)
    if no_overlay:
        result_img = full_img
    else:
        result_img = draw_overlay(
            full_img, page,
            exterior, content_rect,
            snapped_ext, snap_info,
            no_go_zones, final_poly,
        )

    pdf_stem = Path(pdf_path).stem[:20]
    out_name = f"demo_stage4_{pdf_stem}_p{page_number}.png"
    out_path = base / out_name
    result_img.save(str(out_path))
    print(f"Saved: {out_path}")

    doc.close()
    return out_path


# ── CLI entry ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",     default=PDF_DEFAULT)
    parser.add_argument("--page",    type=int, default=PAGE_DEFAULT)
    parser.add_argument("--pages",   type=int, nargs="+")
    parser.add_argument("--snap-r",  type=float, default=SNAP_R)
    parser.add_argument("--backend", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--crops",      action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"ERROR: PDF not found: {args.pdf}"); sys.exit(1)

    pages   = args.pages if args.pages else [args.page]
    base    = Path(__file__).parent
    backend = args.backend

    print(f"Backend: {backend}  Pages: {pages}")
    client, model = get_vision_client(backend)
    print(f"Model: {model}")

    for pn in pages:
        try:
            process_page(args.pdf, pn, args.snap_r, client, model, backend, base,
                         args.crops, getattr(args, "no_overlay", False))
        except Exception as e:
            print(f"ERROR on page {pn}: {e}")


if __name__ == "__main__":
    main()
