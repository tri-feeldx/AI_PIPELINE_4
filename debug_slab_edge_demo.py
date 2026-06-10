"""
Demo: Legend -> Gemini (ranked rows) -> detect slab by fill OR boundary line.

Steps shown as 4-panel image:
  Panel 1: Legend crop with identified rows highlighted
  Panel 2: All LINE-type matched segments highlighted on full drawing
  Panel 3: All FILL-type matched paths highlighted on full drawing
  Panel 4: Final reconstructed polygon

Usage: python debug_slab_edge_demo.py "path.pdf" page_num [output.png]
"""
import sys, os, io, re, math
from collections import defaultdict
import numpy as np
import cv2
import fitz
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

PDF_PATH = sys.argv[1]
PAGE_NUM = int(sys.argv[2]) - 1
OUTPUT   = sys.argv[3] if len(sys.argv) > 3 else "debug_slab_edge_demo.png"

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SCALE    = 1.0   # render scale for full page

print(f"PDF: {PDF_PATH}  page={PAGE_NUM+1}")
print(f"Gemini: {PROJECT} / {MODEL}")

# ── 1. Open PDF ───────────────────────────────────────────────────────────────
doc  = fitz.open(PDF_PATH)
page = doc[PAGE_NUM]
pw, ph = page.rect.width, page.rect.height
drawings = page.get_drawings()
print(f"Page: {pw:.0f}x{ph:.0f}pt  drawings={len(drawings)}")

# ── 2. Find LEGEND ────────────────────────────────────────────────────────────
legend_rect = None
for kw in ["LEGEND", "LEGEND:", "KEY PLAN", "KEYNOTES"]:
    hits = page.search_for(kw)
    if hits:
        r = hits[0]
        legend_rect = fitz.Rect(r.x0 - 200, r.y0 - 20, r.x0 + 400, r.y0 + 400)
        print(f"Legend '{kw}' at ({r.x0:.0f},{r.y0:.0f})")
        break
if legend_rect is None:
    print("ERROR: No LEGEND found"); sys.exit(1)

# ── 3. Extract legend text rows ───────────────────────────────────────────────
blocks = page.get_text("blocks", clip=legend_rect)
legend_lines: list[tuple[float, str]] = []
for blk in sorted(blocks, key=lambda b: b[1]):
    text = blk[4].strip().replace("\n", " ")
    if text:
        legend_lines.append((blk[1], text))

print(f"\nLegend rows ({len(legend_lines)}):")
for i, (y, t) in enumerate(legend_lines):
    print(f"  row{i}: {t[:70]}")

# ── 4. Build fill_by_row (filled paths in legend) ────────────────────────────
fill_by_row: dict[int, tuple] = {}
for d in drawings:
    fill = d.get("fill")
    if fill is None or len(fill) < 3:
        continue
    fill_key = tuple(round(v, 2) for v in fill[:3])
    if fill_key in ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)):
        continue
    rect = d.get("rect")
    if rect is None:
        continue
    cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    if not (legend_rect.x0 <= cx <= legend_rect.x1 and
            legend_rect.y0 <= cy <= legend_rect.y1):
        continue
    nearest = min(range(len(legend_lines)),
                  key=lambda i: abs(legend_lines[i][0] - cy))
    fill_by_row[nearest] = fill_key

# ── 5. Build line_by_row (stroked paths in legend, color != black) ────────────
line_by_row: dict[int, dict] = {}
for d in drawings:
    if d.get("fill") is not None:
        continue
    color = d.get("color")
    if color is None:
        continue
    color_key = tuple(round(v, 2) for v in color[:3])
    width  = round(d.get("width") or 0, 2)
    dashes = (d.get("dashes") or "")[:30]
    pts: list[tuple[float, float]] = []
    for item in d.get("items", []):
        if item[0] == "l":
            pts.append((float(item[1].x if hasattr(item[1], "x") else item[1][0]),
                        float(item[1].y if hasattr(item[1], "y") else item[1][1])))
    if not pts:
        continue
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    if not (legend_rect.x0 <= cx <= legend_rect.x1 and
            legend_rect.y0 <= cy <= legend_rect.y1):
        continue
    nearest = min(range(len(legend_lines)),
                  key=lambda i: abs(legend_lines[i][0] - cy))
    # Keep the entry with the largest width (most prominent line in legend)
    if nearest not in line_by_row or width > line_by_row[nearest]["width"]:
        line_by_row[nearest] = {"color": color_key, "width": width, "dashes": dashes}

print(f"\nFill by row: { {k: v for k,v in fill_by_row.items()} }")
print(f"Line by row: { {k: v for k,v in line_by_row.items()} }")

# ── 6. Build Gemini prompt with symbol context ────────────────────────────────
legend_text = "\n".join(f"  row{i}: {t}" for i, (_, t) in enumerate(legend_lines))

symbol_context_parts = []
for row_idx in sorted(set(list(fill_by_row.keys()) + list(line_by_row.keys()))):
    parts = []
    if row_idx in fill_by_row:
        c = fill_by_row[row_idx]
        parts.append(f"FILL RGB({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")
    if row_idx in line_by_row:
        la = line_by_row[row_idx]
        lc = la["color"]
        parts.append(f"LINE RGB({lc[0]:.2f},{lc[1]:.2f},{lc[2]:.2f}) w={la['width']}pt")
    if parts:
        symbol_context_parts.append(f"  row{row_idx}: {', '.join(parts)}")
symbol_context = "\n".join(symbol_context_parts)

prompt = f"""Structural drawing LEGEND rows:
{legend_text}

Symbols detected (PDF data):
{symbol_context}

Which rows help locate the CONCRETE SLAB? Include: slab fill, slab edge/boundary line, elements embedded in slab (pile caps, etc). Rank by importance (most useful first).

OUTPUT ONLY the numbered list — no intro, no explanation, nothing else:
1. TYPE: fill  ROW: 2
2. TYPE: line  ROW: 0"""

print(f"\nGemini prompt (trimmed):\n{prompt[:400]}...")

# ── 7. Call Gemini ────────────────────────────────────────────────────────────
ranked_rows: list[tuple[str, int]] = []   # [(type, row_idx), ...]
gemini_raw = ""
try:
    import google.genai as genai
    from google.genai import types as gtypes
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    resp = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=500),
    )
    gemini_raw = resp.text.strip() if resp.text else ""
    print(f"\nGemini response:\n{gemini_raw}")

    # Parse: "1. TYPE: fill  ROW: 2" or "TYPE: line ROW: 0"
    matches = re.findall(
        r"TYPE:\s*(fill|line)\s+ROW:\s*(\d+)",
        gemini_raw, re.IGNORECASE
    )
    ranked_rows = [(m[0].lower(), int(m[1])) for m in matches]
    print(f"\nParsed ranked rows: {ranked_rows}")

except Exception as exc:
    print(f"Gemini error: {exc}")
    # Fallback: use first fill row
    if fill_by_row:
        ranked_rows = [("fill", list(fill_by_row.keys())[0])]
    print(f"Fallback rows: {ranked_rows}")

# ── 8. Collect matched elements in drawing body ───────────────────────────────
max_x = pw * 0.78
max_y = ph * 0.88
COLOR_TOL = 0.1
FILL_TOL  = 0.05

line_segments: list[tuple[float,float,float,float]] = []  # (x0,y0,x1,y1)
fill_paths:    list[tuple[float,float,float,float]] = []  # bbox (x0,y0,x1,y1)

for rtype, row_idx in ranked_rows:
    if rtype == "line" and row_idx in line_by_row:
        target_color = line_by_row[row_idx]["color"]
        for d in drawings:
            if d.get("fill") is not None:
                continue
            color = d.get("color")
            if color is None:
                continue
            c = tuple(round(v, 2) for v in color[:3])
            if not all(abs(c[i] - target_color[i]) < COLOR_TOL for i in range(3)):
                continue
            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                p1x = float(item[1].x if hasattr(item[1], "x") else item[1][0])
                p1y = float(item[1].y if hasattr(item[1], "y") else item[1][1])
                p2x = float(item[2].x if hasattr(item[2], "x") else item[2][0])
                p2y = float(item[2].y if hasattr(item[2], "y") else item[2][1])
                if p1x > max_x and p2x > max_x:
                    continue  # skip legend/title area
                if p1y > max_y and p2y > max_y:
                    continue
                line_segments.append((p1x, p1y, p2x, p2y))

    elif rtype == "fill" and row_idx in fill_by_row:
        target_fill = fill_by_row[row_idx]
        for d in drawings:
            fill = d.get("fill")
            if fill is None or len(fill) < 3:
                continue
            if not all(abs(fill[i] - target_fill[i]) < FILL_TOL for i in range(3)):
                continue
            rect = d.get("rect")
            if rect is None:
                continue
            cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
            if cx > max_x or cy > max_y:
                continue
            sym_area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
            if sym_area < 50:
                continue
            fill_paths.append((rect.x0, rect.y0, rect.x1, rect.y1))

print(f"\nMatched: {len(line_segments)} line segments, {len(fill_paths)} fill paths")

# ── 9. Reconstruct polygon from line segments ─────────────────────────────────
from shapely.geometry import box as shapely_box, LineString, MultiLineString
from shapely.ops import unary_union, polygonize

final_poly = None
if line_segments:
    # Try to reconstruct closed polygon from matched line segments
    sys.path.insert(0, "src")
    from slab_extractor import reconstruct_closed_polygons
    page_area = pw * ph
    for tol in [3.0, 10.0, 25.0, 50.0]:
        segs = [((s[0], s[1]), (s[2], s[3])) for s in line_segments]
        polys = reconstruct_closed_polygons([], tol=tol, segments=segs)
        candidates = [p for p in polys if p.area >= page_area * 0.05]
        if candidates:
            final_poly = max(candidates, key=lambda p: p.area)
            print(f"Polygon from line segments (tol={tol}): {final_poly.area/page_area:.1%} of page")
            pts_per_m = 72 / 0.0254
            m2 = final_poly.area / (pts_per_m ** 2) * (100 ** 2)
            print(f"  Area at 1:100 scale: {m2:.1f} m2")
            break

if final_poly is None and fill_paths:
    # Build from fill path centroids
    boxes = [shapely_box(x0, y0, x1, y1) for x0, y0, x1, y1 in fill_paths]
    merged = unary_union(boxes).buffer(100).buffer(-50)
    if not merged.is_empty:
        final_poly = merged if merged.geom_type == "Polygon" else max(merged.geoms, key=lambda g: g.area)
        print(f"Polygon from fill distribution: {final_poly.area/(pw*ph):.1%} of page")

# ── 10. Render full-page image ────────────────────────────────────────────────
mat = fitz.Matrix(SCALE, SCALE)
full_pix = page.get_pixmap(matrix=mat)
full_arr = np.frombuffer(full_pix.samples, dtype=np.uint8).reshape(
    full_pix.height, full_pix.width, full_pix.n)
if full_pix.n == 4:
    full_arr = cv2.cvtColor(full_arr, cv2.COLOR_RGBA2RGB)

# Legend crop
mat_leg = fitz.Matrix(150 / 72, 150 / 72)
leg_pix = page.get_pixmap(matrix=mat_leg, clip=legend_rect)
leg_arr = np.frombuffer(leg_pix.samples, dtype=np.uint8).reshape(
    leg_pix.height, leg_pix.width, leg_pix.n)
if leg_pix.n == 4:
    leg_arr = cv2.cvtColor(leg_arr, cv2.COLOR_RGBA2RGB)
leg_scale = 150 / 72

# ── 11. Build 4-panel image ───────────────────────────────────────────────────
PANEL_H = 600
GAP     = 6

def fit_h(img, h):
    if img.shape[0] == 0:
        return np.zeros((h, 100, 3), np.uint8)
    r = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * r)), h), interpolation=cv2.INTER_AREA)

# Panel 1: Legend with highlighted rows
p1 = leg_arr.copy()
for rank, (rtype, row_idx) in enumerate(ranked_rows[:4]):
    if row_idx >= len(legend_lines):
        continue
    row_y = legend_lines[row_idx][0]
    # find next row y for bottom of box
    next_y = legend_lines[row_idx + 1][0] if row_idx + 1 < len(legend_lines) else row_y + 20
    py0 = int((row_y - legend_rect.y0) * leg_scale) - 2
    py1 = int((next_y - legend_rect.y0) * leg_scale) + 2
    color_box = (0, 0, 200) if rtype == "fill" else (0, 0, 200)  # blue for fill
    if rtype == "line":
        color_box = (200, 0, 0)  # red for line
    cv2.rectangle(p1, (2, max(0, py0)), (p1.shape[1] - 2, min(p1.shape[0] - 1, py1)),
                  color_box, 2)
    label = f"#{rank+1} {rtype}"
    cv2.putText(p1, label, (4, max(12, py0 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_box, 1)
p1 = fit_h(p1, PANEL_H)

# Panel 2: Line segments highlighted
p2 = full_arr.copy()
for x0, y0, x1, y1 in line_segments:
    cv2.line(p2, (int(x0 * SCALE), int(y0 * SCALE)),
             (int(x1 * SCALE), int(y1 * SCALE)), (0, 0, 220), 2)
p2 = fit_h(p2, PANEL_H)

# Panel 3: Fill paths highlighted
p3 = full_arr.copy()
for x0, y0, x1, y1 in fill_paths:
    cv2.rectangle(p3, (int(x0 * SCALE), int(y0 * SCALE)),
                  (int(x1 * SCALE), int(y1 * SCALE)), (0, 180, 0), 2)
p3 = fit_h(p3, PANEL_H)

# Panel 4: Final polygon
p4 = full_arr.copy()
if final_poly is not None:
    from shapely.geometry import Polygon, MultiPolygon
    def draw_poly(img, poly, color, thickness=3):
        if poly is None or poly.is_empty:
            return
        coords = list(poly.exterior.coords)
        pts = np.array([[int(x * SCALE), int(y * SCALE)] for x, y in coords], dtype=np.int32)
        cv2.polylines(img, [pts], True, color, thickness)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (*color[::-1], 100))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    draw_poly(p4, final_poly, (200, 80, 0))
p4 = fit_h(p4, PANEL_H)

# Compose
sp = np.ones((PANEL_H, GAP, 3), np.uint8) * 240
combined = np.concatenate([p1, sp, p2, sp, p3, sp, p4], axis=1)

# Label bar
LABEL_H = 40
lbar = np.ones((LABEL_H, combined.shape[1], 3), np.uint8) * 220
final_img = np.concatenate([lbar, combined], axis=0)

pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil)
labels = [
    f"1. Legend (Gemini: {ranked_rows})",
    "2. LINE matches (red)",
    "3. FILL matches (green)",
    "4. Final polygon",
]
x_cur = 4
for panel, lbl in zip([p1, p2, p3, p4], labels):
    draw.text((x_cur + 4, 8), lbl[:50], fill=(20, 20, 20))
    x_cur += panel.shape[1] + GAP

pil.save(OUTPUT)
print(f"\nSaved -> {OUTPUT}")
