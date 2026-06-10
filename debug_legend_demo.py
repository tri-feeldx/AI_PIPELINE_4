"""
Demo: Legend -> Gemini Vision -> fill-color search -> slab polygon
  Step 1: Find LEGEND text -> crop legend image
  Step 2: Send to Gemini (Vertex AI) -> identify slab fill pattern
  Step 3: Search gray-filled paths in drawing body by that color
  Step 4: Grid occupancy -> slab polygon

Usage: python debug_legend_demo.py "path.pdf" page_num [output.png]
"""
import sys, os, math, json, base64, io
import numpy as np
import cv2
import fitz
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()   # load GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT, etc.

PDF_PATH = sys.argv[1]
PAGE_NUM = int(sys.argv[2]) - 1
OUTPUT   = sys.argv[3] if len(sys.argv) > 3 else "debug_legend_demo.png"

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SCALE    = 1.5

print(f"Project: {PROJECT}  Location: {LOCATION}  Model: {MODEL}")

# ── 1. Open PDF ──────────────────────────────────────────────────────────────
doc  = fitz.open(PDF_PATH)
page = doc[PAGE_NUM]
pw, ph = page.rect.width, page.rect.height
print(f"Page: {pw:.0f}x{ph:.0f} pt")

# ── 2. Find LEGEND and crop ──────────────────────────────────────────────────
legend_rect = None
for kw in ["LEGEND", "LEGEND:", "KEY PLAN", "KEYNOTES"]:
    hits = page.search_for(kw)
    if hits:
        r = hits[0]
        legend_rect   = fitz.Rect(r.x0 - 200, r.y0 - 20, r.x0 + 400, r.y0 + 400)
        legend_text_x = r.x0
        print(f"Found '{kw}' at ({r.x0:.0f},{r.y0:.0f})")
        break

if legend_rect is None:
    print("ERROR: LEGEND not found"); sys.exit(1)

# Render legend at 150 DPI for Gemini
mat_legend = fitz.Matrix(150/72, 150/72)
legend_pix = page.get_pixmap(matrix=mat_legend, clip=legend_rect)
legend_arr = np.frombuffer(legend_pix.samples, dtype=np.uint8).reshape(
                 legend_pix.height, legend_pix.width, legend_pix.n)
if legend_pix.n == 4:
    legend_arr = cv2.cvtColor(legend_arr, cv2.COLOR_RGBA2RGB)

# Save legend PNG for display and for Gemini
legend_pil = Image.fromarray(legend_arr)
legend_png_bytes = io.BytesIO()
legend_pil.save(legend_png_bytes, format="PNG")
legend_png_bytes.seek(0)
print(f"Legend crop: {legend_arr.shape[1]}x{legend_arr.shape[0]}px")

# ── 3. Extract drawings & fill colors in legend (PyMuPDF — exact values) ────
drawings = page.get_drawings()

# ── Ask Gemini to identify slab fill pattern ─────────────────────────────────
print("\n--- Calling Gemini Vision (Vertex AI) ---")
gemini_response = None
identified_color = None

try:
    import google.genai as genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
    )

    from collections import defaultdict

    # Step A: Extract legend TEXT lines (PyMuPDF text, row by row)
    legend_blocks = page.get_text("blocks", clip=legend_rect)
    legend_lines = []
    for blk in sorted(legend_blocks, key=lambda b: b[1]):  # sort by y
        text = blk[4].strip().replace("\n", " ")
        if text:
            legend_lines.append((blk[1], text))  # (y0, text)
    legend_text_str = "\n".join(f"  row{i}: {t}" for i, (_, t) in enumerate(legend_lines))
    print(f"Legend text rows:\n{legend_text_str}")

    # Step B: Build fill-color index by row (match filled paths to nearest text row)
    legend_fill_by_row: dict[int, tuple] = {}   # row_index -> fill_color
    for d in drawings:
        fill = d.get('fill')
        if fill is None or len(fill) < 3: continue
        fill_key = tuple(round(v, 2) for v in fill[:3])
        if fill_key in ((1.0,1.0,1.0), (0.0,0.0,0.0)): continue  # skip white/black
        pts = []
        for item in d.get('items', []):
            if item[0] == 'l':
                for pt in [item[1], item[2]]:
                    pts.append((float(pt.x if hasattr(pt,'x') else pt[0]),
                                 float(pt.y if hasattr(pt,'y') else pt[1])))
            elif item[0] == 're':
                r2 = item[1]; pts += [(r2.x0,r2.y0),(r2.x1,r2.y1)]
        if not pts: continue
        cx_l = sum(p[0] for p in pts)/len(pts)
        cy_l = sum(p[1] for p in pts)/len(pts)
        if not (legend_rect.x0 <= cx_l <= legend_rect.x1 and
                legend_rect.y0 <= cy_l <= legend_rect.y1): continue
        # Find nearest text row
        nearest_row = min(range(len(legend_lines)),
                          key=lambda i: abs(legend_lines[i][0] - cy_l))
        legend_fill_by_row[nearest_row] = fill_key

    row_color_info = "\n".join(
        f"  row{i}: fill_color=RGB({legend_fill_by_row[i][0]:.2f},{legend_fill_by_row[i][1]:.2f},{legend_fill_by_row[i][2]:.2f})"
        for i in sorted(legend_fill_by_row)
    )
    print(f"Fill colors by row:\n{row_color_info or '  (none found)'}")

    # Step C: Send ONLY TEXT to Gemini — ask which row = concrete slab
    prompt = f"""Below are text rows from the LEGEND of a structural engineering drawing:

{legend_text_str}

Task: find the row that describes a CONCRETE SLAB floor area (look for "CONCRETE ELEMENT OVER", "SLAB", "SOG").
Prefer a row that has BOTH load-bearing elements AND concrete over (most structural).

Reply with ONLY a single integer — the row number. Nothing else. Example: 2"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10),
    )

    raw_text = response.text.strip()
    print(f"Gemini raw response: '{raw_text}'")

    # Extract first integer from response
    import re
    nums = re.findall(r'\d+', raw_text)
    row_idx = int(nums[0]) if nums else -1
    identified_color = legend_fill_by_row.get(row_idx, (0.75, 0.75, 0.75))
    print(f"Gemini chose row {row_idx} -> exact PDF fill RGB={identified_color}")
    if row_idx < len(legend_lines):
        print(f"Row text: {legend_lines[row_idx][1]}")

except Exception as e:
    print(f"Gemini call failed: {e}")
    print("Falling back to default: gray (0.75, 0.75, 0.75)")
    identified_color = (0.75, 0.75, 0.75)
    gemini_response = {"item_description": "fallback", "fill_description": "gray", "fill_rgb_estimate": list(identified_color)}

# ── 4. Find all matching-fill paths in drawing body ──────────────────────────
FILL_TOL   = 0.04   # Exact PDF value from PyMuPDF — tight tolerance
LEGEND_X_CUTOFF = pw * 0.78

slab_paths = []
for d in drawings:
    fill = d.get('fill')
    if fill is None: continue
    if len(fill) < 3: continue
    if not all(abs(fill[i] - identified_color[i]) < FILL_TOL for i in range(3)):
        continue
    pts = []
    for item in d.get('items', []):
        if item[0] == 'l':
            for pt in [item[1], item[2]]:
                pts.append((float(pt.x if hasattr(pt,'x') else pt[0]),
                             float(pt.y if hasattr(pt,'y') else pt[1])))
        elif item[0] == 're':
            r2 = item[1]
            pts += [(r2.x0, r2.y0), (r2.x1, r2.y1)]
    if not pts: continue
    x0,y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    x1,y1 = max(p[0] for p in pts), max(p[1] for p in pts)
    cx,cy  = (x0+x1)/2, (y0+y1)/2
    area   = (x1-x0)*(y1-y0)
    if cx > LEGEND_X_CUTOFF: continue   # skip legend area
    if cy > ph * 0.90: continue         # skip title block
    if area < 50: continue              # skip noise
    slab_paths.append((cx, cy, x0, y0, x1, y1, area))

print(f"\nPaths matching slab fill in drawing body: {len(slab_paths)}")

# ── 5. Grid occupancy footprint ──────────────────────────────────────────────
footprint_pts_pt = []
grid = None
gx0 = gy0 = 0

CELL = 50.0  # pt

if slab_paths:
    all_cx = [p[0] for p in slab_paths]
    all_cy = [p[1] for p in slab_paths]
    gx0 = min(all_cx) - CELL
    gy0 = min(all_cy) - CELL
    gx1 = max(all_cx) + CELL
    gy1 = max(all_cy) + CELL

    cols_g  = int(math.ceil((gx1 - gx0) / CELL)) + 1
    rows_g  = int(math.ceil((gy1 - gy0) / CELL)) + 1
    grid    = np.zeros((rows_g, cols_g), dtype=np.uint8)

    for (cx, cy, *_) in slab_paths:
        col = int((cx - gx0) / CELL)
        row = int((cy - gy0) / CELL)
        if 0 <= row < rows_g and 0 <= col < cols_g:
            grid[row, col] = 255

    # Dilate just enough to connect adjacent pile caps (spacing ~400pt, cell=50pt -> ~8 cells)
    # But don't fill large voids — use a cross kernel to avoid diagonal fill
    kernel = np.ones((9, 9), np.uint8)   # ~1 column-spacing of connectivity
    grid_d = cv2.dilate(grid, kernel, iterations=1)

    contours, _ = cv2.findContours(grid_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        for pt in largest.squeeze():
            footprint_pts_pt.append((gx0 + float(pt[0])*CELL, gy0 + float(pt[1])*CELL))

        fp_x0 = min(p[0] for p in footprint_pts_pt)
        fp_y0 = min(p[1] for p in footprint_pts_pt)
        fp_x1 = max(p[0] for p in footprint_pts_pt)
        fp_y1 = max(p[1] for p in footprint_pts_pt)
        print(f"Footprint: ({fp_x0:.0f},{fp_y0:.0f}) to ({fp_x1:.0f},{fp_y1:.0f})")
        print(f"  = {fp_x1-fp_x0:.0f}x{fp_y1-fp_y0:.0f} pt")

# ── 6. Build full-page render ─────────────────────────────────────────────────
mat_full = fitz.Matrix(SCALE, SCALE)
full_pix = page.get_pixmap(matrix=mat_full)
full_arr = np.frombuffer(full_pix.samples, dtype=np.uint8).reshape(
               full_pix.height, full_pix.width, full_pix.n)
if full_pix.n == 4:
    full_arr = cv2.cvtColor(full_arr, cv2.COLOR_RGBA2RGB)

# ── 7. Compose 4-panel image ─────────────────────────────────────────────────
PANEL_H = 700
GAP     = 8
LABEL_H = 50

def fit_h(img, h):
    if img.shape[0] == 0: return np.zeros((h, 100, 3), np.uint8)
    r = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1]*r)), h), interpolation=cv2.INTER_AREA)

# Panel 1: Legend crop + Gemini annotation
p1_base = legend_arr.copy()
lh, lw  = p1_base.shape[:2]
# Green box around first legend items (below header)
tx = int((legend_text_x - legend_rect.x0) * (150/72))
cv2.rectangle(p1_base, (tx-5, int(lh*0.05)), (lw-5, int(lh*0.48)), (0, 200, 0), 3)
p1 = fit_h(p1_base, PANEL_H)

# Panel 2: Full page with slab paths highlighted
p2_base = full_arr.copy()
for (cx, cy, x0, y0, x1, y1, area) in slab_paths:
    cv2.rectangle(p2_base, (int(x0*SCALE), int(y0*SCALE)),
                            (int(x1*SCALE), int(y1*SCALE)), (0, 150, 255), 2)
    cv2.circle(p2_base, (int(cx*SCALE), int(cy*SCALE)), 3, (0, 50, 200), -1)
p2 = fit_h(p2_base, PANEL_H)

# Panel 3: Grid occupancy
p3_base = full_arr.copy()
if grid is not None:
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col]:
                px0 = int((gx0 + col*CELL) * SCALE)
                py0 = int((gy0 + row*CELL) * SCALE)
                px1 = int((gx0 + (col+1)*CELL) * SCALE)
                py1 = int((gy0 + (row+1)*CELL) * SCALE)
                cv2.rectangle(p3_base, (px0,py0), (px1,py1), (0,220,80), -1)
    p3_base = cv2.addWeighted(full_arr.copy(), 0.45, p3_base, 0.55, 0)
p3 = fit_h(p3_base, PANEL_H)

# Panel 4: Final polygon
p4_base = full_arr.copy()
if footprint_pts_pt:
    pts_px = np.array([[int(p[0]*SCALE), int(p[1]*SCALE)]
                        for p in footprint_pts_pt], dtype=np.int32)
    cv2.polylines(p4_base, [pts_px], True, (0, 0, 255), 4)
p4 = fit_h(p4_base, PANEL_H)

# Compose
sp       = np.ones((PANEL_H, GAP, 3), np.uint8) * 255
combined = np.concatenate([p1, sp, p2, sp, p3, sp, p4], axis=1)
lbar     = np.ones((LABEL_H, combined.shape[1], 3), np.uint8) * 220
final    = np.concatenate([lbar, combined], axis=0)

pil  = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil)

gemini_desc = (gemini_response or {}).get("item_description", "")[:50]
labels = [
    f"1. Legend  ->  Gemini: '{gemini_desc}'",
    "2. Matching fill paths (blue)",
    "3. Grid occupancy (green)",
    "4. Slab polygon (blue line)"
]
x_cur = 4
for panel, lbl in zip([p1, p2, p3, p4], labels):
    draw.text((x_cur + 4, 8), lbl, fill=(20, 20, 20))
    x_cur += panel.shape[1] + GAP

pil.save(OUTPUT)
print(f"\nSaved -> {OUTPUT}")
