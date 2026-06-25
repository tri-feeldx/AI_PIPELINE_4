"""
Grid extraction experiment — detect grid bubbles and compute spacing.

Structural drawings have grid lines labeled with:
  - Numbers (1, 2, 3...) along one axis
  - Letters (A, B, C...) along the other axis
Each label sits inside a small circle (grid bubble) at the drawing margins.

This script:
  1. Finds small circles in PDF vector paths
  2. Matches text labels to each circle center
  3. Groups into numbered (horizontal) vs lettered (vertical) grids
  4. Computes real-world spacing between consecutive grid lines
  5. Also extracts dimension text between grids for cross-check

Usage:
    python demo_grid_extract.py "file.pdf" [page_number]
"""

from __future__ import annotations
import sys, re, math
from pathlib import Path
from collections import defaultdict

import fitz


def _find_circles(page: fitz.Page, min_r: float = 5.0, max_r: float = 25.0):
    """Find small circles in PDF drawings (grid bubbles)."""
    drawings = page.get_drawings()
    circles = []
    for d in drawings:
        pts = []
        for item in d["items"]:
            if item[0] == "c":  # cubic bezier — circles are 4 bezier curves
                pts.extend([item[1], item[2], item[3], item[4]])
            elif item[0] == "l":
                pts.extend([item[1], item[2]])
        if len(pts) < 8:
            continue
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        rx = (max(xs) - min(xs)) / 2
        ry = (max(ys) - min(ys)) / 2
        if abs(rx - ry) > 3.0:
            continue
        r = (rx + ry) / 2
        if min_r <= r <= max_r:
            circles.append((cx, cy, r))
    return circles


def _deduplicate_circles(circles, merge_dist: float = 5.0):
    """Merge nearby circles (same bubble drawn multiple times)."""
    if not circles:
        return []
    out = []
    used = set()
    for i, (cx1, cy1, r1) in enumerate(circles):
        if i in used:
            continue
        cluster = [(cx1, cy1, r1)]
        for j, (cx2, cy2, r2) in enumerate(circles):
            if j <= i or j in used:
                continue
            if math.hypot(cx2 - cx1, cy2 - cy1) < merge_dist:
                cluster.append((cx2, cy2, r2))
                used.add(j)
        avg_cx = sum(c[0] for c in cluster) / len(cluster)
        avg_cy = sum(c[1] for c in cluster) / len(cluster)
        avg_r = sum(c[2] for c in cluster) / len(cluster)
        out.append((avg_cx, avg_cy, avg_r))
        used.add(i)
    return out


def _match_text_to_circles(page: fitz.Page, circles, search_r: float = 15.0):
    """Find the text label at the center of each circle."""
    words = page.get_text("words")
    labeled = []
    for cx, cy, r in circles:
        best_text, best_dist = None, search_r
        for w in words:
            wx = (w[0] + w[2]) / 2
            wy = (w[1] + w[3]) / 2
            dist = math.hypot(wx - cx, wy - cy)
            if dist < best_dist:
                best_dist = dist
                best_text = w[4].strip()
        if best_text:
            labeled.append({"label": best_text, "cx": cx, "cy": cy, "r": r})
    return labeled


_GRID_NUM_RE = re.compile(r"^\d{1,2}$")
_GRID_LET_RE = re.compile(r"^[A-Z]$")


def _classify_grids(labeled_circles, page_rect):
    """Split into numbered grids and lettered grids.

    Strategy: grid bubbles form straight lines (same x or same y).
    Find aligned clusters of single-char labels → those are grid lines.
    """
    # only consider potential grid labels (1-char or 2-digit numbers)
    candidates = [c for c in labeled_circles
                  if _GRID_NUM_RE.match(c["label"])
                  or _GRID_LET_RE.match(c["label"])]

    # cluster by Y alignment → horizontal rows of bubbles (numbered grids)
    # cluster by X alignment → vertical columns of bubbles (lettered grids)
    align_tol = 15.0  # pt tolerance for "same line"

    def _cluster_by(circles, key):
        if not circles:
            return []
        sorted_c = sorted(circles, key=lambda c: c[key])
        clusters = [[sorted_c[0]]]
        for c in sorted_c[1:]:
            if abs(c[key] - clusters[-1][-1][key]) < align_tol:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return clusters

    # numbered: cluster by cy (same row), pick rows with ≥2 unique numbers
    num_candidates = [c for c in candidates if _GRID_NUM_RE.match(c["label"])]
    num_rows = _cluster_by(num_candidates, "cy")
    numbered = []
    for row in num_rows:
        unique_labels = set(c["label"] for c in row)
        if len(unique_labels) >= 2:
            # deduplicate: keep one per label (closest to row center)
            by_label = defaultdict(list)
            for c in row:
                by_label[c["label"]].append(c)
            for lbl, group in by_label.items():
                numbered.append(min(group, key=lambda c: c["cy"]))
            break  # take the first (outermost) row

    # lettered: cluster by cx (same column), pick columns with ≥2 unique letters
    let_candidates = [c for c in candidates if _GRID_LET_RE.match(c["label"])]
    let_cols = _cluster_by(let_candidates, "cx")
    lettered = []
    for col in let_cols:
        unique_labels = set(c["label"] for c in col)
        if len(unique_labels) >= 2:
            by_label = defaultdict(list)
            for c in col:
                by_label[c["label"]].append(c)
            for lbl, group in by_label.items():
                lettered.append(min(group, key=lambda c: c["cx"]))
            break

    return numbered, lettered


def _compute_spacing(grids, axis: str, scale: float):
    """Compute real-world spacing between consecutive grid lines.

    axis: 'x' for numbered grids (horizontal spacing),
          'y' for lettered grids (vertical spacing).
    """
    if len(grids) < 2:
        return []
    key = "cx" if axis == "x" else "cy"
    sorted_grids = sorted(grids, key=lambda g: g[key])
    pt_to_mm = (25.4 / 72.0) * scale
    spacings = []
    for i in range(len(sorted_grids) - 1):
        g1 = sorted_grids[i]
        g2 = sorted_grids[i + 1]
        dist_pt = abs(g2[key] - g1[key])
        dist_mm = dist_pt * pt_to_mm
        spacings.append({
            "from": g1["label"],
            "to": g2["label"],
            "dist_pt": round(dist_pt, 1),
            "dist_mm": round(dist_mm, 0),
        })
    return spacings


def _find_dimension_texts(page: fitz.Page):
    """Find dimension values (4-digit+ numbers) that might be grid spacings."""
    words = page.get_text("words")
    dims = []
    for w in words:
        text = w[4].strip()
        if re.match(r"^\d{4,5}$", text):
            cx = (w[0] + w[2]) / 2
            cy = (w[1] + w[3]) / 2
            dims.append({"value": int(text), "cx": cx, "cy": cy})
    return dims


def extract_grids(pdf_path: str, page_number: int = 1, scale: float = 100):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    print(f"Page {page_number}: {page.rect.width:.0f} x {page.rect.height:.0f} pt")
    print(f"Scale: 1:{scale}")
    print()

    # Step 1: find circles
    circles = _find_circles(page)
    circles = _deduplicate_circles(circles)
    print(f"[1] Found {len(circles)} unique circles (grid bubble candidates)")

    # Step 2: match text
    labeled = _match_text_to_circles(page, circles)
    print(f"[2] Matched {len(labeled)} circles to text labels")
    for c in labeled:
        print(f"    {c['label']:>4s}  at ({c['cx']:.0f}, {c['cy']:.0f})")

    # Step 3: classify
    numbered, lettered = _classify_grids(labeled, page.rect)
    print(f"\n[3] Classified (edge-only): {len(numbered)} numbered, {len(lettered)} lettered")
    for c in numbered:
        print(f"    NUM  {c['label']:>4s}  at ({c['cx']:.0f}, {c['cy']:.0f})")
    for c in lettered:
        print(f"    LET  {c['label']:>4s}  at ({c['cx']:.0f}, {c['cy']:.0f})")

    # Step 4: compute spacing
    print("\n[4] Grid spacing (numbered — horizontal):")
    print(f"    {'From':>4s} -> {'To':>4s}  {'pt':>8s}  {'mm':>8s}")
    print("    " + "-" * 32)
    h_spacings = _compute_spacing(numbered, "x", scale)
    for s in h_spacings:
        print(f"    {s['from']:>4s} -> {s['to']:>4s}  {s['dist_pt']:8.1f}  {s['dist_mm']:8.0f}")

    print("\n    Grid spacing (lettered — vertical):")
    print(f"    {'From':>4s} -> {'To':>4s}  {'pt':>8s}  {'mm':>8s}")
    print("    " + "-" * 32)
    v_spacings = _compute_spacing(lettered, "y", scale)
    for s in v_spacings:
        print(f"    {s['from']:>4s} -> {s['to']:>4s}  {s['dist_pt']:8.1f}  {s['dist_mm']:8.0f}")

    # Step 5: dimension text cross-check
    dims = _find_dimension_texts(page)
    if dims:
        print(f"\n[5] Dimension text found on page ({len(dims)} values):")
        for d in dims:
            print(f"    {d['value']}mm at ({d['cx']:.0f}, {d['cy']:.0f})")

    doc.close()

    return {
        "numbered": numbered,
        "lettered": lettered,
        "h_spacings": h_spacings,
        "v_spacings": v_spacings,
        "dimension_texts": dims,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_grid_extract.py <pdf> [page_number] [scale]")
        sys.exit(1)
    pdf = sys.argv[1]
    pg = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    sc = float(sys.argv[3]) if len(sys.argv) > 3 else 100
    extract_grids(pdf, pg, sc)
