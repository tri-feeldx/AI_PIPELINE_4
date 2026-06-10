"""
Final diagnostic: understand 3.96pt path and find best slab boundary.
Usage: python debug_thick.py "path/to/pdf" page_num
"""
import sys, math
import fitz
from collections import defaultdict

sys.path.insert(0, "src")
from slab_extractor import _is_sheet_border

def path_segs(d):
    segs = []
    for item in d.get("items", []):
        if item[0] == "l":
            p1 = (float(item[1].x if hasattr(item[1], "x") else item[1][0]),
                  float(item[1].y if hasattr(item[1], "y") else item[1][1]))
            p2 = (float(item[2].x if hasattr(item[2], "x") else item[2][0]),
                  float(item[2].y if hasattr(item[2], "y") else item[2][1]))
            if math.dist(p1, p2) > 0.5:
                segs.append((p1, p2))
    return segs

def main():
    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) - 1
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    drawings = page.get_drawings()
    page_area = page.rect.width * page.rect.height

    # --- 3.96pt path analysis ---
    print("=== 3.96pt path analysis ===")
    for d in drawings:
        w = d.get("width") or 0
        if round(w, 2) != 3.96:
            continue
        segs = path_segs(d)
        lengths = [math.dist(s[0], s[1]) for s in segs]
        lengths.sort()
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        print(f"  {len(segs)} segs, lengths: min={lengths[0]:.1f} median={lengths[len(lengths)//2]:.1f} max={lengths[-1]:.1f}")
        print(f"  x range: {min(xs):.0f} to {max(xs):.0f}")
        print(f"  y range: {min(ys):.0f} to {max(ys):.0f}")
        # Sample first 20 segments
        print("  First 10 segs:")
        for s in segs[:10]:
            print(f"    ({s[0][0]:.1f},{s[0][1]:.1f}) -> ({s[1][0]:.1f},{s[1][1]:.1f})  len={math.dist(s[0],s[1]):.1f}")
        # Check if horizontal/vertical
        h_count = sum(1 for s in segs if abs(s[0][1]-s[1][1]) < 2)
        v_count = sum(1 for s in segs if abs(s[0][0]-s[1][0]) < 2)
        print(f"  Horizontal segs: {h_count}, Vertical segs: {v_count}")

    # --- Strategy: bbox of heaviest path with most segments ---
    print("\n=== Best bbox from most-segment heavy path ===")
    best_seg_count = 0
    best_d = None
    best_w = 0
    for d in drawings:
        w = d.get("width") or 0
        if w < 0.96 or d.get("color") is None:
            continue
        segs = path_segs(d)
        if len(segs) > best_seg_count:
            xs = [p[0] for s in segs for p in s]
            ys = [p[1] for s in segs for p in s]
            bb_area = (max(xs)-min(xs)) * (max(ys)-min(ys))
            if bb_area >= page_area * 0.10:
                best_seg_count = len(segs)
                best_d = d
                best_w = w

    if best_d:
        segs = path_segs(best_d)
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        bb = (min(xs), min(ys), max(xs), max(ys))
        bb_area = (bb[2]-bb[0]) * (bb[3]-bb[1])
        print(f"Best: w={best_w}pt, {len(segs)} segs, bbox={tuple(round(b,0) for b in bb)}, area={bb_area/page_area:.1%}")
        from shapely.geometry import box as shapely_box
        poly = shapely_box(*bb)
        print(f"Sheet border? {_is_sheet_border(poly, page)}")

    # --- Strategy: union of all structural element bboxes > 1% ---
    print("\n=== Top 5 paths by segment count (>=0.96pt) ===")
    path_data = []
    for d in drawings:
        w = d.get("width") or 0
        if w < 0.96 or d.get("color") is None:
            continue
        segs = path_segs(d)
        if not segs:
            continue
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        bb = (min(xs), min(ys), max(xs), max(ys))
        bb_area = (bb[2]-bb[0]) * (bb[3]-bb[1])
        path_data.append((w, len(segs), bb, bb_area))
    path_data.sort(key=lambda x: x[1], reverse=True)
    for w, sc, bb, ba in path_data[:5]:
        from shapely.geometry import box as shapely_box
        poly = shapely_box(*bb)
        print(f"  w={w}pt segs={sc} bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) "
              f"area={ba/page_area:.1%} sheet_border={_is_sheet_border(poly, page)}")

    # --- Convex hull of all segment endpoints from >=0.96pt paths within 78% width ---
    print("\n=== Convex hull of structural endpoints (>=0.96pt, within 78% width) ===")
    max_x = page.rect.x0 + page.rect.width * 0.78
    all_pts = []
    for d in drawings:
        w = d.get("width") or 0
        if w < 0.96 or d.get("color") is None:
            continue
        for s in path_segs(d):
            for pt in s:
                if pt[0] < max_x:
                    all_pts.append(pt)
    if all_pts:
        from shapely.geometry import MultiPoint
        hull = MultiPoint(all_pts).convex_hull
        print(f"  {len(all_pts)} points, hull area={hull.area/page_area:.1%}")
        from shapely.geometry import box as shapely_box
        hb = hull.bounds
        print(f"  hull bounds=({hb[0]:.0f},{hb[1]:.0f},{hb[2]:.0f},{hb[3]:.0f})")

if __name__ == "__main__":
    main()
