"""ARCH cross-reference demo CLI (Phase 2.5).

  python -X utf8 demo_arch_ref.py levels "<arch_pdf>" <page_1based>
  python -X utf8 demo_arch_ref.py zones  "<arch_pdf>" <page_1based>
  python -X utf8 demo_arch_ref.py grid   "<pdf>"      <page_1based>
  python -X utf8 demo_arch_ref.py match  "<arch_pdf>" <arch_page> "<str_pdf>" <str_page> [out_dir]

`match` prints both grids, the per-axis spacing in mm (using each sheet's
scale) and writes arch_ref_match.json + an overlay debug image into out_dir
(default: debug_arch_ref/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.arch_ref.levels import extract_levels
from src.arch_ref.grids import extract_grid

MM_PER_PT = 25.4 / 72.0


def _detect_scale(page: fitz.Page) -> int | None:
    """Most common explicit '1 : N' on the sheet (plan scales only)."""
    import re
    from collections import Counter
    hits = re.findall(r"1\s*:\s*(\d{2,4})\b", page.get_text())
    plausible = [int(h) for h in hits if int(h) in (50, 100, 150, 200, 250, 500)]
    return Counter(plausible).most_common(1)[0][0] if plausible else None


def cmd_levels(pdf: str, page_no: int) -> None:
    doc = fitz.open(pdf)
    table = extract_levels(doc[page_no - 1])
    print(f"{'level':<16} {'elev (m)':>9} {'floor-to-floor (m)':>19}")
    for name, z in sorted(table.elevations_m.items(), key=lambda kv: kv[1]):
        f2f = table.floor_to_floor_m.get(name)
        print(f"{name:<16} {z:>9.3f} {f2f if f2f is not None else '':>19}")
    if table.conflicts:
        print("CONFLICTS:", table.conflicts)
    print(f"({table.n_datums} datum labels sampled)")


def cmd_zones(pdf: str, page_no: int) -> None:
    from src.arch_ref.zone_levels import extract_zone_levels
    doc = fitz.open(pdf)
    zones = extract_zone_levels(doc[page_no - 1])
    if not zones.zones:
        print("no RL spot labels found on this page")
        return
    vals = sorted(zones.zones)
    print(f"{len(vals)} deck zone(s):")
    for rl in vals:
        pts = ", ".join(f"({x:.0f},{y:.0f})" for x, y in zones.zones[rl])
        print(f"  RL {rl:7.3f} m  confirmed by {len(zones.zones[rl])} labels at {pts}")
    if len(vals) >= 2:
        print(f"deck offset: {vals[1] - vals[0]:.3f} m")


def cmd_grid(pdf: str, page_no: int) -> None:
    doc = fitz.open(pdf)
    page = doc[page_no - 1]
    grid = extract_grid(page)
    scale = _detect_scale(page)
    print(f"scale 1:{scale}" if scale else "scale: not detected")
    for fam_name, fam in (("cols", grid.cols), ("rows", grid.rows)):
        axes = sorted(fam.items(), key=lambda kv: kv[1])
        print(f"{fam_name}: {' '.join(l for l, _ in axes)}")
        if scale:
            gaps = [f"{(b - a) * MM_PER_PT * scale:,.0f}"
                    for (_, a), (_, b) in zip(axes, axes[1:])]
            print(f"  spacing mm: {' | '.join(gaps)}")


def cmd_match(arch_pdf: str, arch_page: int, str_pdf: str, str_page: int,
              out_dir: str = "debug_arch_ref") -> None:
    a_doc, s_doc = fitz.open(arch_pdf), fitz.open(str_pdf)
    a_pg, s_pg = a_doc[arch_page - 1], s_doc[str_page - 1]
    a_grid, s_grid = extract_grid(a_pg), extract_grid(s_pg)
    a_scale, s_scale = _detect_scale(a_pg), _detect_scale(s_pg)
    report: dict = {
        "arch": {"pdf": arch_pdf, "page": arch_page, "scale": a_scale},
        "str": {"pdf": str_pdf, "page": str_page, "scale": s_scale},
        "axes": [],
    }
    shared_cols = sorted(set(a_grid.cols) & set(s_grid.cols),
                         key=lambda l: a_grid.cols[l])
    print(f"shared column axes: {' '.join(shared_cols)}")
    if a_scale and s_scale:
        ax = [a_grid.cols[l] for l in shared_cols]
        sx = [s_grid.cols[l] for l in shared_cols]
        print(f"{'span':<8} {'ARCH mm':>10} {'STR mm':>10} {'diff':>7}")
        for i in range(len(shared_cols) - 1):
            a_mm = abs(ax[i + 1] - ax[i]) * MM_PER_PT * a_scale
            s_mm = abs(sx[i + 1] - sx[i]) * MM_PER_PT * s_scale
            span = f"{shared_cols[i]}-{shared_cols[i+1]}"
            print(f"{span:<8} {a_mm:>10,.0f} {s_mm:>10,.0f} {a_mm - s_mm:>7.0f}")
            report["axes"].append({"span": span, "arch_mm": round(a_mm, 1),
                                   "str_mm": round(s_mm, 1),
                                   "diff_mm": round(a_mm - s_mm, 1)})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arch_ref_match.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {out / 'arch_ref_match.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "levels":
        cmd_levels(sys.argv[2], int(sys.argv[3]))
    elif cmd == "zones":
        cmd_zones(sys.argv[2], int(sys.argv[3]))
    elif cmd == "grid":
        cmd_grid(sys.argv[2], int(sys.argv[3]))
    elif cmd == "match":
        cmd_match(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]),
                  *(sys.argv[6:7] or []))
    else:
        print(__doc__)
