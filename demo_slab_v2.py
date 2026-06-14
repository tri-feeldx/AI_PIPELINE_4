"""
slab_v2 demo CLI — no Streamlit needed.

  python demo_slab_v2.py profile "C:\\path\\to\\drawings.pdf" [--top 10]
  python demo_slab_v2.py extract "C:\\path\\to\\drawings.pdf" --page 62 [--no-ai] [--scale 100]
  python demo_slab_v2.py extract "C:\\path\\to\\drawings.pdf" --pages 60-68

profile  dumps per-page drawing counts + style-class tables to
         debug_slab_v2/<stem>/profile.json and renders step_01 images for
         the --top pages by drawing count.
extract  runs the pipeline on the given page(s); --no-ai stops after the
         deterministic kernel (steps 00-04).
build    extracts several pages (one per storey) and writes ONE .rb that
         stacks them at their FFL elevations:
         python demo_slab_v2.py build "drawings.pdf" --pages 8,10,11
             [--ffl "8=0,10=3.2,11=6.4"] [--out building.rb]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz

from src.slab_v2.config import SlabV2Config
from src.slab_v2 import vector_extract
from src.slab_v2.pipeline import extract_slabs_v2, _content_rect, run_dir
from src.slab_v2.debug_render import PageRenderer


def cmd_profile(args) -> int:
    cfg = SlabV2Config()
    doc = fitz.open(args.pdf)
    out_root = run_dir(cfg, args.pdf)

    pages = []
    for i in range(len(doc)):
        page = doc[i]
        content = _content_rect(page)
        paths, classes = vector_extract.extract_paths(page, cfg, content)
        n_segs = sum(p_.n_segments for p_ in classes)
        pages.append({
            "page": i + 1,
            "size_pt": [round(page.rect.width), round(page.rect.height)],
            "n_drawings": len(paths),
            "n_segments": n_segs,
            "n_style_classes": len(classes),
            "has_fill_classes": any(c.key.fill is not None for c in classes),
            "style_classes": vector_extract.class_summary_table(classes)[:15],
        })
        print(f"p{i + 1}: paths={len(paths):5d} segs={n_segs:6d} "
              f"classes={len(classes):3d}")

    with open(out_root / "profile.json", "w", encoding="utf-8") as fh:
        json.dump(pages, fh, indent=2)
    print(f"\nprofile.json -> {out_root / 'profile.json'}")

    # render step_01 for the busiest pages
    top = sorted(pages, key=lambda p: -p["n_segments"])[: args.top]
    for info in top:
        i = info["page"] - 1
        page = doc[i]
        content = _content_rect(page)
        paths, classes = vector_extract.extract_paths(page, cfg, content)
        rend = PageRenderer(page, cfg, str(out_root / f"page_{i + 1}"))
        path = rend.step01_paths_by_style(paths, classes)
        print(f"step_01 -> {path}")
    return 0


def _parse_page_spec(spec: str) -> list[int]:
    """'8,10,11' or '8-11' or '8,10-12' (1-based) -> 0-based indices."""
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.extend(range(int(a) - 1, int(b or a)))
    return sorted(set(out))


def _parse_pages(args) -> list[int]:
    if args.page is not None:
        return [args.page - 1]
    if args.pages:
        return _parse_page_spec(args.pages)
    print("ERROR: provide --page N or --pages A-B", file=sys.stderr)
    sys.exit(2)


def cmd_extract(args) -> int:
    cfg = SlabV2Config()
    ok = True
    for pi in _parse_pages(args):
        print(f"\n=== page {pi + 1} ===")
        result = extract_slabs_v2(
            args.pdf, pi, cfg, use_ai=not args.legacy_no_ai,
            scale=args.scale)
        print(f"status={result.status} scale={result.scale} "
              f"gemini_calls={result.gemini_calls}")
        v = result.verification
        if v and v.edge_matches:
            worst = max(m["rel_err"] for m in v.edge_matches)
            print(f"  dims: {v.n_dims_associated} associated, "
                  f"{len(v.edge_matches)} edges matched, "
                  f"max err {worst:.2%}"
                  + (f", precise scale 1:{v.scale_precise:.2f}"
                     if v.scale_precise else ""))
        for s in result.slabs:
            print(f"  {s['label']}: area={s.get('area_m2')} m2")
        for e in result.elements:
            print(f"  opening: {e.type} '{e.label}'")
        for w in result.warnings:
            print(f"  WARN: {w}")
        print(f"debug -> {result.debug_dir}")
        if result.status != "OK":
            ok = False
    return 0 if ok else 1


def cmd_export(args) -> int:
    import fitz
    from src.slab_v2.export_ruby import generate_ruby

    cfg = SlabV2Config()
    pi = args.page - 1
    result = extract_slabs_v2(args.pdf, pi, cfg,
                              use_ai=not args.legacy_no_ai,
                              scale=args.scale)
    print(f"status={result.status} slabs={len(result.slabs)} "
          f"elements={len(result.elements)}")
    if not result.slabs:
        print("no slabs to export", file=sys.stderr)
        return 1
    doc = fitz.open(args.pdf)
    out = args.out or str(run_dir(cfg, args.pdf) / f"slab_v2_p{args.page}.rb")
    path = generate_ruby(result, doc[pi], out, cfg)
    print(f"ruby -> {path}")
    for e in result.elements:
        print(f"  opening: {e.type} '{e.label}'")
    return 0


# strict FFL pattern: keyword must not be part of a word (v1's FFL_PATTERN
# matches "LEVEL 03" via the embedded "EL 03" -> fake 3.0 m), and bare
# EL/FL are dropped because they collide with LEVEL/FLOOR
_FFL_STRICT_RE = re.compile(
    r"(?<![A-Za-z])(?:FFL|SSL|RL|AHD|NGL)\s*[=:+]?\s*"
    r"([+-]?\d{1,4}(?:\.\d{1,3})?)\s*(?:m\b|mAHD\b)?", re.I)

_LEVEL_NO_RE = re.compile(
    r"\bLEVEL\s*0?(\d{1,2})\b|\bL(\d{1,2})\s+(?:OUTLINE|SLAB|FLOOR)\b", re.I)


def _dominant_ffl_m(page) -> float | None:
    """Most-mentioned FFL value on the page (same idea as floor_detector)."""
    from collections import Counter
    counts = Counter(round(float(m.group(1)), 3)
                     for m in _FFL_STRICT_RE.finditer(page.get_text()))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _level_number(page) -> int | None:
    """Most-mentioned storey number ('LEVEL 03 OUTLINE PLAN') — ordering
    fallback for pages without a real FFL annotation."""
    from collections import Counter
    counts = Counter(int(m.group(1) or m.group(2))
                     for m in _LEVEL_NO_RE.finditer(page.get_text()))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def cmd_build(args) -> int:
    """Extract several pages and stack them into one building .rb by FFL."""
    from src.slab_v2.export_ruby import generate_building_ruby

    cfg = SlabV2Config()
    doc = fitz.open(args.pdf)
    page_idx = _parse_page_spec(args.pages)

    ffl_override: dict[int, float] = {}
    if args.ffl:
        for part in args.ffl.split(","):
            p, _, v = part.partition("=")
            ffl_override[int(p) - 1] = float(v)

    storeys, missing = [], []
    for pi in page_idx:
        print(f"\n=== page {pi + 1} ===")
        result = extract_slabs_v2(args.pdf, pi, cfg,
                                  use_ai=not args.legacy_no_ai,
                                  scale=args.scale)
        if result.status != "OK" or not result.slabs:
            print(f"  SKIP: status={result.status}", file=sys.stderr)
            continue
        ffl_m = ffl_override.get(pi)
        if ffl_m is None:
            ffl_m = _dominant_ffl_m(doc[pi])
        st = {"result": result, "page": doc[pi],
              "ffl_mm": None if ffl_m is None else ffl_m * 1000.0}
        (missing if ffl_m is None else storeys).append(st)
        for w in result.warnings:
            print(f"  WARN: {w}")
    if not storeys and not missing:
        print("no storeys extracted", file=sys.stderr)
        return 1

    # pages without an FFL stack above the highest known level, ordered by
    # the LEVEL number in the drawing title (page order is unreliable —
    # sheet sets often run top floor first), default storey height apart
    if missing:
        def order_key(s):
            lvl = _level_number(doc[s["result"].page_index])
            return (lvl is None, lvl, s["result"].page_index)
        top = max((s["ffl_mm"] for s in storeys), default=None)
        z = 0.0 if top is None else top + cfg.default_storey_height_mm
        for st in sorted(missing, key=order_key):
            st["ffl_mm"] = z
            lvl = _level_number(doc[st["result"].page_index])
            print(f"WARN: page {st['result'].page_index + 1}: no FFL found "
                  f"— stacked at +{z / 1000:.3f}m by "
                  f"{'LEVEL ' + str(lvl) if lvl is not None else 'page order'}"
                  f" ({cfg.default_storey_height_mm:.0f}mm storeys)")
            z += cfg.default_storey_height_mm
        storeys += missing

    out = args.out or str(run_dir(cfg, args.pdf) / "slab_v2_building.rb")
    path, warnings = generate_building_ruby(storeys, out, cfg)

    print(f"\n{'page':>5} {'FFL (m)':>9} {'area m2':>9} {'openings':>9}")
    for st in sorted(storeys, key=lambda s: s["ffl_mm"]):
        r = st["result"]
        area = sum(s.get("area_m2") or 0 for s in r.slabs)
        print(f"{r.page_index + 1:>5} {st['ffl_mm'] / 1000:>+9.3f} "
              f"{area:>9.1f} {len(r.elements):>9}")
    for w in warnings:
        print(f"WARN: {w}")
    print(f"ruby -> {path}")
    return 0


def cmd_auto(args) -> int:
    """End-to-end: Gemini document analysis -> per-floor extraction
    (slabs + columns) -> one .rb per building, stacked by FFL."""
    from src.slab_v2.doc_analyze import analyze_document
    from src.slab_v2.export_ruby import generate_building_ruby

    cfg = SlabV2Config()
    doc = fitz.open(args.pdf)
    stem = Path(args.pdf).stem.replace(" ", "_")

    ana = analyze_document(args.pdf, cfg)
    print(f"=== document analysis ({ana.confidence}) ===")
    for b in ana.buildings:
        print(f"building: {b.name}")
        for f in b.floors:
            print(f"  {f.level_id:<12} FFL="
                  f"{'?' if f.ffl_m is None else f'{f.ffl_m:+.3f}m'} "
                  f"pages={[p + 1 for p in f.pages]}")
    if ana.column_types:
        types = ", ".join(f"{t.symbol} {t.width_mm:.0f}x{t.depth_mm:.0f}"
                          for t in ana.column_types.values())
        print(f"column schedule: {types} "
              f"(pages {[p + 1 for p in ana.column_schedule_pages]})")
    else:
        print("column schedule: none found")
    print(f"stair detail pages (parked for a later round): "
          f"{[p + 1 for p in ana.stair_detail_pages]}")
    print(f"lift detail pages (parked): "
          f"{[p + 1 for p in ana.lift_detail_pages]}")
    for w in ana.warnings:
        print(f"WARN: {w}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir(cfg, args.pdf)
    rc = 0
    for b in ana.buildings:
        if args.building and args.building.lower() not in b.name.lower():
            continue
        print(f"\n=== building: {b.name} ===")
        storeys, prev_ffl = [], None
        for f in b.floors:
            ffl_m = f.ffl_m
            if ffl_m is None:
                ffl_m = (prev_ffl or 0.0) + cfg.default_storey_height_mm \
                    / 1000.0
                print(f"WARN: {f.level_id}: no FFL from analysis — "
                      f"stacked at +{ffl_m:.3f}m")
            prev_ffl = ffl_m
            census = next(
                (e["counts"] for e in ana.columns_per_floor
                 if e["level_id"] == f.level_id
                 and (not e.get("building") or e["building"] == b.name)),
                None)
            for pi in f.pages:
                print(f"--- {f.level_id} page {pi + 1} ---")
                result = extract_slabs_v2(
                    args.pdf, pi, cfg, use_ai=not args.legacy_no_ai,
                    scale=args.scale, column_types=ana.column_types,
                    columns_per_floor=census)
                if result.status != "OK" or not result.slabs:
                    print(f"  SKIP: status={result.status}", file=sys.stderr)
                    rc = 1
                    continue
                area = sum(s.get("area_m2") or 0 for s in result.slabs)
                print(f"  area={area:.1f} m2  openings="
                      f"{len(result.elements)}  columns="
                      f"{len(result.columns)}  scale={result.scale}")
                for w in result.warnings:
                    print(f"  WARN: {w}")
                storeys.append({"result": result, "page": doc[pi],
                                "ffl_mm": ffl_m * 1000.0})
            # census cross-check, warning only
            census = next(
                (e["counts"] for e in ana.columns_per_floor
                 if e["level_id"] == f.level_id
                 and (not e["building"] or e["building"] == b.name)), None)
            if census:
                expect = sum(census.values())
                got = sum(len(st["result"].columns) for st in storeys
                          if st["result"].page_index in f.pages)
                if expect and got != expect:
                    print(f"  WARN: census expects {expect} column(s) on "
                          f"{f.level_id}, detected {got}")
        if not storeys:
            print("  no storeys extracted", file=sys.stderr)
            rc = 1
            continue
        # keep junk levels out of the customer model: a "floor" whose
        # largest single slab is tiny vs the building's largest single slab
        # is a failed extraction (roof/steel sheets produce many small
        # fragments that can sum to a big total but no single real slab)
        def _max_slab(st):
            return max((s.get("area_m2") or 0 for s in st["result"].slabs),
                       default=0)
        biggest = max(_max_slab(st) for st in storeys)
        kept = []
        for st in storeys:
            a = _max_slab(st)
            if biggest > 0 and a < 0.10 * biggest:
                total = sum(s.get("area_m2") or 0
                            for s in st["result"].slabs)
                n = len(st["result"].slabs)
                print(f"  WARN: page {st['result'].page_index + 1} largest "
                      f"slab {a:.1f} m2 ({n} fragments, {total:.1f} m2 "
                      f"total) is <10% of the building max "
                      f"({biggest:.1f} m2) — left out of .rb")
            else:
                kept.append(st)
        storeys = kept
        bid = "".join(ch if ch.isalnum() else "_" for ch in b.name).strip("_")
        out = out_dir / f"{stem}_{bid}.rb"
        path, warnings = generate_building_ruby(storeys, str(out), cfg)
        for w in warnings:
            print(f"  WARN: {w}")
        print(f"  ruby -> {path}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="slab_v2 demo CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("profile", help="per-page vector profile")
    p1.add_argument("pdf")
    p1.add_argument("--top", type=int, default=6,
                    help="render step_01 for N busiest pages")
    p1.set_defaults(fn=cmd_profile)

    p2 = sub.add_parser("extract", help="run the slab_v2 pipeline")
    p2.add_argument("pdf")
    p2.add_argument("--page", type=int, help="1-based page number")
    p2.add_argument("--pages", help="1-based inclusive range, e.g. 60-68")
    p2.add_argument("--scale", type=int, default=None,
                    help="drawing scale denominator, e.g. 100 for 1:100")
    p2.add_argument("--no-ai", dest="legacy_no_ai", action="store_true",
                    help="skip Gemini class election: union faces of ALL "
                         "classes (debug/fallback — may over-include)")
    p2.set_defaults(fn=cmd_extract)

    p3 = sub.add_parser("export", help="extract + write SketchUp Ruby script")
    p3.add_argument("pdf")
    p3.add_argument("--page", type=int, required=True)
    p3.add_argument("--scale", type=int, default=None)
    p3.add_argument("--out", default=None, help="output .rb path")
    p3.add_argument("--no-ai", dest="legacy_no_ai", action="store_true")
    p3.set_defaults(fn=cmd_export)

    p4 = sub.add_parser(
        "build", help="extract several pages and stack them by FFL into "
                      "one building .rb")
    p4.add_argument("pdf")
    p4.add_argument("--pages", required=True,
                    help="1-based pages: '8,10,11' or '8-11'")
    p4.add_argument("--ffl", default=None,
                    help="override FFL metres per page, e.g. "
                         "'8=0,10=3.2,11=6.4' (default: detected from text)")
    p4.add_argument("--scale", type=int, default=None)
    p4.add_argument("--out", default=None, help="output .rb path")
    p4.add_argument("--no-ai", dest="legacy_no_ai", action="store_true")
    p4.set_defaults(fn=cmd_build)

    p5 = sub.add_parser(
        "auto", help="Gemini doc analysis -> slabs+columns per floor -> "
                     "one .rb per building")
    p5.add_argument("pdf")
    p5.add_argument("--building", default=None,
                    help="only process buildings whose name contains this")
    p5.add_argument("--scale", type=int, default=None)
    p5.add_argument("--out-dir", default=None,
                    help="output directory (default debug_slab_v2)")
    p5.add_argument("--no-ai", dest="legacy_no_ai", action="store_true")
    p5.set_defaults(fn=cmd_auto)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
