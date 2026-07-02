"""Batch evaluation across PDFs (Phase 4).

  python -X utf8 batch_eval.py [--pdfs a.pdf b.pdf ...] [--pages 1-30]
                               [--out-dir batch_eval_out]

Runs extract_slabs_v2 (AI off) on every page, one JSONL line per page:
{pdf, page, status, scale, slab_count, slab_m2, cols, walls, steel,
 openings, failure_reason, elapsed_s}; then writes summary.csv and
report.html with pass-rate per status taxonomy.  Crashes are caught and
reported as status=CRASH — no page may abort the batch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_PDFS = [
    r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf",
    r"C:\Users\LENOVO\Downloads\2402. South Melbourne Primary School - CIVIL & STR - 260610.pdf",
    r"C:\Users\LENOVO\Downloads\Combined Structural.pdf",
]


def eval_page(pdf: str, pi: int, cfg) -> dict:
    from src.slab_v2.pipeline import extract_slabs_v2
    t0 = time.time()
    row = {"pdf": Path(pdf).name, "page": pi + 1}
    try:
        r = extract_slabs_v2(pdf, pi, cfg, use_ai=False)
        row.update({
            "status": r.status,
            "scale": r.scale,
            "page_role": (r.page_role_classification or {}).get("role"),
            "slab_count": len(r.slabs),
            "slab_m2": round(sum(s.get("area_m2") or 0 for s in r.slabs), 1),
            "cols": len(r.columns),
            "walls": len(r.walls),
            "steel": len(r.steel_members),
            "openings": len(r.verified_cut_openings),
            "failure_reason": "" if r.status == "OK" else (
                r.warnings[-1][:160] if r.warnings else r.status),
        })
    except Exception as exc:                     # noqa: BLE001
        row.update({"status": "CRASH",
                    "failure_reason": f"{type(exc).__name__}: {exc}"[:200]})
        traceback.print_exc()
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def write_reports(rows: list[dict], out: Path) -> None:
    import csv
    keys = ["pdf", "page", "status", "page_role", "scale", "slab_count",
            "slab_m2", "cols", "walls", "steel", "openings", "elapsed_s",
            "failure_reason"]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_status = Counter(r["status"] for r in rows)
    n = len(rows)
    ok = by_status.get("OK", 0)
    html = ["<html><meta charset='utf-8'><body style='font-family:monospace'>",
            f"<h2>batch_eval — {n} pages, OK {ok} ({ok / max(n,1):.0%})</h2>",
            "<h3>Status taxonomy</h3><ul>"]
    for st, c in by_status.most_common():
        html.append(f"<li>{st}: {c}</li>")
    html.append("</ul><h3>Pages</h3><table border=1 cellpadding=3>")
    html.append("<tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>")
    for r in rows:
        color = ("#e8ffe8" if r["status"] == "OK"
                 else "#ffe8e8" if r["status"] == "CRASH" else "#fff8dc")
        html.append(f"<tr style='background:{color}'>" + "".join(
            f"<td>{r.get(k, '')}</td>" for k in keys) + "</tr>")
    html.append("</table></body></html>")
    (out / "report.html").write_text("\n".join(html), encoding="utf-8")


def main() -> int:
    import fitz
    from src.slab_v2.config import SlabV2Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="*", default=DEFAULT_PDFS)
    ap.add_argument("--pages", default=None,
                    help="1-based range like '1-30' (default: all)")
    ap.add_argument("--out-dir", default="batch_eval_out")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SlabV2Config(debug_images=False, enable_opening_judge=False,
                       enable_slab_face_judge=False,
                       enable_floor_system_judge=False)

    rows = []
    jsonl = open(out / "pages.jsonl", "w", encoding="utf-8")
    for pdf in args.pdfs:
        if not Path(pdf).exists():
            print(f"SKIP missing {pdf}")
            continue
        doc = fitz.open(pdf)
        n = len(doc)
        doc.close()
        idxs = range(n)
        if args.pages:
            a, _, b = args.pages.partition("-")
            idxs = range(int(a) - 1, min(int(b or a), n))
        for pi in idxs:
            row = eval_page(pdf, pi, cfg)
            rows.append(row)
            jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl.flush()
            print(f"{row['pdf'][:36]:<36} p{row['page']:>3} "
                  f"{row['status']:<22} {row.get('slab_m2', ''):>9} "
                  f"{row['elapsed_s']:>6}s")
    jsonl.close()
    write_reports(rows, out)
    print(f"\nreports -> {out / 'summary.csv'}, {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
