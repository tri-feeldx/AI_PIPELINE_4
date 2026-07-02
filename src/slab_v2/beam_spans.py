"""Horizontal member spans from the structural grid (Phase 2.5.5).

axis_spans reads the grid bubbles of the sheet itself (arch_ref.grids
works on STR sheets too — bubbles are 11-segment polylines there) and
returns centre-to-centre spans between adjacent axes in real mm.
clear_span_mm subtracts the half-sizes of the end columns using the
on-page schedule types (schedule_parser).
"""
from __future__ import annotations

import fitz

from src.arch_ref.grids import extract_grid
from src.slab_v2.schedule_parser import ColumnType

MM_PER_PT = 25.4 / 72.0


def axis_spans(page: fitz.Page, scale: int) -> list[dict]:
    """[{a, b, span_mm}] for adjacent axes of both grid families."""
    grid = extract_grid(page)
    out = []
    for fam in (grid.cols, grid.rows):
        axes = sorted(fam.items(), key=lambda kv: kv[1])
        for (la, ca), (lb, cb) in zip(axes, axes[1:]):
            out.append({
                "a": la, "b": lb,
                "span_mm": round(abs(cb - ca) * MM_PER_PT * scale, 1),
            })
    return out


def clear_span_mm(centre_span_mm: float,
                  col_a: ColumnType | None,
                  col_b: ColumnType | None,
                  axis: str = "x") -> float:
    """Centre span minus half of each end column along the span direction.

    axis="x": use each column's first size dimension; "y": the second.
    Missing columns contribute nothing (span stays centre-to-centre).
    """
    idx = 0 if axis == "x" else 1
    span = centre_span_mm
    for col in (col_a, col_b):
        if col is not None and col.size_mm is not None:
            span -= col.size_mm[idx] / 2.0
    return span
