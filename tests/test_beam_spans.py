"""Horizontal member spans from the structural grid (Phase 2.5.5).

Span between two adjacent grid axes = centre distance (from the grid
bubbles on the STR sheet itself) minus half of each end column's size —
column sizes come from the on-page schedule (schedule_parser).
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.slab_v2.beam_spans import axis_spans, clear_span_mm
from src.slab_v2.schedule_parser import ColumnType

PDF_STR = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")

# axis 1..10 centre spacings from the dim chain (mm)
EXPECTED = [10700, 10800, 10800, 8400, 8400, 8400, 10800, 10800, 10700]


@pytest.mark.skipif(not PDF_STR.exists(), reason="STR PDF not present")
def test_axis_spans_match_dim_chain():
    doc = fitz.open(str(PDF_STR))
    spans = axis_spans(doc[16], scale=100)  # GA LEVEL 01
    num = [s for s in spans if s["a"].isdigit()]
    assert len(num) == 9
    for s, want in zip(num, EXPECTED):
        assert s["span_mm"] == pytest.approx(want, abs=25), (s, want)


def test_clear_span_subtracts_column_halves():
    a = ColumnType(mark="C-A1", size_mm=(450, 1200))
    b = ColumnType(mark="C-C", size_mm=(450, 800))
    # span along the 450 faces of both columns
    assert clear_span_mm(10700, a, b, axis="x") == pytest.approx(
        10700 - 225 - 225)
    # along the deep faces
    assert clear_span_mm(10700, a, b, axis="y") == pytest.approx(
        10700 - 600 - 400)


def test_clear_span_without_columns_is_centre_span():
    assert clear_span_mm(8400, None, None) == 8400
