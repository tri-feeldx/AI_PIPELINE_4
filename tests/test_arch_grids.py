"""ARCH cross-reference: structural grid from bubbles (Phase 2.5.2).

Ground truth measured on the 2381 MSCP pair:
- ARCH p4 (1:200): bubbles 1..10 along the bottom (y~2181.6) and A..F on the
  left (x~77.9); axis 1-2 gap 151.7pt * 70.556 mm/pt ~ 10700 mm.
- STR p17 (1:100): same grid, bubbles drawn as 11-segment polylines, sheet
  mirrored (1 at x=2788, 10 at x=243); PT/RC tags also live inside circles
  and must be filtered out by the alignment rule.
Grid spacing in mm must match between the two files (same building).
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.arch_ref.grids import extract_grid

PDF_ARCH = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_ARCH_Combine.pdf")
PDF_STR = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")

MM_PER_PT = 25.4 / 72.0
# expected axis 1..10 spacing chain from the dim string on both files (mm)
EXPECTED_NUM_SPACING = [10700, 10800, 10800, 8400, 8400, 8400, 10800, 10800, 10700]


@pytest.mark.skipif(not PDF_ARCH.exists(), reason="ARCH PDF not present")
class TestArchGrid:

    @pytest.fixture(scope="class")
    def grid(self):
        doc = fitz.open(str(PDF_ARCH))
        return extract_grid(doc[3])  # p4

    def test_families_complete(self, grid):
        assert set(grid.cols) == {str(i) for i in range(1, 11)}
        assert set(grid.rows) == set("ABCDEF")

    def test_numeric_axis_spacing_matches_dims(self, grid):
        xs = [grid.cols[str(i)] for i in range(1, 11)]
        gaps_mm = [abs(b - a) * MM_PER_PT * 200 for a, b in zip(xs, xs[1:])]
        for got, want in zip(gaps_mm, EXPECTED_NUM_SPACING):
            assert got == pytest.approx(want, abs=25), (gaps_mm, EXPECTED_NUM_SPACING)


@pytest.mark.skipif(not PDF_STR.exists(), reason="STR PDF not present")
class TestStrGrid:

    @pytest.fixture(scope="class")
    def grid(self):
        doc = fitz.open(str(PDF_STR))
        return extract_grid(doc[16])  # STR GA p17

    def test_families_complete_and_tags_filtered(self, grid):
        assert set(grid.cols) == {str(i) for i in range(1, 11)}
        assert set(grid.rows) == set("ABCDEF")
        assert "PT" not in grid.cols and "PT" not in grid.rows
        assert "RC" not in grid.cols and "RC" not in grid.rows

    def test_numeric_axis_spacing_matches_dims(self, grid):
        xs = [grid.cols[str(i)] for i in range(1, 11)]
        gaps_mm = [abs(b - a) * MM_PER_PT * 100 for a, b in zip(xs, xs[1:])]
        for got, want in zip(gaps_mm, EXPECTED_NUM_SPACING):
            assert got == pytest.approx(want, abs=25), (gaps_mm, EXPECTED_NUM_SPACING)


@pytest.mark.skipif(not (PDF_ARCH.exists() and PDF_STR.exists()),
                    reason="both PDFs needed")
def test_arch_and_str_grids_agree():
    """Same building: axis spacing must agree across disciplines (< 25mm)."""
    arch = extract_grid(fitz.open(str(PDF_ARCH))[3])
    str_ = extract_grid(fitz.open(str(PDF_STR))[16])
    ax = [arch.cols[str(i)] for i in range(1, 11)]
    sx = [str_.cols[str(i)] for i in range(1, 11)]
    a_mm = [abs(b - a) * MM_PER_PT * 200 for a, b in zip(ax, ax[1:])]
    s_mm = [abs(b - a) * MM_PER_PT * 100 for a, b in zip(sx, sx[1:])]
    for a, s in zip(a_mm, s_mm):
        assert a == pytest.approx(s, abs=25)


def test_empty_page_gives_empty_grid():
    doc = fitz.open()
    grid = extract_grid(doc.new_page(width=1000, height=700))
    assert grid.cols == {} and grid.rows == {}
