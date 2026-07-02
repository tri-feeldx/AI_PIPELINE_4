"""Content-rect must be valid regardless of where the LEGEND text sits.

Bug (2026-07-02): find_drawing_content_rect() assumed the legend is a
right-side panel and used legend.x0 as the content cut.  On GA sheets with
the legend at the bottom-left (2381 MSCP p16-p27) this produced an inverted
rect (x1 = -4 < x0 = 101), so every downstream %-of-content computation ran
against an empty area and the no-fill pages fail-closed with
NO_EXPORT_TINY_SLAB.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.vision_refiner import find_drawing_content_rect, find_legend_rect

PDF_2381 = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")

PW, PH = 3371.0, 2384.0  # A0-ish landscape like the BVN sheets


def _make_sheet(legend_xy: tuple[float, float] | None) -> fitz.Page:
    """Synthetic sheet: outer frame, title-block divider, optional LEGEND text."""
    doc = fitz.open()
    page = doc.new_page(width=PW, height=PH)
    shape = page.new_shape()
    # outer drawing frame
    shape.draw_rect(fitz.Rect(57, 57, PW - 57, PH - 57))
    # title-block divider: full-height vertical line on the right
    shape.draw_line(fitz.Point(PW * 0.924, 57), fitz.Point(PW * 0.924, PH - 57))
    shape.finish(color=(0, 0, 0), width=1.0)
    shape.commit()
    if legend_xy is not None:
        page.insert_text(fitz.Point(*legend_xy), "LEGEND:", fontsize=12)
    return page


def _assert_valid(rect: fitz.Rect, page: fitz.Page) -> None:
    assert rect.x1 > rect.x0 and rect.y1 > rect.y0, f"inverted rect {rect}"
    frac = (rect.x1 - rect.x0) * (rect.y1 - rect.y0) / \
        (page.rect.width * page.rect.height)
    assert frac >= 0.5, f"content rect covers only {frac:.0%} of the page"


def test_legend_bottom_left_gives_valid_content_rect():
    page = _make_sheet(legend_xy=(105, PH * 0.86))
    legend = find_legend_rect(page)
    rect = find_drawing_content_rect(page, legend)
    _assert_valid(rect, page)
    # the plan area right of the legend must be inside the content rect
    assert rect.x1 > PW * 0.6


def test_legend_right_panel_still_cuts_at_legend():
    page = _make_sheet(legend_xy=(PW * 0.83, PH * 0.7))
    legend = find_legend_rect(page)
    rect = find_drawing_content_rect(page, legend)
    _assert_valid(rect, page)
    assert rect.x1 <= legend.x0, "right-panel legend must stay outside content"


def test_no_legend_text_gives_valid_content_rect():
    page = _make_sheet(legend_xy=None)
    legend = find_legend_rect(page)
    rect = find_drawing_content_rect(page, legend)
    _assert_valid(rect, page)


@pytest.mark.skipif(not PDF_2381.exists(), reason="2381 PDF not present")
@pytest.mark.parametrize("page_no", [16, 17, 18, 24, 27])
def test_2381_ga_pages_content_rect_valid(page_no):
    doc = fitz.open(str(PDF_2381))
    page = doc[page_no - 1]
    rect = find_drawing_content_rect(page, find_legend_rect(page))
    _assert_valid(rect, page)
    # GA plan spans most of the sheet width left of the title block
    assert rect.x1 > page.rect.width * 0.8


@pytest.mark.skipif(not PDF_2381.exists(), reason="2381 PDF not present")
def test_2381_p5_right_legend_unchanged():
    """p5 (loading plan, legend on the right) worked before — keep it."""
    doc = fitz.open(str(PDF_2381))
    page = doc[4]
    rect = find_drawing_content_rect(page, find_legend_rect(page))
    _assert_valid(rect, page)
    assert 2600 < rect.x1 < 2800  # was 2722 pre-fix
