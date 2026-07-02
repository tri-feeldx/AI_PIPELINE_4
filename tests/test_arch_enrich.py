"""ARCH -> 3D height enrichment (Phase 2.5.4).

build_level_table scans the ARCH set once (sections for level elevations,
plans for split-deck RL zones) and map_str_pages matches STR GA titles to
those levels so generate_building_ruby gets real FFLs and storey heights
instead of default_storey_height_mm.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.arch_ref.enrich import build_level_table, map_str_pages

PDF_ARCH = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_ARCH_Combine.pdf")
PDF_STR = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")

needs_pdfs = pytest.mark.skipif(
    not (PDF_ARCH.exists() and PDF_STR.exists()), reason="2381 PDFs needed")


@pytest.fixture(scope="module")
def table():
    return build_level_table(str(PDF_ARCH))


@needs_pdfs
class TestLevelTable2381:

    def test_levels_present_and_verified(self, table):
        for name in ("LEVEL GROUND", "LEVEL 01", "LEVEL 02", "LEVEL 10",
                     "TOP OF ROOF"):
            assert name in table.levels, sorted(table.levels)
        lv = table.levels["LEVEL 02"]
        assert lv.elevation_m == pytest.approx(10.880)
        assert lv.floor_to_floor_m == pytest.approx(3.0)
        assert lv.confidence == "VERIFIED"

    def test_split_deck_zones_attached(self, table):
        zones = table.levels["LEVEL 02"].zones
        rls = sorted(z["rl_m"] for z in zones)
        assert rls == pytest.approx([10.88, 12.38])
        assert all(len(z["positions"]) >= 2 for z in zones)


@needs_pdfs
class TestStrMapping2381:

    @pytest.fixture(scope="class")
    def mapping(self, table):
        return map_str_pages(str(PDF_STR), table)

    def _by_page(self, mapping, page_no):
        return next(m for m in mapping if m["page_no"] == page_no)

    def test_ga_pages_mapped(self, mapping):
        assert self._by_page(mapping, 16)["level_name"] == "LEVEL GROUND"
        assert self._by_page(mapping, 17)["level_name"] == "LEVEL 01"
        assert self._by_page(mapping, 18)["level_name"] == "LEVEL 02"
        assert self._by_page(mapping, 27)["level_name"] == "TOP OF ROOF"

    def test_ffl_and_height_values(self, mapping):
        p18 = self._by_page(mapping, 18)
        assert p18["ffl_mm"] == pytest.approx(10880.0)
        assert p18["height_mm"] == pytest.approx(3000.0)
        assert p18["confidence"] == "VERIFIED"

    def test_ground_storey_height(self, mapping):
        p16 = self._by_page(mapping, 16)
        assert p16["ffl_mm"] == pytest.approx(3380.0)
        assert p16["height_mm"] == pytest.approx(4500.0)


def test_empty_arch_gives_empty_table(tmp_path):
    doc = fitz.open()
    doc.new_page(width=800, height=600)
    p = tmp_path / "empty.pdf"
    doc.save(str(p))
    table = build_level_table(str(p))
    assert table.levels == {}
