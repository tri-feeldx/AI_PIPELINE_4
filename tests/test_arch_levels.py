"""ARCH cross-reference: level elevations from section sheets (Phase 2.5.1).

Ground truth measured on 2381_MSCP_ARCH_Combine.pdf p16 (SECTION A-E):
level datum text is real vector text ("LEVEL 01" + "3.380m"), repeated on
several section columns per sheet.  The parser must return one consensus
elevation per level and the derived floor-to-floor heights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.arch_ref.levels import extract_levels, LevelTable

PDF_ARCH = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_ARCH_Combine.pdf")


@pytest.mark.skipif(not PDF_ARCH.exists(), reason="ARCH PDF not present")
class TestArchLevels2381:

    @pytest.fixture(scope="class")
    def table(self) -> LevelTable:
        doc = fitz.open(str(PDF_ARCH))
        return extract_levels(doc[15])  # p16, SECTION A-E

    def test_all_levels_found(self, table):
        names = set(table.elevations_m)
        for want in ["LEVEL GROUND", "LEVEL 01", "LEVEL 05", "LEVEL 10",
                     "TOP OF ROOF"]:
            assert want in names, f"{want} missing from {sorted(names)}"

    def test_known_elevations(self, table):
        assert table.elevations_m["LEVEL GROUND"] == pytest.approx(3.380)
        assert table.elevations_m["LEVEL 01"] == pytest.approx(7.880)
        assert table.elevations_m["LEVEL 02"] == pytest.approx(10.880)
        assert table.elevations_m["LEVEL 10"] == pytest.approx(34.880)
        assert table.elevations_m["TOP OF ROOF"] == pytest.approx(37.880)

    def test_floor_to_floor(self, table):
        f2f = table.floor_to_floor_m
        assert f2f["LEVEL GROUND"] == pytest.approx(4.5)   # ground -> L01
        for lv in ["LEVEL 01", "LEVEL 02", "LEVEL 05", "LEVEL 09"]:
            assert f2f[lv] == pytest.approx(3.0), lv

    def test_consensus_no_conflicts(self, table):
        # several section columns on the sheet must agree
        assert table.conflicts == {}
        assert table.n_datums >= 20  # many datum labels sampled


def test_empty_page_returns_empty_table():
    doc = fitz.open()
    page = doc.new_page(width=1000, height=700)
    table = extract_levels(page)
    assert table.elevations_m == {}
    assert table.floor_to_floor_m == {}
