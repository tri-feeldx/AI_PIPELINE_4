"""On-page column/wall schedule parsing (Phase 2.1).

2381 GA sheets print CONCRETE COLUMN SCHEDULE and WALL SCHEDULE as real
vector text.  Parsing them deterministically gives per-page element types
(sizes, thicknesses, reinforcement rates) — Gemini census becomes a
cross-check, not the source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.slab_v2.schedule_parser import parse_schedules

PDF_2381 = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")

EXPECTED_COLUMNS = {
    "C-A1": (450, 1200), "C-A2": (450, 1200), "C-A3": (450, 1200),
    "C-B": (450, 1000), "C-C": (450, 800), "C-D": (400, 600),
    "C-E": (350, 350), "C-F": (350, 800),
}
EXPECTED_WALL_THICKNESS = {
    "BW1": 190, "IW20": 200, "IW25": 250, "IW30": 300, "IW35": 350,
    "NLB1": 190,
}


@pytest.mark.skipif(not PDF_2381.exists(), reason="2381 PDF not present")
class TestSchedules2381P17:

    @pytest.fixture(scope="class")
    def sched(self):
        doc = fitz.open(str(PDF_2381))
        return parse_schedules(doc[16])  # GA LEVEL 01

    def test_all_column_types_found(self, sched):
        assert set(sched.columns) == set(EXPECTED_COLUMNS)

    def test_column_sizes_exact(self, sched):
        for mark, (w, h) in EXPECTED_COLUMNS.items():
            assert sched.columns[mark].size_mm == (w, h), mark

    def test_column_reinforcement_rates(self, sched):
        assert sched.columns["C-A1"].reinforcement_rate_kg_m3 == 350
        assert sched.columns["C-F"].reinforcement_rate_kg_m3 == 300

    def test_all_wall_types_found(self, sched):
        assert set(sched.walls) == set(EXPECTED_WALL_THICKNESS)

    def test_wall_thicknesses_exact(self, sched):
        for mark, t in EXPECTED_WALL_THICKNESS.items():
            assert sched.walls[mark].thickness_mm == t, mark


def test_rebar_mass_from_rate():
    from src.slab_v2.schedule_parser import ColumnType
    c = ColumnType(mark="C-A1", size_mm=(450, 1200),
                   reinforcement_rate_kg_m3=350)
    # 0.45 * 1.2 * 3.0 m3 * 350 kg/m3 = 567 kg
    assert c.rebar_mass_kg(3000) == pytest.approx(567.0)
    # split-deck short column: 1.5m storey
    assert c.rebar_mass_kg(1500) == pytest.approx(283.5)
    assert ColumnType(mark="X").rebar_mass_kg(3000) is None


def test_page_without_schedules_returns_empty():
    doc = fitz.open()
    page = doc.new_page(width=1000, height=700)
    sched = parse_schedules(page)
    assert sched.columns == {} and sched.walls == {}
