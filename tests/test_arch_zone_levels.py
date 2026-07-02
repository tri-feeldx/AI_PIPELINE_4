"""ARCH cross-reference: split-deck zone elevations (Phase 2.5.1b).

2381 MSCP is a split-deck car park: every level sheet carries TWO RL spot
labels 1.5m apart (p4: RL 10.88 / RL 12.38 x3 pairs; p8: 22.88 / 24.38).
The single per-level elevation from levels.py is only the LOW deck, so
column heights must be resolved per position, not per level.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.arch_ref.zone_levels import extract_zone_levels
from src.arch_ref.levels import extract_levels

PDF_ARCH = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_ARCH_Combine.pdf")


@pytest.mark.skipif(not PDF_ARCH.exists(), reason="ARCH PDF not present")
class TestZones2381P4:
    """p4 = LEVEL 02 plan: low deck RL 10.88, high deck RL 12.38."""

    @pytest.fixture(scope="class")
    def zones(self):
        doc = fitz.open(str(PDF_ARCH))
        return extract_zone_levels(doc[3])

    def test_two_zones_found(self, zones):
        assert sorted(zones.zones) == [10.88, 12.38]

    def test_each_zone_confirmed_by_multiple_labels(self, zones):
        for rl, pts in zones.zones.items():
            assert len(pts) >= 2, (rl, pts)

    def test_elevation_at_positions(self, zones):
        # measured label positions: low (328,1711), high (328,1505)
        assert zones.elevation_at(328, 1740) == pytest.approx(10.88)
        assert zones.elevation_at(328, 1470) == pytest.approx(12.38)

    def test_low_zone_matches_level_table(self, zones):
        doc = fitz.open(str(PDF_ARCH))
        table = extract_levels(doc[15])  # section sheet
        assert min(zones.zones) == pytest.approx(
            table.elevations_m["LEVEL 02"], abs=0.005)

    def test_zone_split_is_1500mm(self, zones):
        vals = sorted(zones.zones)
        assert vals[1] - vals[0] == pytest.approx(1.5, abs=0.001)


def test_empty_page_has_no_zones():
    doc = fitz.open()
    zones = extract_zone_levels(doc.new_page(width=1000, height=700))
    assert zones.zones == {}
    assert zones.elevation_at(500, 350) is None
