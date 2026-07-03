"""Split-deck slab partition + STR->ARCH axis mapping (Bước 3).

Generic n-zone design: 1 zone = flat floor passes through unchanged;
n zones = Voronoi partition of the RL label points clipped to the slab.
Coordinates cross drawings via a per-axis linear fit on shared grid
labels (handles the mirrored 2381 STR sheets and any scale pair).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import box

from src.arch_ref.zone_split import split_polygon_by_zones, fit_axis_map

PDF_STR = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")
PDF_ARCH = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_ARCH_Combine.pdf")


class TestSplitPolygon:

    def test_single_zone_unchanged(self):
        slab = box(0, 0, 100, 80)
        parts = split_polygon_by_zones(slab, {10.0: [(50, 40)]})
        assert len(parts) == 1
        rl, poly = parts[0]
        assert rl == 10.0 and poly.equals(slab)

    def test_two_zones_split_roughly_in_half(self):
        slab = box(0, 0, 100, 80)
        zones = {10.0: [(20, 20), (20, 60)],
                 11.5: [(80, 20), (80, 60)]}
        parts = dict(split_polygon_by_zones(slab, zones))
        assert set(parts) == {10.0, 11.5}
        total = sum(p.area for p in parts.values())
        assert total == pytest.approx(slab.area, rel=1e-6)
        assert parts[10.0].area == pytest.approx(slab.area / 2, rel=0.05)
        # boundary is the perpendicular bisector x=50
        assert parts[10.0].bounds[2] == pytest.approx(50, abs=1)

    def test_three_zones(self):
        slab = box(0, 0, 90, 30)
        zones = {1.0: [(15, 15), (15, 5)], 2.0: [(45, 15), (45, 5)],
                 3.0: [(75, 15), (75, 5)]}
        parts = dict(split_polygon_by_zones(slab, zones))
        assert len(parts) == 3
        for p in parts.values():
            assert p.area == pytest.approx(900, rel=0.05)


class TestAxisMap:

    def test_mirrored_scaled_fit(self):
        # STR mirrored in x at half scale: x_arch = -0.5*x_str + 200
        pairs = [(100.0, 150.0), (300.0, 50.0), (500.0, -50.0)]
        f = fit_axis_map(pairs)
        assert f(200.0) == pytest.approx(100.0)
        assert f(0.0) == pytest.approx(200.0)

    @pytest.mark.skipif(not (PDF_STR.exists() and PDF_ARCH.exists()),
                        reason="2381 PDFs needed")
    def test_real_grid_mapping_2381(self):
        import fitz
        from src.arch_ref.grids import extract_grid
        str_g = extract_grid(fitz.open(str(PDF_STR))[17])    # STR p18 (L02)
        arch_g = extract_grid(fitz.open(str(PDF_ARCH))[3])   # ARCH p4 (L02)
        shared = sorted(set(str_g.cols) & set(arch_g.cols), key=int)
        assert len(shared) >= 5
        fx = fit_axis_map([(str_g.cols[l], arch_g.cols[l]) for l in shared])
        # every shared axis must land within 3pt on the ARCH sheet
        for l in shared:
            assert fx(str_g.cols[l]) == pytest.approx(arch_g.cols[l], abs=3.0)
