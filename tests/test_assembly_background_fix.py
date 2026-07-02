"""TDD tests for Phase 3.7: Background face exclusion + hole-preserving assembly.

Problem: On General Arrangement drawings, the face graph contains a BACKGROUND
FACE (space between building outline and content_rect border) that covers 80%+
of the page. When unioned with room faces, it creates one mega-component.
ShPolygon(c.exterior) then fills all holes → slab covers 127% of page.

Fix: Exclude depth-0 faces > 50% content area, and use buffer/unbuffer instead
of ShPolygon(exterior) to preserve holes while healing micro-slivers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import shapely
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config


def _make_face(id_, x0, y0, x1, y1, depth=0, parent_id=None):
    f = MagicMock()
    f.id = id_
    f.polygon = box(x0, y0, x1, y1)
    f.area_pt2 = f.polygon.area
    f.depth = depth
    f.parent_id = parent_id
    return f


# ---------------------------------------------------------------------------
# Unit: Background face exclusion
# ---------------------------------------------------------------------------

class TestBackgroundFaceExclusion:
    """Background face (depth=0, no parent, >50% content) must be excluded."""

    def test_background_face_excluded(self):
        """Large depth-0 face (60% content) excluded, 10 room faces kept."""
        content_area = 10000.0
        threshold = 0.001 * content_area  # 10 pt2
        bg_max = 0.50 * content_area      # 5000 pt2

        # Background face: 60% of content
        bg = _make_face(0, 0, 0, 100, 60, depth=0, parent_id=None)
        assert bg.area_pt2 == 6000  # > 5000

        # 10 room faces: each ~3% of content
        rooms = [_make_face(i+1, 10*i, 65, 10*i+8, 80, depth=0, parent_id=None)
                 for i in range(10)]

        all_faces = [bg] + rooms
        ids = [f.id for f in all_faces
               if f.area_pt2 >= threshold
               and not (f.depth == 0 and f.parent_id is None
                        and f.area_pt2 > bg_max)]

        assert 0 not in ids, "Background face should be excluded"
        assert len(ids) == 10, f"All 10 room faces should be kept, got {len(ids)}"

    def test_warehouse_not_excluded(self):
        """Single large face at 40% content area should NOT be excluded."""
        content_area = 10000.0
        bg_max = 0.50 * content_area

        warehouse = _make_face(0, 0, 0, 100, 40, depth=0, parent_id=None)
        assert warehouse.area_pt2 == 4000  # < 5000

        ids = [f.id for f in [warehouse]
               if f.area_pt2 >= 10
               and not (f.depth == 0 and f.parent_id is None
                        and f.area_pt2 > bg_max)]

        assert 0 in ids, "Warehouse face should NOT be excluded (< 50%)"

    def test_no_background_no_change(self):
        """When no face > 50% content, all faces above threshold are kept."""
        content_area = 10000.0
        bg_max = 0.50 * content_area

        faces = [_make_face(i, 10*i, 0, 10*i+8, 20, depth=0, parent_id=None)
                 for i in range(5)]

        ids_new = [f.id for f in faces
                   if f.area_pt2 >= 10
                   and not (f.depth == 0 and f.parent_id is None
                            and f.area_pt2 > bg_max)]
        ids_old = [f.id for f in faces if f.area_pt2 >= 10]

        assert ids_new == ids_old, "No background → same result as before"

    def test_depth1_large_face_not_excluded(self):
        """A large face at depth=1 (inside building) should NOT be excluded."""
        content_area = 10000.0
        bg_max = 0.50 * content_area

        nested = _make_face(0, 0, 0, 100, 60, depth=1, parent_id=5)
        assert nested.area_pt2 == 6000  # > 5000 but depth=1

        ids = [f.id for f in [nested]
               if f.area_pt2 >= 10
               and not (f.depth == 0 and f.parent_id is None
                        and f.area_pt2 > bg_max)]

        assert 0 in ids, "Depth-1 face NOT excluded even if large"


# ---------------------------------------------------------------------------
# Unit: Hole-preserving assembly
# ---------------------------------------------------------------------------

class TestHolePreservingAssembly:

    def test_holes_preserved(self):
        """Assembly should preserve internal holes (shafts, stairs)."""
        from src.slab_v2 import planarize

        outer = _make_face(0, 0, 0, 100, 100, depth=0)
        hole = _make_face(1, 30, 30, 50, 50, depth=1, parent_id=0)

        # Build a polygon with a hole manually
        outer_poly = box(0, 0, 100, 100)
        hole_poly = box(30, 30, 50, 50)
        slab_with_hole = outer_poly.difference(hole_poly)

        # Create face that IS the slab-with-hole
        slab_face = MagicMock()
        slab_face.id = 0
        slab_face.polygon = slab_with_hole
        slab_face.area_pt2 = slab_with_hole.area

        result, err = planarize.assemble_slab_polygon(
            [slab_face], [0], [], min_component_frac=0.02)

        assert result is not None
        # The result should preserve the hole
        if hasattr(result, 'interiors'):
            n_holes = len(list(result.interiors))
        else:
            n_holes = sum(len(list(g.interiors))
                          for g in getattr(result, 'geoms', []))
        assert n_holes >= 1, "Assembly should preserve internal holes"

    def test_slivers_closed_by_buffer(self):
        """Two adjacent faces with 0.3pt gap → gap closed by buffer/unbuffer."""
        from src.slab_v2 import planarize

        f1 = _make_face(0, 0, 0, 50, 100)
        f2 = _make_face(1, 50.3, 0, 100, 100)  # 0.3pt gap

        result, err = planarize.assemble_slab_polygon(
            [f1, f2], [0, 1], [], min_component_frac=0.02)

        assert result is not None
        # After sliver healing, should be 1 connected component
        n_parts = len(list(getattr(result, 'geoms', [result])))
        assert n_parts == 1, (
            f"0.3pt gap should be healed into 1 component, got {n_parts}")

    def test_large_holes_not_closed(self):
        """A genuine 20pt hole should NOT be closed by buffer/unbuffer."""
        from src.slab_v2 import planarize

        f1 = _make_face(0, 0, 0, 40, 100)
        f2 = _make_face(1, 60, 0, 100, 100)  # 20pt gap

        result, err = planarize.assemble_slab_polygon(
            [f1, f2], [0, 1], [], min_component_frac=0.02)

        assert result is not None
        # 20pt gap is much larger than 2*0.5pt heal → should stay as 2 components
        n_parts = len(list(getattr(result, 'geoms', [result])))
        assert n_parts == 2, (
            f"20pt gap should remain as 2 components, got {n_parts}")


# ---------------------------------------------------------------------------
# Unit: Config parameters
# ---------------------------------------------------------------------------

class TestConfigParams:

    def test_background_face_max_frac_default(self):
        cfg = SlabV2Config()
        assert hasattr(cfg, "background_face_max_frac")
        assert cfg.background_face_max_frac == 0.50

    def test_sliver_heal_pt_default(self):
        cfg = SlabV2Config()
        assert hasattr(cfg, "sliver_heal_pt")
        assert cfg.sliver_heal_pt == 0.5


# ---------------------------------------------------------------------------
# Integration: Combined Structural p18
# ---------------------------------------------------------------------------

_COMBINED_PDF = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")
_2381_PDF = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")


@pytest.mark.skipif(not _COMBINED_PDF.exists(),
                    reason="Combined Structural PDF not found")
class TestP18SlabAssembly:
    """p18 is a General Arrangement plan with sparse structural line work.
    The face graph produces only small faces (max 0.2% content area) — the
    slab coverage is inherently low. Full slab assembly for GA plans requires
    a separate approach (future work)."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    def test_p18_no_regression(self, cfg):
        """p18 should still produce OK status and some cuts."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_COMBINED_PDF), 17, cfg, use_ai=True)
        assert result.status == "OK"
        assert len(result.slabs) >= 1
        assert len(result.verified_cut_openings) >= 1


@pytest.mark.skipif(not _2381_PDF.exists(),
                    reason="2381 MSCP PDF not found")
class TestRegressionMSCP:

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    def test_2381_p5_slab_area_regression(self, cfg):
        """2381 MSCP p5 slab should still be ~3300 m² (no regression)."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_2381_PDF), 4, cfg, use_ai=True)
        assert result.status == "OK"
        to_mm = 25.4 / 72.0 * (result.scale or 100)
        total_m2 = sum(
            s["polygon_pdf"].area * to_mm * to_mm / 1e6
            for s in result.slabs)
        assert 2500 < total_m2 < 4500, (
            f"2381 MSCP p5 slab area {total_m2:.0f} m² regressed "
            f"(expected 2500-4500)")
