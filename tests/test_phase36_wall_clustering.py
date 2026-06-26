"""TDD tests for Phase 3.6 Fix B: Cluster core walls by spatial proximity.

The core wall detector previously lumped ALL CW/LW walls into one convex hull.
When CW1a-d and CW2a-d are separate cores, the hull spans the building and
elements inside CW2 fail the 98% containment check.

Fix: cluster walls by spatial proximity, process each cluster independently.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config


def _make_wall(label, x0, y0, x1, y1):
    w = MagicMock()
    w.polygon = box(x0, y0, x1, y1)
    w.label = label
    return w


# ---------------------------------------------------------------------------
# Unit: _cluster_walls function
# ---------------------------------------------------------------------------

class TestClusterWalls:
    """Test spatial clustering of core walls."""

    def test_two_separate_cores_form_two_clusters(self):
        from src.slab_v2.opening_resolver import _cluster_walls

        scale = 100
        to_mm = 25.4 / 72 * scale
        gap_pt = 200.0 / to_mm  # 200mm gap tolerance

        # CW1 walls at x=100-200, CW2 walls at x=600-700 (far apart)
        walls = [
            _make_wall("CW1a", 100, 100, 200, 105),
            _make_wall("CW1b", 100, 195, 200, 200),
            _make_wall("CW1c", 100, 100, 105, 200),
            _make_wall("CW1d", 195, 100, 200, 200),
            _make_wall("CW2a", 600, 100, 700, 105),
            _make_wall("CW2b", 600, 195, 700, 200),
            _make_wall("CW2c", 600, 100, 605, 200),
            _make_wall("CW2d", 695, 100, 700, 200),
        ]

        clusters = _cluster_walls(walls, gap_pt)
        assert len(clusters) == 2, (
            f"Expected 2 clusters for separate CW1/CW2, got {len(clusters)}")
        for cluster in clusters:
            assert len(cluster) == 4

    def test_adjacent_walls_form_one_cluster(self):
        from src.slab_v2.opening_resolver import _cluster_walls

        scale = 100
        to_mm = 25.4 / 72 * scale
        gap_pt = 200.0 / to_mm

        # All walls touching or very close
        walls = [
            _make_wall("LW1", 100, 100, 200, 105),
            _make_wall("LW2", 100, 195, 200, 200),
            _make_wall("LW3", 100, 100, 105, 200),
            _make_wall("LW4", 195, 100, 200, 200),
        ]

        clusters = _cluster_walls(walls, gap_pt)
        assert len(clusters) == 1, (
            f"Expected 1 cluster for adjacent walls, got {len(clusters)}")
        assert len(clusters[0]) == 4

    def test_single_wall_forms_own_cluster(self):
        from src.slab_v2.opening_resolver import _cluster_walls

        walls = [_make_wall("CW1a", 100, 100, 200, 105)]
        clusters = _cluster_walls(walls, 10.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_empty_input(self):
        from src.slab_v2.opening_resolver import _cluster_walls

        clusters = _cluster_walls([], 10.0)
        assert len(clusters) == 0


# ---------------------------------------------------------------------------
# Unit: core wall detector with multiple clusters
# ---------------------------------------------------------------------------

class TestCoreWallMultiCluster:
    """Core wall detector should process each cluster independently."""

    def test_element_inside_cw2_detected_with_separate_hull(self):
        """Element inside CW2 should be detected when CW2 has its own hull."""
        from src.slab_v2.opening_resolver import _verified_core_wall_opening_candidates
        import fitz

        scale = 100
        to_mm = 25.4 / 72 * scale
        shaft_pt = 3000 / to_mm  # 3m shaft
        wall_thick_pt = 150 / to_mm

        # CW1 at x=200, CW2 at x=700 (separate cores)
        walls = []
        for prefix, cx in [("CW1", 200), ("CW2", 700)]:
            cy = 350
            half = shaft_pt / 2
            walls.extend([
                _make_wall(f"{prefix}a", cx - half, cy - half,
                          cx + half, cy - half + wall_thick_pt),
                _make_wall(f"{prefix}b", cx - half, cy + half - wall_thick_pt,
                          cx + half, cy + half),
                _make_wall(f"{prefix}c", cx - half, cy - half,
                          cx - half + wall_thick_pt, cy + half),
                _make_wall(f"{prefix}d", cx + half - wall_thick_pt, cy - half,
                          cx + half, cy + half),
            ])

        # Element inside CW2 (should be detected)
        cx2, cy2 = 700, 350
        half = shaft_pt / 2
        elem = MagicMock()
        elem.polygon = box(cx2 - half + wall_thick_pt + 1,
                          cy2 - half + wall_thick_pt + 1,
                          cx2 + half - wall_thick_pt - 1,
                          cy2 + half - wall_thick_pt - 1)
        elem.type = "VOID"
        elem.label = "VOID"
        elem.area_pt2 = elem.polygon.area
        elem.anchor_bbox = elem.polygon.bounds

        page = MagicMock()
        page.get_text_blocks = MagicMock(return_value=[])

        content_rect = fitz.Rect(0, 0, 1000, 700)
        slab_union = box(0, 0, 1000, 700)
        cfg = SlabV2Config()

        candidates, defaults, warnings = _verified_core_wall_opening_candidates(
            walls, [elem], page, content_rect, slab_union, scale, cfg)

        verified = [c for c in candidates
                    if c.get("verification_status") == "verified"
                    and c.get("kind_hint") != "CORE_CONTEXT"]
        assert len(verified) >= 1, (
            f"CW2 element should be verified. Got {len(verified)} verified. "
            f"Candidates: {[(c['id'], c.get('verification_status')) for c in candidates]}")

    def test_element_between_clusters_not_detected(self):
        """Element in the gap between CW1 and CW2 should NOT be detected."""
        from src.slab_v2.opening_resolver import _verified_core_wall_opening_candidates
        import fitz

        scale = 100
        to_mm = 25.4 / 72 * scale
        shaft_pt = 2000 / to_mm
        wall_thick_pt = 150 / to_mm

        walls = []
        for prefix, cx in [("CW1", 200), ("CW2", 700)]:
            cy = 350
            half = shaft_pt / 2
            walls.extend([
                _make_wall(f"{prefix}a", cx - half, cy - half,
                          cx + half, cy - half + wall_thick_pt),
                _make_wall(f"{prefix}b", cx - half, cy + half - wall_thick_pt,
                          cx + half, cy + half),
                _make_wall(f"{prefix}c", cx - half, cy - half,
                          cx - half + wall_thick_pt, cy + half),
                _make_wall(f"{prefix}d", cx + half - wall_thick_pt, cy - half,
                          cx + half, cy + half),
            ])

        # Element in the GAP between CW1 and CW2 (should NOT be detected)
        elem = MagicMock()
        elem.polygon = box(400, 300, 500, 400)
        elem.type = "VOID"
        elem.label = "VOID"
        elem.area_pt2 = elem.polygon.area
        elem.anchor_bbox = elem.polygon.bounds

        page = MagicMock()
        page.get_text_blocks = MagicMock(return_value=[])

        content_rect = fitz.Rect(0, 0, 1000, 700)
        slab_union = box(0, 0, 1000, 700)
        cfg = SlabV2Config()

        candidates, defaults, warnings = _verified_core_wall_opening_candidates(
            walls, [elem], page, content_rect, slab_union, scale, cfg)

        verified = [c for c in candidates
                    if c.get("verification_status") == "verified"
                    and c.get("kind_hint") != "CORE_CONTEXT"]
        assert len(verified) == 0, (
            f"Element in gap should NOT be verified. Got {len(verified)}")


# ---------------------------------------------------------------------------
# Integration: Combined Structural p18 CW2 detection
# ---------------------------------------------------------------------------

_COMBINED_PDF = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")


@pytest.mark.skipif(not _COMBINED_PDF.exists(),
                    reason="Combined Structural PDF not found")
class TestP18WallClustering:
    """p18 CW2 void should now be detected via clustered core walls."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    def test_p18_has_more_cuts_after_clustering(self, cfg):
        """p18 should have >=2 verified cuts (CW1 + at least CW2)."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_COMBINED_PDF), 17, cfg, use_ai=True)
        assert result.status == "OK"
        assert len(result.verified_cut_openings) >= 2
