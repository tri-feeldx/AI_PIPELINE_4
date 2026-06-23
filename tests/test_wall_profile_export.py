import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import fitz
from shapely.geometry import box

from src.slab_v2.export_ruby import _profile_wall_lines, _walls_mm
from src.slab_v2.models import SlabV2Result, WallFootprint, WallType
from src.slab_v2.wall_profile_resolver import (
    _visible_profile_polygon,
    resolve_plan_wall_topology,
)


class WallProfileExportTests(unittest.TestCase):
    def setUp(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=1000, height=700)

    def tearDown(self):
        self.doc.close()

    def test_symbol_keyed_profile_reaches_ruby_export(self):
        wall = WallFootprint(
            label="W1",
            polygon=box(100, 100, 500, 110),
            w_mm=250,
            l_mm=14000,
            centerline=[(100, 105), (500, 105)],
            profile_id="profile_W1",
            mapping_status="verified",
        )
        result = SlabV2Result(page_index=0, walls=[wall], scale=100)
        result.wall_profiles = {
            "W1": {
                "profile_id": "profile_W1",
                "status": "verified",
                "panels": [{
                    "polygon_station_z": [
                        [0.0, 0.0], [1.0, 0.0],
                        [1.0, 3200.0], [0.0, 2000.0],
                    ]
                }],
            }
        }

        item = _walls_mm(result, self.page, 100)[0]
        self.assertEqual(item["profile"]["profile_id"], "profile_W1")

        lines, warnings = _profile_wall_lines(item, "wall_grp", 0.0, 3000.0)
        ruby = "\n".join(lines)
        self.assertIn("W1 elevation panel 1", ruby)
        self.assertNotIn("pushpull", ruby)
        self.assertEqual(warnings, [])

    def test_visible_profile_keeps_sloped_vector_top(self):
        shape = self.page.new_shape()
        shape.draw_rect(fitz.Rect(100, 100, 500, 300))
        shape.finish(fill=(0.9, 0.9, 0.9), color=None)
        shape.commit()
        self.page.draw_line((100, 260), (250, 200),
                            color=(0, 0, 0), width=1)
        self.page.draw_line((250, 200), (500, 120),
                            color=(0, 0, 0), width=1)

        polygon = _visible_profile_polygon(
            self.page, (100.0, 100.0, 500.0, 300.0))
        top = polygon[2:]
        self.assertGreater(len(polygon), 4)
        self.assertGreater(max(y for _x, y in top) - min(y for _x, y in top),
                           100)

    def test_keyplan_topology_recovers_complete_perimeter_runs(self):
        self.page.insert_text((450, 105), "W1")
        self.page.insert_text((890, 300), "W2")
        self.page.insert_text((450, 495), "W3")
        slab = box(100, 100, 900, 500)
        walls = [
            WallFootprint("W1", box(400, 100, 500, 107), 250, 3500),
            WallFootprint("W2", box(893, 250, 900, 350), 250, 3500),
            WallFootprint("W3", box(400, 493, 500, 500), 250, 3500),
        ]
        types = {symbol: WallType(symbol=symbol, thickness_mm=250)
                 for symbol in ("W1", "W2", "W3")}
        registry = {
            "keyplan": {"symbols": {
                "W1": {"orientation": "horizontal"},
                "W2": {"orientation": "vertical"},
                "W3": {"orientation": "horizontal"},
            }, "grid_anchors": {
                "1": [0, 0], "2": [100, 0],
                "A": [0, 100], "B": [0, 200]}},
            "profiles": {symbol: {
                "profile_id": f"profile_{symbol}", "status": "verified",
                "grid_start": ("A" if symbol == "W2" else "1"),
                "grid_end": ("B" if symbol == "W2" else "2")}
                for symbol in ("W1", "W2", "W3")},
        }
        with tempfile.TemporaryDirectory() as temp:
            with patch("src.slab_v2.wall_profile_resolver._grid_anchors",
                       return_value={
                           "1": [100, 100], "2": [900, 100],
                           "A": [100, 100], "B": [100, 500]}):
                resolved, report = resolve_plan_wall_topology(
                    self.page, slab, walls, types,
                    {"W1": 1, "W2": 1, "W3": 1}, registry, 100,
                    Path(temp))
            by_label = {wall.label: wall for wall in resolved}
            self.assertEqual(report["status"], "verified")
            self.assertGreater(by_label["W1"].l_mm, 28000)
            self.assertGreater(by_label["W2"].l_mm, 14000)
            self.assertGreater(by_label["W3"].l_mm, 28000)
            self.assertTrue((Path(temp) /
                             "wall_3d_mapping_report_p01.json").exists())


if __name__ == "__main__":
    unittest.main()
