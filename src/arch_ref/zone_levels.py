"""Split-deck zone elevations from RL spot labels on ARCH floor plans.

Split-deck car parks (like 2381 MSCP) have two deck halves per level sheet,
offset (typically 1.5m) and connected by ramps.  The sheet carries RL spot
labels ("RL 12.38") as real vector text; clustering the labels by value
yields the deck zones, and an element's elevation is taken from the nearest
label — evidence for storey heights per position, never geometry.

Fail-closed rules live with the caller: a zone is only trustworthy when it
is confirmed by >= 2 labels and its low value matches the section level
table (see tests).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz

_RL_RE = re.compile(r"^RL\s*(\d{1,3}\.\d{2,3})$")
# labels whose values differ by less than this are the same zone (m)
_ZONE_TOL_M = 0.01


@dataclass
class ZoneLevels:
    """RL value (m) -> positions (pt) of the labels confirming it."""
    zones: dict[float, list[tuple[float, float]]] = field(default_factory=dict)

    def elevation_at(self, x: float, y: float) -> float | None:
        """Elevation of the zone whose label is nearest to (x, y)."""
        best_rl, best_d = None, None
        for rl, pts in self.zones.items():
            for px, py in pts:
                d = (x - px) ** 2 + (y - py) ** 2
                if best_d is None or d < best_d:
                    best_rl, best_d = rl, d
        return best_rl


def extract_zone_levels(page: fitz.Page) -> ZoneLevels:
    found: list[tuple[float, tuple[float, float]]] = []
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            for sp in ln.get("spans", []):
                m = _RL_RE.match(sp["text"].strip())
                if not m:
                    continue
                x0, y0, x1, y1 = sp["bbox"]
                found.append((float(m.group(1)),
                              ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))

    zones = ZoneLevels()
    for val, pos in found:
        for known in zones.zones:
            if abs(known - val) <= _ZONE_TOL_M:
                zones.zones[known].append(pos)
                break
        else:
            zones.zones[val] = [pos]
    return zones
