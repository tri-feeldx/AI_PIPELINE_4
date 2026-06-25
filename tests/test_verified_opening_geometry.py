import json
from pathlib import Path

import fitz
from shapely.geometry import box

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ElementFootprint, VectorPath, WallFootprint
from src.slab_v2.opening_resolver import (
    _apply_multi_intent_policy,
    _raw_candidates,
    _snap_penetration_to_slab_edge,
    _verified_core_wall_opening_candidates,
)


FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
                      "verified_openings_structural.json").read_text())


def _iou(left, right):
    return left.intersection(right).area / left.union(right).area


def _wall(label, bounds):
    return WallFootprint(label=label, polygon=box(*bounds))


def _element(bounds):
    polygon = box(*bounds)
    return ElementFootprint(
        type="VOID", polygon=polygon, label="VOID",
        anchor_bbox=polygon.bounds, area_pt2=polygon.area)


def test_core_envelope_is_context_and_only_interior_faces_are_openings():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    walls = [
        _wall("LW6", (100, 100, 300, 108)),
        _wall("LW1", (100, 292, 300, 300)),
        _wall("LW2", (100, 108, 108, 292)),
        _wall("LW7", (292, 108, 300, 292)),
        _wall("LW4", (148, 108, 156, 292)),
        _wall("LW5", (220, 108, 228, 292)),
        _wall("LW3", (156, 220, 220, 228)),
    ]
    raw = [
        _element((108, 108, 148, 292)),
        _element((156, 108, 220, 220)),
        _element((228, 108, 292, 292)),
    ]
    candidates, defaults, _warnings = _verified_core_wall_opening_candidates(
        walls, raw, page, page.rect, box(0, 0, 500, 500), 100,
        SlabV2Config(debug_images=False))
    context = candidates[0]
    assert context["id"] == "core_lw_wall_enclosed"
    assert context["destructive_allowed"] is False
    assert context["default_action"] == "exclude"
    assert len(defaults) == 3
    verified = [candidate for candidate in candidates
                if candidate["id"] in defaults]
    assert all(candidate["wall_intersection_ratio"] <= 0.01
               for candidate in verified)
    doc.close()


def test_single_spanning_diagonal_recovers_only_its_closed_core_face():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    walls = [
        _wall("LW6", (100, 100, 300, 108)),
        _wall("LW1", (100, 292, 300, 300)),
        _wall("LW2", (100, 108, 108, 292)),
        _wall("LW7", (292, 108, 300, 292)),
        _wall("LW4", (148, 108, 156, 292)),
        _wall("LW5", (220, 108, 228, 292)),
        _wall("LW3", (156, 220, 220, 228)),
    ]
    diagonal = VectorPath(
        id=1, style_id=0, segments=[((156, 108), (220, 220))],
        is_closed=False, is_filled=False, seqno=1)

    candidates, defaults, _warnings = _verified_core_wall_opening_candidates(
        walls, [], page, page.rect, box(0, 0, 500, 500), 100,
        SlabV2Config(debug_images=False), paths=[diagonal])

    verified = [candidate for candidate in candidates
                if candidate["id"] in defaults]
    assert len(verified) == 1
    candidate = verified[0]
    assert candidate["source_element_type"] == "SINGLE_DIAGONAL_SHAFT_SEED"
    assert candidate["polygon"].bounds == (156.0, 108.0, 220.0, 220.0)
    assert candidate["boundary_coverage"] >= 0.70
    assert candidate["geometry_audit"]["spanning_diagonals"]
    doc.close()


def test_boundary_attached_penetration_snaps_only_one_verified_side():
    cfg = SlabV2Config(debug_images=False)
    opening = box(15, 100, 200, 300)
    slab = box(0, 0, 500, 500)
    snapped, audit = _snap_penetration_to_slab_edge(
        opening, slab, 100, cfg, [1.0, 1.0, 1.0, 1.0])
    assert audit["status"] == "verified_snap"
    assert audit["side"] == "left"
    assert snapped.bounds == (0.0, 100.0, 200.0, 300.0)


def test_boundary_snap_is_blocked_by_protected_geometry():
    cfg = SlabV2Config(debug_images=False)
    opening = box(15, 100, 200, 300)
    slab = box(0, 0, 500, 500)
    protected = box(5, 150, 12, 250)
    snapped, audit = _snap_penetration_to_slab_edge(
        opening, slab, 100, cfg, [1.0, 1.0, 1.0, 1.0],
        protected_solids=protected)
    assert audit["status"] == "not_snapped"
    assert snapped.equals(opening)
    assert audit["prevented_candidates"]


def test_p8_golden_core_faces_are_independent_and_envelope_is_never_cut():
    fixture = FIXTURE["p8"]
    doc = fitz.open()
    page = doc.new_page(width=1600, height=1400)
    walls = [_wall(label, bounds)
             for label, bounds in fixture["wall_bounds"].items()]
    raw = [_element(bounds) for bounds in fixture["shaft_faces"]]
    candidates, defaults, _warnings = _verified_core_wall_opening_candidates(
        walls, raw, page, page.rect, box(0, 0, 1600, 1400), 100,
        SlabV2Config(debug_images=False))
    context = next(candidate for candidate in candidates
                   if candidate["id"] == "core_lw_wall_enclosed")
    assert context["destructive_allowed"] is False
    assert all(abs(actual - expected) < 0.01 for actual, expected in zip(
        context["polygon"].bounds, fixture["core_context_bbox"]))
    verified = [candidate for candidate in candidates
                if candidate["id"] in defaults]
    assert len(verified) == 3
    for candidate, expected_bounds in zip(verified, fixture["shaft_faces"]):
        assert _iou(candidate["polygon"], box(*expected_bounds)) >= 0.98
        assert candidate["wall_intersection_ratio"] <= 0.01
    doc.close()


def test_p10_golden_penetration_snaps_only_to_left_slab_edge():
    fixture = FIXTURE["p10"]
    cfg = SlabV2Config(debug_images=False)
    opening = box(*fixture["before_opening_bbox"])
    slab = box(528.75, 650, 1450, 1320)
    snapped, audit = _snap_penetration_to_slab_edge(
        opening, slab, 100, cfg, [1.0, 1.0, 1.0, 1.0])
    expected = box(*fixture["expected_opening_bbox"])
    assert audit["status"] == "verified_snap"
    assert audit["side"] == fixture["expected_snap_side"]
    assert abs(audit["gap_mm"] - fixture["expected_gap_mm"]) < 0.1
    assert _iou(snapped, expected) >= 0.98
    assert snapped.bounds[1:] == opening.bounds[1:]


def test_isolated_xcross_is_verified_from_slab_penetration_legend():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text((350, 50), "SLAB PENETRATION")
    element = _element((200, 200, 225, 215))
    candidates, defaults, _warnings = _raw_candidates(
        [element], [], page, page.rect, box(50, 50, 450, 450), 100,
        columns=[], cfg=SlabV2Config(debug_images=False))
    assert defaults == ["raw_01_slab_penetration"]
    candidate = candidates[0]
    assert candidate["kind_hint"] == "SLAB_PENETRATION"
    assert candidate["destructive_allowed"] is True
    assert candidate["geometry_audit"]["slab_containment_ratio"] == 1.0
    assert candidate["geometry_audit"]["structural_intersection_ratio"] == 0.0
    doc.close()


def test_isolated_xcross_without_legend_stays_review_and_does_not_cut():
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    element = _element((200, 200, 225, 215))
    candidates, defaults, _warnings = _raw_candidates(
        [element], [], page, page.rect, box(50, 50, 450, 450), 100,
        columns=[], cfg=SlabV2Config(debug_images=False))
    assert defaults == []
    assert candidates[0]["default_action"] == "review"
    assert candidates[0]["destructive_allowed"] is False
    doc.close()


def test_verified_stair_geometry_is_context_without_independent_intent():
    verified = {
        "id": "stair_STAIR_04_flight_union",
        "kind_hint": "STAIR_OPENING",
        "label": "STAIR 04",
        "nearby_text": ["STAIR", "04"],
        "object_roles": ["STAIR"],
        "verification_status": "verified",
        "default_action": "opening",
        "destructive_allowed": True,
    }
    report = _apply_multi_intent_policy([verified])
    assert report["verified_cut_ids"] == []
    assert report["prevented_stair_cut_ids"] == [verified["id"]]
    assert verified["opening_intent"] == "NONE"
    assert verified["cut_eligible"] is False
    assert verified["verification_status"] == "context_only"


def test_stair_context_does_not_veto_independent_penetration_intent():
    mixed = {
        "id": "raw_01_slab_penetration",
        "kind_hint": "SLAB_PENETRATION",
        "label": "SLAB PENETRATION",
        "nearby_text": ["STAIR", "01", "SLAB", "PENETRATION"],
        "object_roles": ["STAIR"],
        "verification_status": "verified",
        "default_action": "opening",
        "destructive_allowed": True,
    }
    report = _apply_multi_intent_policy([mixed])
    assert report["verified_cut_ids"] == [mixed["id"]]
    assert report["mixed_stair_penetration_ids"] == [mixed["id"]]
    assert mixed["opening_intent"] == "SLAB_PENETRATION"
    assert mixed["cut_eligible"] is True


def test_verified_closed_stairwell_boundary_becomes_mixed_penetration_cut():
    mixed = {
        "id": "stair_STAIR_01_closed_stairwell",
        "kind_hint": "STAIRWELL",
        "label": "STAIR 01",
        "nearby_text": ["STAIR", "01"],
        "object_roles": ["STAIR"],
        "verification_status": "verified",
        "default_action": "opening",
        "destructive_allowed": True,
        "boundary_coverage": 0.77,
        "contained_seed_ids": ["stair_STAIR_01_xcross_penetration_01"],
        "source": "x_seed+flight_seed+orthogonal_vector_enclosure",
        "geometry_audit": {"boundary_snap": {"status": "verified_snap"}},
    }
    report = _apply_multi_intent_policy([mixed])
    assert report["verified_cut_ids"] == [mixed["id"]]
    assert report["mixed_stair_penetration_ids"] == [mixed["id"]]
    assert report["penetration_boundary_restored_ids"] == [mixed["id"]]
    assert mixed["opening_intent"] == "SLAB_PENETRATION"
    assert mixed["cut_eligible"] is True
    assert "verified_boundary_snap" in mixed["opening_evidence_ids"]


def test_x_hull_stair_candidate_without_boundary_remains_context_only():
    hull = {
        "id": "stair_STAIR_01_xcross_penetration_01",
        "kind_hint": "STAIR_PENETRATION",
        "label": "STAIR 01",
        "nearby_text": ["STAIR", "01"],
        "object_roles": ["STAIR"],
        "verification_status": "verified",
        "default_action": "opening",
        "destructive_allowed": True,
        "source": "x_cross_convex_hull",
    }
    report = _apply_multi_intent_policy([hull])
    assert report["verified_cut_ids"] == []
    assert report["prevented_stair_cut_ids"] == [hull["id"]]
    assert report["x_hull_rejected_ids"] == [hull["id"]]
    assert hull["opening_intent"] == "NONE"
    assert hull["cut_eligible"] is False


def test_p8_verified_stair_opening_golden_footprints_are_distinct():
    expected = FIXTURE["p8"]["stair_openings"]
    assert set(expected) == {"STAIR 04", "STAIR 05"}
    left = box(*expected["STAIR 04"])
    right = box(*expected["STAIR 05"])
    assert left.is_valid and right.is_valid
    assert left.disjoint(right)
    assert left.area > 0 and right.area > 0
