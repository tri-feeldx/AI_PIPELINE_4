from types import SimpleNamespace

from shapely.geometry import box

from src.slab_v2.drawing_contract import (
    apply_contract_export_policy,
    build_drawing_contract,
    reconcile_drawing_contract,
)


def _result(**kwargs):
    defaults = {
        "slabs": [],
        "columns": [],
        "column_detection_report": {},
        "column_readiness": {},
        "walls": [],
        "wall_readiness": {},
        "steel_members": [],
        "steel_readiness": {},
        "verified_cut_openings": [],
        "opening_review_candidates": [],
        "page_index": 0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class _SlottedCandidate:
    __slots__ = ("symbol", "polygon", "confidence", "source")

    def __init__(self, symbol, polygon, confidence=1.0, source="local_vector"):
        self.symbol = symbol
        self.polygon = polygon
        self.confidence = confidence
        self.source = source


def test_contract_reports_missing_geometry():
    contract = {
        "contract_items": [
            {
                "subsystem": "rc_column",
                "building": "A",
                "level": "LEVEL 01",
                "symbol": "C1",
                "role": "",
                "expected_count": 2,
            }
        ]
    }
    storeys = {
        "A": [{
            "level_id": "LEVEL 01",
            "level_name": "LEVEL 01",
            "result": _result(columns=[
                SimpleNamespace(symbol="C1", source="global_text_assignment")
            ]),
        }]
    }

    report = reconcile_drawing_contract(contract, storeys)

    row = report["counts_by_level"][0]
    assert row["exported_count"] == 1
    assert row["missing_count"] == 1
    assert row["status"] == "partial"
    assert report["critical_unfulfilled_count"] == 1


def test_contract_reports_extra_export_without_contract():
    contract = {"contract_items": []}
    storeys = {
        "A": [{
            "level_id": "LEVEL 01",
            "level_name": "LEVEL 01",
            "result": _result(columns=[
                SimpleNamespace(symbol="C99", source="global_text_assignment")
            ]),
        }]
    }

    report = reconcile_drawing_contract(contract, storeys)

    assert report["missing_extra_blocked"][0]["status"] == "extra"
    assert report["missing_extra_blocked"][0]["symbol"] == "C99"


def test_contract_exports_only_best_wall_candidate_for_expected_count():
    contract = {
        "contract_items": [{
            "subsystem": "wall",
            "building": "A",
            "level": "LEVEL 01",
            "symbol": "W1",
            "role": "",
            "expected_count": 1,
        }]
    }
    walls = [
        SimpleNamespace(
            label="W1", polygon=box(0, 0, 10, 1), l_mm=1000,
            confidence=0.2, mapping_status="review"),
        SimpleNamespace(
            label="W1", polygon=box(0, 0, 100, 1), l_mm=10000,
            confidence=0.9, mapping_status="verified"),
        SimpleNamespace(
            label="W1", polygon=box(0, 0, 30, 1), l_mm=3000,
            confidence=0.5, mapping_status="review"),
    ]
    storeys = {"A": [{
        "level_id": "LEVEL 01",
        "level_name": "LEVEL 01",
        "result": _result(walls=walls),
    }]}

    decisions = apply_contract_export_policy(contract, storeys)
    report = reconcile_drawing_contract(contract, storeys)

    kept = storeys["A"][0]["result"].walls
    assert len(kept) == 1
    assert kept[0].l_mm == 10000
    assert decisions["summary"]["wall"]["extra_hidden"] == 2
    assert report["counts_by_level"][0]["exported_count"] == 1
    assert report["counts_by_level"][0]["status"] == "fulfilled"


def test_contract_blocks_dashed_candidate_and_keeps_count_unfulfilled():
    contract = {
        "contract_items": [{
            "subsystem": "steel",
            "building": "A",
            "level": "LEVEL 01",
            "symbol": "SH1",
            "role": "steel_column",
            "expected_count": 2,
        }]
    }
    steel = [
        SimpleNamespace(
            id="s1", symbol="SH1", member_type="COLUMN",
            polygon=box(0, 0, 1, 1), status="review",
            confidence=0.3, source="solid"),
        SimpleNamespace(
            id="s2", symbol="SH1", member_type="COLUMN",
            polygon=box(2, 0, 3, 1), status="review",
            confidence=0.9, source="dashed reference"),
    ]
    storeys = {"A": [{
        "level_id": "LEVEL 01",
        "level_name": "LEVEL 01",
        "result": _result(steel_members=steel),
    }]}

    apply_contract_export_policy(contract, storeys)
    report = reconcile_drawing_contract(contract, storeys)

    kept = storeys["A"][0]["result"].steel_members
    assert len(kept) == 1
    assert kept[0].id == "s1"
    row = report["counts_by_level"][0]
    assert row["exported_count"] == 1
    assert row["blocked_count"] == 1
    assert row["missing_count"] == 1
    assert row["status"] == "blocked"


def test_contract_hides_candidates_when_subsystem_has_no_contract_item():
    contract = {"contract_items": []}
    columns = [
        SimpleNamespace(
            symbol="C1", polygon=box(0, 0, 1, 1),
            confidence=1.0, source="local_vector"),
    ]
    storeys = {"A": [{
        "level_id": "LEVEL 01",
        "level_name": "LEVEL 01",
        "result": _result(columns=columns),
    }]}

    decisions = apply_contract_export_policy(contract, storeys)

    result = storeys["A"][0]["result"]
    assert result.columns == []
    assert result.candidate_registry[0]["export_decision"] == "no_contract_hidden"
    assert result.candidate_registry[0]["reject_reason"] == (
        "no_contract_item_for_subsystem_level_page")
    assert decisions["summary"] == {}


def test_contract_policy_handles_candidates_without_dynamic_attributes():
    contract = {
        "contract_items": [{
            "subsystem": "rc_column",
            "building": "A",
            "level": "LEVEL 01",
            "symbol": "C1",
            "role": "",
            "expected_count": 1,
        }]
    }
    columns = [_SlottedCandidate("C1", box(0, 0, 1, 1))]
    storeys = {"A": [{
        "level_id": "LEVEL 01",
        "level_name": "LEVEL 01",
        "result": _result(columns=columns),
    }]}

    decisions = apply_contract_export_policy(contract, storeys)
    report = reconcile_drawing_contract(contract, storeys)

    result = storeys["A"][0]["result"]
    assert len(result.columns) == 1
    assert result.candidate_registry[0]["export_decision"] == "exported"
    assert decisions["summary"]["rc_column"]["exported"] == 1
    assert report["counts_by_level"][0]["status"] == "fulfilled"


def test_contract_level_aliases_match_ground_and_underscore_levels():
    contract = {
        "contract_items": [
            {
                "subsystem": "slab",
                "building": "A",
                "level": "GROUND",
                "symbol": "",
                "role": "FLOOR_SLAB",
                "expected_count": 1,
            },
            {
                "subsystem": "rc_column",
                "building": "A",
                "level": "LEVEL_1",
                "symbol": "C1",
                "role": "",
                "expected_count": 1,
            },
        ]
    }
    storeys = {
        "A": [
            {
                "level_id": "GROUND FLOOR",
                "level_name": "GROUND FLOOR",
                "result": _result(slabs=[
                    SimpleNamespace(label="SLAB", polygon=box(0, 0, 10, 10))
                ], page_index=0),
            },
            {
                "level_id": "LEVEL 01",
                "level_name": "LEVEL 01",
                "result": _result(columns=[
                    SimpleNamespace(
                        symbol="C1", polygon=box(0, 0, 1, 1),
                        source="local_vector", confidence=1.0)
                ], page_index=1),
            },
        ]
    }

    decisions = apply_contract_export_policy(contract, storeys)
    report = reconcile_drawing_contract(contract, storeys)

    assert len(storeys["A"][0]["result"].slabs) == 1
    assert len(storeys["A"][1]["result"].columns) == 1
    assert decisions["summary"]["slab"]["exported"] == 1
    assert decisions["summary"]["rc_column"]["exported"] == 1
    statuses = {
        (row["subsystem"], row["level"]): row["status"]
        for row in report["counts_by_level"]
    }
    assert statuses[("slab", "GROUND FLOOR")] == "fulfilled"
    assert statuses[("rc_column", "LEVEL 01")] == "fulfilled"


def test_contract_uses_wall_census_counts_from_floor_info():
    doc = SimpleNamespace(
        buildings=[
            SimpleNamespace(
                name="B",
                floors=[
                    SimpleNamespace(
                        level_id="LEVEL 01",
                        level_name="LEVEL 01",
                        pages=[0],
                        titles=["GENERAL ARRANGEMENT PLAN - LEVEL 01"],
                        ffl_m=None,
                        storey_height_mm=0,
                        columns={},
                        walls={"IW30": 8, "BW1": 2},
                    )
                ],
            )
        ],
        columns_per_floor=[],
        column_types={},
    )
    storeys = {
        "B": [{
            "level_id": "LEVEL 01",
            "level_name": "LEVEL 01",
            "page_idx": 0,
            "result": _result(page_index=0),
        }]
    }

    contract = build_drawing_contract(doc, storeys)
    wall_rows = [
        r for r in contract["contract_items"]
        if r["subsystem"] == "wall"
    ]

    assert {(r["symbol"], r["expected_count"]) for r in wall_rows} == {
        ("BW1", 2),
        ("IW30", 8),
    }


def test_contract_prefers_storey_steel_level_counts_over_unknown_symbols():
    doc = SimpleNamespace(
        buildings=[
            SimpleNamespace(
                name="B",
                floors=[
                    SimpleNamespace(
                        level_id="LEVEL 01",
                        level_name="LEVEL 01",
                        pages=[0],
                        titles=["LEVEL 01 PLAN"],
                        ffl_m=None,
                        storey_height_mm=0,
                        columns={},
                        walls={},
                    )
                ],
            )
        ],
        columns_per_floor=[],
        column_types={},
    )
    result = _result(
        page_index=0,
        steel_readiness={
            "counts_by_level_and_symbol": [{
                "level": "LEVEL 01",
                "symbol": "SH1",
                "role": "steel_column",
                "expected": 3,
                "detected": 3,
                "exported": 0,
            }]
        },
    )
    storeys = {
        "B": [{
            "level_id": "LEVEL 01",
            "level_name": "LEVEL 01",
            "page_idx": 0,
            "result": result,
        }]
    }

    contract = build_drawing_contract(
        doc, storeys,
        steel_census={"expected_symbols": ["BT075", "CT050"]},
    )
    steel_rows = [
        r for r in contract["contract_items"]
        if r["subsystem"] == "steel"
    ]

    assert len(steel_rows) == 1
    assert steel_rows[0]["level"] == "LEVEL 01"
    assert steel_rows[0]["symbol"] == "SH1"
    assert steel_rows[0]["expected_count"] == 3


def test_debug_export_all_detected_steel_ignores_contract_cap():
    contract = {
        "contract_items": [{
            "subsystem": "steel",
            "building": "A",
            "level": "LEVEL 01",
            "symbol": "SH1",
            "role": "steel_column",
            "expected_count": 1,
        }]
    }
    members = [
        SimpleNamespace(
            id=f"s{i}", symbol="SH1", member_type="steel_column",
            polygon=box(i, 0, i + 1, 1), status="review",
            confidence=0.8, is_dashed=False, is_reference_only=False)
        for i in range(3)
    ]
    result = _result(
        steel_members=members,
        steel_readiness={"export_all_detected_steel": True},
    )
    storeys = {
        "A": [{
            "level_id": "LEVEL 01",
            "level_name": "LEVEL 01",
            "page_idx": 0,
            "result": result,
        }]
    }

    decisions = apply_contract_export_policy(contract, storeys)

    assert len(result.steel_members) == 3
    assert decisions["summary"]["steel"]["exported"] == 3
    assert {m.contract_export_decision for m in result.steel_members} == {"exported"}
    assert all(
        "debug_export_all_detected_steel" in m.contract_export_reason
        for m in result.steel_members
    )


def test_debug_export_all_detected_steel_still_blocks_dashed():
    contract = {"contract_items": []}
    keep = SimpleNamespace(
        id="keep", symbol="SH1", member_type="steel_column",
        polygon=box(0, 0, 1, 1), status="review",
        confidence=0.8, is_dashed=False, is_reference_only=False)
    dashed = SimpleNamespace(
        id="dash", symbol="SH1", member_type="steel_column",
        polygon=box(2, 0, 3, 1), status="review",
        confidence=0.8, is_dashed=True, is_reference_only=False)
    result = _result(
        steel_members=[keep, dashed],
        steel_readiness={"export_all_detected_steel": True},
    )
    storeys = {
        "A": [{
            "level_id": "LEVEL 01",
            "level_name": "LEVEL 01",
            "page_idx": 0,
            "result": result,
        }]
    }

    decisions = apply_contract_export_policy(contract, storeys)

    assert result.steel_members == [keep]
    assert keep.contract_export_decision == "exported"
    assert dashed.contract_export_decision == "blocked_dashed"
    assert decisions["summary"]["steel"]["exported"] == 1
    assert decisions["summary"]["steel"]["blocked"] == 1
