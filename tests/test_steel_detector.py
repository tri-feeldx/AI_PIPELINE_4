from src.slab_v2.models import ColumnType
from src.slab_v2.steel_detector import (
    _canonical_member_type,
    _normalize,
    _steel_types,
)
from src.slab_v2.steel_source_planner import (
    _classify_source_page,
    _is_plausible_steel_symbol,
    _member_type_from_role,
    _steel_mark_hits,
    _steel_role,
)
from src.slab_v2.steel_position_resolver import (
    _build_level_census,
    _member_level_hints,
    _resolve_level_assignment,
)


def test_normalize_compacts_steel_symbol_text():
    assert _normalize("UC 150*") == "UC150"
    assert _normalize("sh-08d") == "SH08D"


def test_steel_types_come_from_material_or_known_prefix():
    column_types = {
        "C1": ColumnType(symbol="C1", width_mm=600, depth_mm=600, material="RC"),
        "SH08d": ColumnType(symbol="SH08d", material="UNKNOWN"),
        "ProjectMark": ColumnType(symbol="ProjectMark", material="STEEL"),
    }

    steel = _steel_types(column_types)

    assert set(steel) == {"SH08d", "ProjectMark"}


def test_steel_source_page_classification_is_role_aware():
    assert _classify_source_page("LEVEL 05 STEEL MARKING PLAN") == "marking_plan"
    assert _classify_source_page("STEEL ELEVATION 2 SCALE 1:100") == "elevation"
    assert (
        _classify_source_page("LEVEL 02 SEISMIC DIAPHRAGM REINFORCEMENT PLAN")
        == "diaphragm_reinforcement"
    )


def test_level_outline_plan_with_steel_marks_is_position_source():
    text = (
        "LEVEL 03 OUTLINE PLAN - 200 POST TENSIONED SLAB U.N.O "
        "CH35a SH08d UB36b REFER DRAWING FOR STEELWORK PLAN "
        "STEEL COLUMN SCHEDULE"
    )

    assert _classify_source_page(text) == "floor_plan_with_steel_marks"


def test_position_level_hint_wins_over_broad_profile_range():
    meta = {"levels": ["LEVEL 01", "LEVEL 02", "LEVEL 03", "ROOF"]}

    assert _member_level_hints(meta, ["LEVEL 03"]) == ["LEVEL 03"]


def test_level_assignment_uses_plan_level_not_detail_override():
    final_level, status, reason = _resolve_level_assignment(
        ["LEVEL 03"], ["LEVEL 01", "LEVEL 02", "LEVEL 03", "ROOF"])

    assert final_level == "LEVEL 03"
    assert status == "verified"
    assert "verified" in reason


def test_level_assignment_conflict_prevents_export():
    final_level, status, reason = _resolve_level_assignment(
        ["LEVEL 03"], ["LEVEL 05", "ROOF"])

    assert final_level == ""
    assert status == "review"
    assert "conflicts" in reason


def test_steel_level_census_counts_exported_and_review():
    census, counts, expected = _build_level_census([
        {
            "id": "a",
            "symbol": "SH08D",
            "member_type": "COLUMN",
            "role": "steel_column",
            "position_level": "LEVEL 03",
            "final_level": "LEVEL 03",
            "source_page": 8,
            "status": "verified",
            "exported": True,
        },
        {
            "id": "b",
            "symbol": "UB36B",
            "member_type": "BEAM",
            "role": "steel_beam",
            "position_level": "LEVEL 03",
            "final_level": "",
            "source_page": 8,
            "status": "review",
            "exported": False,
            "level_assignment_reason": "position level LEVEL 03 conflicts with profile/detail range ROOF",
        },
    ], {"status": "verified_steel"})

    assert expected["LEVEL 03"]["exported"] == 1
    assert expected["LEVEL 03"]["review"] == 1
    assert any(row["symbol"] == "SH08D" for row in counts)
    assert census["prevented_wrong_level_exports"][0]["symbol"] == "UB36B"


def test_diaphragm_role_is_not_beam_or_column_export():
    role = _steel_role(
        "SH10e",
        "LEVEL 02 SEISMIC DIAPHRAGM REINFORCEMENT PLAN",
        "diaphragm_reinforcement",
    )

    assert role == "DIAPHRAGM_REINFORCEMENT"
    assert _member_type_from_role(role) == "FLOOR"
    assert _canonical_member_type("DIAPHRAGM_REINFORCEMENT") == "FLOOR"


def test_detail_roles_map_to_detector_member_types():
    assert _canonical_member_type("PURLIN_GIRT") == "BEAM"
    assert _canonical_member_type("STEEL_BRACING") == "BRACING"
    assert _canonical_member_type("REFERENCE_ONLY") == "REVIEW_ONLY"


def test_project_specific_steel_marks_are_supported():
    assert _steel_role("PF25a", "provide purlin locations", "elevation") == "PURLIN_GIRT"
    assert _canonical_member_type("PURLIN_GIRT") == "BEAM"


def test_project_specific_symbol_families_need_steel_context():
    text = (
        "LEVEL 03 OUTLINE PLAN STEELWORK PLAN "
        "BT075 CT050 D013 LA46 TF100 EA100 UA75"
    )

    hits = set(_steel_mark_hits(text))

    assert {"BT075", "CT050", "D013", "LA46", "TF100", "EA100", "UA75"} <= hits
    assert _classify_source_page(text) == "floor_plan_with_steel_marks"
    assert _is_plausible_steel_symbol(
        "D013", source="pdf_text_scan", context=text)
    assert not _is_plausible_steel_symbol(
        "D013", source="pdf_text_scan", context="drawing date 2026 general notes")


def test_project_specific_roles_are_not_reference_only_in_steel_context():
    assert _steel_role("BT075", "steel framing beam", "floor_plan_with_steel_marks") == "STEEL_BEAM"
    assert _steel_role("CT050", "steel column frame", "floor_plan_with_steel_marks") == "STEEL_COLUMN"
    assert _member_type_from_role(
        _steel_role("LA46", "purlin/girt facade steel", "elevation")
    ) == "BEAM"
