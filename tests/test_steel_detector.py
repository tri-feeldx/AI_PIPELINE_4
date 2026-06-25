from src.slab_v2.models import ColumnType
from src.slab_v2.steel_detector import _normalize, _steel_types


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
