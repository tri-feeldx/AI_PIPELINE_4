import json
from pathlib import Path

from src.slab_v2 import trace


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_quality_gate_blocks_tiny_verified_slab(tmp_path):
    run_root = tmp_path / "upload"
    _write_json(run_root / "run_trace.json", {
        "pdf": {"name": "fixture.pdf"},
        "config": {},
    })

    for page, area, frac in ((10, 3000.0, 0.7), (11, 3200.0, 0.72)):
        page_dir = run_root / f"page_{page}"
        _write_json(page_dir / "page_trace.json", {
            "page_number": page,
            "status": "OK",
            "scale": 100,
            "counts": {},
            "stage_snapshots": [{
                "stage": "assembly",
                "slab_fraction_of_content": frac,
            }],
        })
        _write_json(page_dir / "result.json", {
            "slabs": [{"area_m2": area}],
            "floor_system_readiness": {"status": "verified"},
            "slab_readiness": {"status": "verified"},
        })

    page_dir = run_root / "page_23"
    _write_json(page_dir / "page_trace.json", {
        "page_number": 23,
        "status": "OK",
        "scale": 15,
        "warnings": ["assembled slab covers only 1% after 3 attempts - using best result anyway"],
        "counts": {},
        "stage_snapshots": [{
            "stage": "assembly",
            "slab_fraction_of_content": 0.008,
        }],
    })
    _write_json(page_dir / "result.json", {
        "slabs": [{"area_m2": 2.0}],
        "floor_system_readiness": {"status": "verified"},
        "slab_readiness": {"status": "verified"},
    })

    reports = trace.write_run_audit_reports(run_root)
    quality = reports["quality_gate"]
    codes = {issue["code"] for issue in quality["issues"]}

    assert quality["status"] == "blocked"
    assert "slab_area_outlier" in codes
    assert "tiny_slab_fraction" in codes
    assert "verified_status_contradicts_trace" in codes
    assert reports["delivery_readiness"]["ready_for_client_or_boss"] is False
