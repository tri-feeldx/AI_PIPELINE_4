"""Aggregate slab, opening, and height evidence into export readiness."""

from __future__ import annotations

from src.slab_v2.models import ModelReadinessReport


def build_model_readiness(storeys: list[dict], level_datums: list,
                          building: str) -> ModelReadinessReport:
    reasons = []
    slab_states = [s["result"].slab_readiness.get("status", "review")
                   for s in storeys]
    slab_status = "verified" if slab_states and all(
        x == "verified" for x in slab_states) else "review"
    if slab_status != "verified":
        reasons.append("One or more slab faces require review.")

    opening_review = False
    for s in storeys:
        result = s["result"]
        opening_report = result.opening_report or {}
        unresolved = (opening_report.get("unresolved_candidate_ids", [])
                      or opening_report.get("high_impact_review_ids", []))
        if (unresolved or
                (result.opening_candidates and result.opening_judgement.get(
                    "status") != "accepted")):
            opening_review = True
            break
    opening_status = "review" if opening_review else "verified"
    if opening_review:
        reasons.append(
            "One or more high-impact opening candidates lack verified geometry.")

    wall_rows = [s["result"].wall_readiness for s in storeys
                 if s["result"].wall_readiness]
    wall_bad = [row for row in wall_rows
                if row.get("status", "review") not in {"verified", "not_required"}
                or row.get("missing")]
    wall_status = "verified" if not wall_bad else "review"
    if wall_status != "verified":
        reasons.append("One or more expected walls lack verified topology/profile geometry.")

    junction_bad = [row for row in wall_rows
                    if row.get("junction_status", "not_required")
                    not in {"verified", "not_required"}]
    wall_junction_status = "verified" if not junction_bad else "review"
    if wall_junction_status != "verified":
        reasons.append("One or more wall junctions are not geometrically verified.")

    column_reports = [s["result"].column_detection_report for s in storeys]
    column_bad = [report for report in column_reports
                  if report.get("missing") or report.get("extra")
                  or report.get("ambiguous_count", 0)
                  or report.get("status", "review")
                  not in {"verified", "not_required"}]
    column_status = "verified" if not column_bad else "review"
    if column_status != "verified":
        reasons.append("Expected RC columns are missing, extra, or ambiguous.")

    steel_reports = [s["result"].steel_readiness for s in storeys
                     if getattr(s["result"], "steel_readiness", None)]
    steel_bad = [report for report in steel_reports
                 if report.get("status", "not_required")
                 not in {"verified", "verified_steel", "not_required"}]
    steel_status = "verified" if steel_reports and not steel_bad else "not_required"
    if steel_bad:
        steel_status = "review"
        reasons.append(
            "Steel members require review or lack verified geometry.")

    contract_rows = [
        getattr(s["result"], "contract_reconciliation", {}) or {}
        for s in storeys
        if getattr(s["result"], "contract_reconciliation", None)
    ]
    contract_bad = False
    for report in contract_rows:
        if int(report.get("critical_unfulfilled_count", 0) or 0) > 0:
            contract_bad = True
            break
        if report.get("contract_status") not in {None, "", "fulfilled"}:
            contract_bad = True
            break
    if contract_bad:
        reasons.append(
            "Drawing contract has missing, extra, blocked, or partial expected items.")

    rendered_shafts = [element for s in storeys
                       for element in s["result"].render_elements
                       if element.type in {"SHAFT", "LIFT", "CORE"}]
    shaft_render_status = "verified" if not rendered_shafts else "review"
    if rendered_shafts:
        reasons.append("Duplicate shaft/core solids remain enabled.")

    rendered_stairs = [element for s in storeys
                       for element in s["result"].render_elements
                       if element.type == "STAIR"]
    stair_render_status = "verified" if not rendered_stairs else "review"
    if rendered_stairs:
        reasons.append("Stair solids remain enabled; customer output is opening-only.")

    datums = [d for d in level_datums if d.building == building]
    states = {d.status for d in datums}
    if not datums or "default_unsafe" in states:
        height_status = "default_unsafe"
        reasons.append("At least one level uses an unsafe default height.")
    elif "conflict" in states:
        height_status = "conflict"
        reasons.append("Strong height evidence is conflicting.")
    elif states - {"manual", "verified_explicit", "verified_consensus"}:
        height_status = "inferred"
        reasons.append("At least one level datum is inferred, not verified.")
    else:
        height_status = "verified"

    final = (slab_status == "verified" and height_status == "verified"
             and opening_status == "verified" and wall_status == "verified"
             and wall_junction_status == "verified"
             and column_status == "verified"
             and steel_status in {"verified", "not_required"}
             and shaft_render_status == "verified"
             and stair_render_status == "verified"
             and not contract_bad)
    return ModelReadinessReport(
        slab_status=slab_status, height_status=height_status,
        opening_status=opening_status, wall_status=wall_status,
        column_status=column_status,
        steel_status=steel_status,
        wall_junction_status=wall_junction_status,
        shaft_render_status=shaft_render_status,
        stair_render_status=stair_render_status,
        model_status="final" if final else "debug", reasons=reasons)
