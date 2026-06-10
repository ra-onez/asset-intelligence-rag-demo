import pandas as pd


CRITICALITY_POINTS = {
    "High": 25,
    "Medium": 15,
    "Low": 5,
}


def _risk_level(score):
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def calculate_risk_scores(assets, work_orders, inspections, failures):
    rows = []

    for _, asset in assets.iterrows():
        asset_id = asset["asset_id"]
        score = 0
        reasons = []

        criticality = asset["criticality"]
        criticality_points = CRITICALITY_POINTS.get(criticality, 5)
        score += criticality_points
        reasons.append(f"{criticality} criticality (+{criticality_points})")

        asset_wos = work_orders[work_orders["asset_id"] == asset_id]
        open_wos = asset_wos[asset_wos["status"].str.lower() == "open"]
        open_high_priority = open_wos[
            open_wos["priority"].isin(["High", "Critical"])
        ]

        if not open_high_priority.empty:
            score += 25
            wo_ids = ", ".join(open_high_priority["work_order_id"].tolist())
            reasons.append(f"open high/critical work order {wo_ids} (+25)")
        elif not open_wos.empty:
            score += 10
            wo_ids = ", ".join(open_wos["work_order_id"].tolist())
            reasons.append(f"open work order {wo_ids} (+10)")

        asset_inspections = inspections[inspections["asset_id"] == asset_id]
        if not asset_inspections.empty:
            latest_inspection = asset_inspections.sort_values("inspection_date").iloc[-1]
            condition_score = int(latest_inspection["condition_score"])
            if condition_score < 50:
                score += 25
                reasons.append(
                    f"poor inspection condition score {condition_score} (+25)"
                )
            elif condition_score < 60:
                score += 20
                reasons.append(
                    f"poor inspection condition score {condition_score} (+20)"
                )
            elif condition_score < 75:
                score += 10
                reasons.append(
                    f"moderate inspection condition score {condition_score} (+10)"
                )

        asset_failures = failures[failures["asset_id"] == asset_id]
        total_failure_downtime = asset_failures["downtime_hours"].sum()
        total_wo_downtime = asset_wos["downtime_hours"].sum()
        total_downtime = int(total_failure_downtime + total_wo_downtime)

        if total_downtime >= 25:
            score += 20
            reasons.append(f"high total downtime of {total_downtime} hours (+20)")
        elif total_downtime >= 10:
            score += 12
            reasons.append(f"moderate total downtime of {total_downtime} hours (+12)")
        elif total_downtime > 0:
            score += 5
            reasons.append(f"recorded downtime of {total_downtime} hours (+5)")

        failure_count = len(asset_failures)
        if failure_count == 1:
            score += 10
            reasons.append("one recorded failure event (+10)")
        elif failure_count > 1:
            score += 20
            reasons.append(f"{failure_count} recorded failure events (+20)")
            score += 10
            reasons.append("repeated failures on the same asset (+10)")

        rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset["asset_name"],
                "asset_type": asset["asset_type"],
                "plant_area": asset["plant_area"],
                "criticality": criticality,
                "risk_score": int(score),
                "risk_level": _risk_level(score),
                "risk_reason": "; ".join(reasons),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["risk_score", "criticality"], ascending=[False, True])
        .reset_index(drop=True)
    )
