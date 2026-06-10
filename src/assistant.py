import re

import pandas as pd


SUPPORTED_QUESTIONS = [
    "Which assets are high risk?",
    "Why is C-201 high risk?",
    "Why is P-101 high risk?",
    "Show maintenance history of P-101",
    "Which assets caused the most downtime?",
    "Which assets have open work orders?",
    "Which assets need urgent attention?",
    "What inspection findings exist for HX-401?",
]


def _format_sources(sources):
    seen = set()
    lines = []
    for source in sources:
        key = (source["file"], source["record"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {source['file']}: {source['record']}")
    return "\n".join(lines)


def _source(file_name, record_id):
    return {"file": file_name, "record": str(record_id)}


def _asset_sources(asset_id, assets, work_orders, inspections, failures):
    sources = [_source("assets.csv", asset_id)]

    for _, row in inspections[inspections["asset_id"] == asset_id].iterrows():
        sources.append(_source("inspections.csv", row["inspection_id"]))

    for _, row in work_orders[work_orders["asset_id"] == asset_id].iterrows():
        sources.append(_source("work_orders.csv", row["work_order_id"]))

    for _, row in failures[failures["asset_id"] == asset_id].iterrows():
        sources.append(_source("failure_events.csv", row["failure_id"]))

    return sources


def _answer_with_sources(answer, sources):
    return f"{answer.strip()}\n\nSources:\n{_format_sources(sources)}"


def _asset_context(asset_id, risk_df, assets, work_orders, inspections, failures):
    asset = assets[assets["asset_id"] == asset_id].iloc[0]
    risk = risk_df[risk_df["asset_id"] == asset_id].iloc[0]
    asset_wos = work_orders[work_orders["asset_id"] == asset_id]
    open_wos = asset_wos[asset_wos["status"].str.lower() == "open"]
    asset_inspections = inspections[inspections["asset_id"] == asset_id]
    latest_inspection = None
    if not asset_inspections.empty:
        latest_inspection = asset_inspections.sort_values("inspection_date").iloc[-1]
    asset_failures = failures[failures["asset_id"] == asset_id]
    return asset, risk, asset_wos, open_wos, latest_inspection, asset_failures


def _why_asset(asset_id, risk_df, assets, work_orders, inspections, failures):
    asset, risk, _, open_wos, latest_inspection, asset_failures = _asset_context(
        asset_id, risk_df, assets, work_orders, inspections, failures
    )

    points = [
        f"{asset_id} is {risk['risk_level']} risk with a score of {risk['risk_score']}.",
        (
            f"It is a {asset['criticality'].lower()}-criticality "
            f"{asset['asset_type'].lower()} in {asset['plant_area']}."
        ),
    ]

    if latest_inspection is not None:
        points.append(
            "The latest inspection recorded a condition score of "
            f"{int(latest_inspection['condition_score'])} and found: "
            f"{latest_inspection['findings']}."
        )

    if not open_wos.empty:
        open_text = "; ".join(
            f"{row['work_order_id']} ({row['priority']}): {row['description']}"
            for _, row in open_wos.iterrows()
        )
        points.append(f"Open work orders include {open_text}.")

    if not asset_failures.empty:
        failure_text = "; ".join(
            f"{row['failure_id']} {row['failure_mode']} due to {row['root_cause']}"
            for _, row in asset_failures.iterrows()
        )
        points.append(f"Failure history includes {failure_text}.")

    points.append(f"Transparent score reason: {risk['risk_reason']}.")
    return _answer_with_sources(" ".join(points), _asset_sources(asset_id, assets, work_orders, inspections, failures))


def _high_risk_assets(risk_df, assets, work_orders, inspections, failures):
    high_risk = risk_df[risk_df["risk_level"].isin(["High", "Critical"])]
    if high_risk.empty:
        high_risk = risk_df.head(3)

    lines = ["The highest-risk assets are:"]
    sources = []
    for _, row in high_risk.iterrows():
        lines.append(
            f"- {row['asset_id']} ({row['asset_name']}): "
            f"{row['risk_level']} risk, score {row['risk_score']}. "
            f"{row['risk_reason']}"
        )
        sources.extend(
            _asset_sources(row["asset_id"], assets, work_orders, inspections, failures)
        )

    return _answer_with_sources("\n".join(lines), sources)


def _maintenance_history(asset_id, assets, work_orders, failures):
    asset_wos = work_orders[work_orders["asset_id"] == asset_id].sort_values("planned_date")
    asset_failures = failures[failures["asset_id"] == asset_id].sort_values("failure_date")

    lines = [f"Maintenance and failure history for {asset_id}:"]
    if asset_wos.empty and asset_failures.empty:
        lines.append("- No work orders or failures are recorded in the demo data.")
    else:
        for _, row in asset_wos.iterrows():
            date = row["planned_date"].date()
            lines.append(
                f"- {row['work_order_id']} on {date}: {row['work_type']} work, "
                f"{row['description']} ({row['status']}, {row['priority']}), "
                f"{int(row['downtime_hours'])} downtime hours."
            )
        for _, row in asset_failures.iterrows():
            date = row["failure_date"].date()
            lines.append(
                f"- {row['failure_id']} on {date}: {row['failure_mode']} caused by "
                f"{row['root_cause']}, {int(row['downtime_hours'])} downtime hours."
            )

    sources = [_source("assets.csv", asset_id)]
    sources.extend(_source("work_orders.csv", row["work_order_id"]) for _, row in asset_wos.iterrows())
    sources.extend(_source("failure_events.csv", row["failure_id"]) for _, row in asset_failures.iterrows())
    return _answer_with_sources("\n".join(lines), sources)


def _downtime(assets, work_orders, failures):
    wo_downtime = work_orders.groupby("asset_id", as_index=False)["downtime_hours"].sum()
    failure_downtime = failures.groupby("asset_id", as_index=False)["downtime_hours"].sum()
    downtime = pd.merge(
        wo_downtime,
        failure_downtime,
        on="asset_id",
        how="outer",
        suffixes=("_work_orders", "_failures"),
    ).fillna(0)
    downtime["total_downtime"] = (
        downtime["downtime_hours_work_orders"] + downtime["downtime_hours_failures"]
    )
    downtime = downtime.merge(assets[["asset_id", "asset_name"]], on="asset_id", how="left")
    downtime = downtime.sort_values("total_downtime", ascending=False)

    lines = ["Assets causing the most downtime are:"]
    sources = []
    for _, row in downtime.head(5).iterrows():
        lines.append(
            f"- {row['asset_id']} ({row['asset_name']}): "
            f"{int(row['total_downtime'])} total hours "
            f"({int(row['downtime_hours_work_orders'])} work order, "
            f"{int(row['downtime_hours_failures'])} failure)."
        )
        sources.append(_source("assets.csv", row["asset_id"]))
        for _, wo in work_orders[work_orders["asset_id"] == row["asset_id"]].iterrows():
            sources.append(_source("work_orders.csv", wo["work_order_id"]))
        for _, failure in failures[failures["asset_id"] == row["asset_id"]].iterrows():
            sources.append(_source("failure_events.csv", failure["failure_id"]))

    return _answer_with_sources("\n".join(lines), sources)


def _open_work_orders(assets, work_orders):
    open_wos = work_orders[work_orders["status"].str.lower() == "open"].merge(
        assets[["asset_id", "asset_name"]], on="asset_id", how="left"
    )

    if open_wos.empty:
        return _answer_with_sources(
            "There are no open work orders in the demo data.",
            [_source("work_orders.csv", "all records")],
        )

    lines = ["Assets with open work orders are:"]
    sources = []
    for _, row in open_wos.iterrows():
        lines.append(
            f"- {row['asset_id']} ({row['asset_name']}): {row['work_order_id']} "
            f"{row['description']} ({row['priority']}, planned {row['planned_date'].date()})."
        )
        sources.append(_source("assets.csv", row["asset_id"]))
        sources.append(_source("work_orders.csv", row["work_order_id"]))

    return _answer_with_sources("\n".join(lines), sources)


def _urgent_attention(risk_df, assets, work_orders, inspections, failures):
    urgent = risk_df[
        risk_df["risk_level"].isin(["High", "Critical"])
    ].copy()
    urgent = urgent.sort_values("risk_score", ascending=False).head(3)

    lines = ["Assets needing urgent attention are:"]
    sources = []
    for _, row in urgent.iterrows():
        lines.append(
            f"- {row['asset_id']} ({row['asset_name']}): {row['risk_level']} risk, "
            f"score {row['risk_score']}. {row['risk_reason']}"
        )
        sources.extend(
            _asset_sources(row["asset_id"], assets, work_orders, inspections, failures)
        )

    return _answer_with_sources("\n".join(lines), sources)


def _inspection_findings(asset_id, assets, inspections):
    asset_inspections = inspections[inspections["asset_id"] == asset_id].sort_values(
        "inspection_date"
    )

    if asset_inspections.empty:
        return _answer_with_sources(
            f"No inspection findings are recorded for {asset_id} in the demo data.",
            [_source("assets.csv", asset_id), _source("inspections.csv", "all records")],
        )

    lines = [f"Inspection findings for {asset_id}:"]
    sources = [_source("assets.csv", asset_id)]
    for _, row in asset_inspections.iterrows():
        lines.append(
            f"- {row['inspection_id']} on {row['inspection_date'].date()}: "
            f"condition score {int(row['condition_score'])}; {row['findings']}. "
            f"Recommendation: {row['recommendation']}."
        )
        sources.append(_source("inspections.csv", row["inspection_id"]))

    return _answer_with_sources("\n".join(lines), sources)


def answer_question(question, risk_df, assets, work_orders, inspections, failures):
    q = re.sub(r"\s+", " ", question.strip().lower())

    if not q:
        return "Ask a question about risk, maintenance history, downtime, open work orders, or inspection findings."

    if "downtime" in q:
        return _downtime(assets, work_orders, failures)

    if "open work order" in q or ("open" in q and "work order" in q):
        return _open_work_orders(assets, work_orders)

    if "urgent" in q or "need attention" in q or "needs attention" in q:
        return _urgent_attention(risk_df, assets, work_orders, inspections, failures)

    if "high risk" in q and not any(asset_id.lower() in q for asset_id in assets["asset_id"]):
        return _high_risk_assets(risk_df, assets, work_orders, inspections, failures)

    for asset_id in assets["asset_id"]:
        asset_key = asset_id.lower()
        if asset_key not in q:
            continue

        if "maintenance" in q or "history" in q or "work order" in q:
            return _maintenance_history(asset_id, assets, work_orders, failures)

        if "inspection" in q or "finding" in q:
            return _inspection_findings(asset_id, assets, inspections)

        if "why" in q or "risk" in q:
            return _why_asset(
                asset_id, risk_df, assets, work_orders, inspections, failures
            )

    return (
        "I can answer source-backed questions such as:\n\n"
        + "\n".join(f"- {question}" for question in SUPPORTED_QUESTIONS)
    )
