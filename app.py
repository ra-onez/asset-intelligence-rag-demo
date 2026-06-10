import pandas as pd
import streamlit as st
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Industrial Asset Knowledge Assistant",
    layout="wide"
)


@st.cache_data
def load_data():
    assets = pd.read_csv("data/assets.csv")
    work_orders = pd.read_csv("data/work_orders.csv")
    inspections = pd.read_csv("data/inspections.csv")
    failures = pd.read_csv("data/failure_events.csv")

    work_orders["planned_date"] = pd.to_datetime(work_orders["planned_date"])
    work_orders["completed_date"] = pd.to_datetime(work_orders["completed_date"], errors="coerce")
    inspections["inspection_date"] = pd.to_datetime(inspections["inspection_date"])
    failures["failure_date"] = pd.to_datetime(failures["failure_date"])

    return assets, work_orders, inspections, failures


def calculate_risk_scores(assets, work_orders, inspections, failures):
    rows = []

    for _, asset in assets.iterrows():
        asset_id = asset["asset_id"]
        score = 0
        reasons = []

        # Criticality score
        if asset["criticality"] == "High":
            score += 30
            reasons.append("High criticality asset")
        elif asset["criticality"] == "Medium":
            score += 15
            reasons.append("Medium criticality asset")
        else:
            score += 5
            reasons.append("Low criticality asset")

        # Open work orders
        asset_wos = work_orders[work_orders["asset_id"] == asset_id]
        open_wos = asset_wos[asset_wos["status"] == "Open"]

        if not open_wos.empty:
            if any(open_wos["priority"].isin(["Critical", "High"])):
                score += 25
                reasons.append("Open high/critical work order")
            else:
                score += 10
                reasons.append("Open work order")

        # Inspection condition
        asset_inspections = inspections[inspections["asset_id"] == asset_id]
        if not asset_inspections.empty:
            latest_inspection = asset_inspections.sort_values("inspection_date").iloc[-1]
            condition_score = latest_inspection["condition_score"]

            if condition_score < 60:
                score += 25
                reasons.append(f"Poor condition score: {condition_score}")
            elif condition_score < 75:
                score += 10
                reasons.append(f"Moderate condition score: {condition_score}")

        # Downtime
        asset_failures = failures[failures["asset_id"] == asset_id]
        total_failure_downtime = asset_failures["downtime_hours"].sum()
        total_wo_downtime = asset_wos["downtime_hours"].sum()
        total_downtime = total_failure_downtime + total_wo_downtime

        if total_downtime > 20:
            score += 20
            reasons.append(f"High total downtime: {total_downtime} hours")
        elif total_downtime > 10:
            score += 15
            reasons.append(f"Moderate downtime: {total_downtime} hours")

        # Recent failures
        if not asset_failures.empty:
            score += 20
            reasons.append("Recorded failure history")

        rows.append({
            "asset_id": asset_id,
            "asset_name": asset["asset_name"],
            "asset_type": asset["asset_type"],
            "plant_area": asset["plant_area"],
            "criticality": asset["criticality"],
            "risk_score": score,
            "risk_reason": "; ".join(reasons)
        })

    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)


def answer_question(question, risk_df, assets, work_orders, inspections, failures):
    q = question.lower()

    if "high risk" in q or "urgent" in q or "priority" in q:
        top_assets = risk_df.head(3)
        answer = "The highest priority assets are:\n\n"
        for _, row in top_assets.iterrows():
            answer += f"- {row['asset_id']} ({row['asset_name']}): risk score {row['risk_score']}. Reason: {row['risk_reason']}\n"
        return answer

    if "downtime" in q:
        failure_downtime = failures.groupby("asset_id")["downtime_hours"].sum().reset_index()
        wo_downtime = work_orders.groupby("asset_id")["downtime_hours"].sum().reset_index()
        downtime = pd.merge(failure_downtime, wo_downtime, on="asset_id", how="outer", suffixes=("_failure", "_wo")).fillna(0)
        downtime["total_downtime"] = downtime["downtime_hours_failure"] + downtime["downtime_hours_wo"]
        downtime = downtime.merge(assets[["asset_id", "asset_name"]], on="asset_id", how="left")
        downtime = downtime.sort_values("total_downtime", ascending=False)

        answer = "Assets with the highest downtime are:\n\n"
        for _, row in downtime.head(3).iterrows():
            answer += f"- {row['asset_id']} ({row['asset_name']}): {row['total_downtime']} hours\n"
        return answer

    for asset_id in assets["asset_id"]:
        if asset_id.lower() in q:
            asset_row = assets[assets["asset_id"] == asset_id].iloc[0]
            risk_row = risk_df[risk_df["asset_id"] == asset_id].iloc[0]
            asset_wos = work_orders[work_orders["asset_id"] == asset_id]
            asset_inspections = inspections[inspections["asset_id"] == asset_id]
            asset_failures = failures[failures["asset_id"] == asset_id]

            answer = f"Asset {asset_id} is {asset_row['asset_name']}, a {asset_row['criticality']} criticality {asset_row['asset_type']} in {asset_row['plant_area']}.\n\n"
            answer += f"Risk score: {risk_row['risk_score']}\n"
            answer += f"Risk reason: {risk_row['risk_reason']}\n\n"

            if not asset_inspections.empty:
                latest = asset_inspections.sort_values("inspection_date").iloc[-1]
                answer += f"Latest inspection finding: {latest['findings']}\n"
                answer += f"Recommendation: {latest['recommendation']}\n\n"

            if not asset_wos.empty:
                answer += "Work order history:\n"
                for _, wo in asset_wos.iterrows():
                    answer += f"- {wo['work_order_id']}: {wo['description']} ({wo['status']}, {wo['priority']})\n"

            if not asset_failures.empty:
                answer += "\nFailure history:\n"
                for _, failure in asset_failures.iterrows():
                    answer += f"- {failure['failure_id']}: {failure['failure_mode']} due to {failure['root_cause']}\n"

            return answer

    return (
        "I can answer questions about high-risk assets, downtime, open work orders, "
        "or specific assets such as P-101, C-201, HX-401, V-301, P-102, and M-101."
    )


def build_graph(assets, work_orders, inspections, failures):
    G = nx.Graph()

    for _, row in assets.iterrows():
        G.add_node(row["asset_id"], label=row["asset_name"], node_type="Asset")

    for _, row in work_orders.iterrows():
        G.add_node(row["work_order_id"], label=row["description"], node_type="Work Order")
        G.add_edge(row["asset_id"], row["work_order_id"], relationship="HAS_WORK_ORDER")

    for _, row in inspections.iterrows():
        G.add_node(row["inspection_id"], label=row["findings"], node_type="Inspection")
        G.add_edge(row["asset_id"], row["inspection_id"], relationship="HAS_INSPECTION")

    for _, row in failures.iterrows():
        G.add_node(row["failure_id"], label=row["failure_mode"], node_type="Failure")
        G.add_edge(row["asset_id"], row["failure_id"], relationship="HAD_FAILURE")

    # Example equipment relationship
    G.add_edge("P-101", "M-101", relationship="CONNECTED_TO")

    return G


assets, work_orders, inspections, failures = load_data()
risk_df = calculate_risk_scores(assets, work_orders, inspections, failures)

st.title("Industrial Asset Knowledge Assistant")
st.caption("A prototype for asset-intensive industries that connects asset registers, work orders, inspection findings, "
    "failure history, risk scoring, and graph-based asset intelligence.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Asset Overview",
        "Risk Ranking",
        "Asset Detail",
        "Knowledge Graph",
        "AI Assistant"
    ]
)

if page == "Asset Overview":
    st.header("Asset Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_assets = len(assets)
    high_criticality = len(assets[assets["criticality"] == "High"])
    open_work_orders = len(work_orders[work_orders["status"] == "Open"])
    total_downtime = work_orders["downtime_hours"].sum() + failures["downtime_hours"].sum()

    col1.metric("Total Assets", total_assets)
    col2.metric("High Criticality Assets", high_criticality)
    col3.metric("Open Work Orders", open_work_orders)
    col4.metric("Total Downtime Hours", total_downtime)

    st.subheader("Assets by Type")
    fig = px.bar(
        assets.groupby("asset_type").size().reset_index(name="count"),
        x="asset_type",
        y="count",
        title="Asset Count by Type"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Asset Register")
    st.dataframe(assets, use_container_width=True)


elif page == "Risk Ranking":
    st.header("Risk Ranking")

    st.write("This table ranks assets using a transparent rule-based risk score.")
    st.dataframe(risk_df, use_container_width=True)

    fig = px.bar(
        risk_df,
        x="asset_id",
        y="risk_score",
        color="criticality",
        hover_data=["asset_name", "risk_reason"],
        title="Asset Risk Score"
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "Asset Detail":
    st.header("Asset Detail")

    selected_asset = st.selectbox(
        "Select an asset",
        assets["asset_id"].tolist()
    )

    asset_info = assets[assets["asset_id"] == selected_asset]
    asset_risk = risk_df[risk_df["asset_id"] == selected_asset]
    asset_wos = work_orders[work_orders["asset_id"] == selected_asset]
    asset_inspections = inspections[inspections["asset_id"] == selected_asset]
    asset_failures = failures[failures["asset_id"] == selected_asset]

    st.subheader("Asset Information")
    st.dataframe(asset_info, use_container_width=True)

    st.subheader("Risk Explanation")
    st.dataframe(asset_risk[["asset_id", "risk_score", "risk_reason"]], use_container_width=True)

    st.subheader("Work Orders")
    st.dataframe(asset_wos, use_container_width=True)

    st.subheader("Inspections")
    st.dataframe(asset_inspections, use_container_width=True)

    st.subheader("Failure Events")
    st.dataframe(asset_failures, use_container_width=True)


elif page == "Knowledge Graph":
    st.header("Knowledge Graph")

    st.write("This graph connects assets with work orders, inspections, failure events, and related equipment.")

    G = build_graph(assets, work_orders, inspections, failures)
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1800,
        font_size=8,
        ax=ax
    )
    st.pyplot(fig)

    st.write("Graph relationships include:")
    st.code(
        """
Asset -> HAS_WORK_ORDER -> Work Order
Asset -> HAS_INSPECTION -> Inspection
Asset -> HAD_FAILURE -> Failure Event
Asset -> CONNECTED_TO -> Related Asset
        """
    )


elif page == "AI Assistant":
    st.header("AI Assistant")

    st.write("Ask a question about plant assets, maintenance history, downtime, or risk.")

    example_questions = [
        "Which assets are high risk?",
        "Why is C-201 high risk?",
        "Show maintenance history of P-101",
        "Which assets caused the most downtime?",
        "Which assets need urgent attention?"
    ]

    selected_example = st.selectbox("Try an example question", example_questions)
    user_question = st.text_input("Ask your question", value=selected_example)

    if st.button("Ask"):
        answer = answer_question(user_question, risk_df, assets, work_orders, inspections, failures)
        st.markdown("### Answer")
        st.write(answer)