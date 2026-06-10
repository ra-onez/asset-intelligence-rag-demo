import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from src.assistant import SUPPORTED_QUESTIONS, answer_question
from src.data_loader import load_data
from src.graph_builder import build_graph
from src.risk_scoring import calculate_risk_scores


st.set_page_config(
    page_title="Industrial Asset Intelligence",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_risk_scores(assets, work_orders, inspections, failures):
    return calculate_risk_scores(assets, work_orders, inspections, failures)


def format_dates(df):
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted[column]):
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
            formatted[column] = formatted[column].fillna("")
    return formatted


def render_page_intro(title, body):
    st.title(title)
    st.caption(body)


def render_asset_overview(assets, work_orders, inspections, failures, risk_df):
    render_page_intro(
        "Asset Overview",
        "Plant-level snapshot of the synthetic asset register, work orders, inspections, and downtime.",
    )

    total_downtime = int(work_orders["downtime_hours"].sum() + failures["downtime_hours"].sum())
    open_work_orders = len(work_orders[work_orders["status"].str.lower() == "open"])
    critical_assets = len(assets[assets["criticality"] == "High"])
    urgent_assets = len(risk_df[risk_df["risk_level"].isin(["High", "Critical"])])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Assets", len(assets))
    col2.metric("High Criticality", critical_assets)
    col3.metric("Open Work Orders", open_work_orders)
    col4.metric("Downtime Hours", total_downtime)

    col5, col6 = st.columns([2, 1])
    with col5:
        st.subheader("Asset Count by Type")
        asset_type_counts = assets.groupby("asset_type", as_index=False).size()
        asset_type_counts = asset_type_counts.rename(columns={"size": "count"})
        fig = px.bar(
            asset_type_counts,
            x="asset_type",
            y="count",
            color="asset_type",
            text="count",
            labels={"asset_type": "Asset Type", "count": "Assets"},
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Risk Summary")
        risk_counts = risk_df.groupby("risk_level", as_index=False).size()
        risk_counts = risk_counts.rename(columns={"size": "count"})
        st.metric("High or Critical Risk", urgent_assets)
        st.dataframe(risk_counts, use_container_width=True, hide_index=True)

    st.subheader("Asset Register")
    st.write("Each row represents an asset in the demo plant dataset.")
    st.dataframe(format_dates(assets), use_container_width=True, hide_index=True)


def render_risk_ranking(risk_df):
    render_page_intro(
        "Risk Ranking",
        "Transparent rule-based scoring that prioritises assets using criticality, open work, inspection condition, downtime, and failures.",
    )

    st.subheader("Ranked Assets")
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        risk_df,
        x="asset_id",
        y="risk_score",
        color="risk_level",
        hover_data=["asset_name", "asset_type", "plant_area", "risk_reason"],
        labels={"asset_id": "Asset", "risk_score": "Risk Score"},
        category_orders={"risk_level": ["Low", "Medium", "High", "Critical"]},
        color_discrete_map={
            "Low": "#2ca02c",
            "Medium": "#f2b701",
            "High": "#ff7f0e",
            "Critical": "#d62728",
        },
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Risk is explainable by design: each score includes the factors that contributed points."
    )


def render_asset_detail(assets, work_orders, inspections, failures, risk_df):
    render_page_intro(
        "Asset Detail",
        "Select an asset to inspect its risk explanation, work orders, inspection findings, and failure events.",
    )

    options = assets["asset_id"] + " - " + assets["asset_name"]
    selected_label = st.selectbox("Select asset", options)
    selected_asset = selected_label.split(" - ")[0]

    asset_info = assets[assets["asset_id"] == selected_asset]
    asset_risk = risk_df[risk_df["asset_id"] == selected_asset]
    asset_wos = work_orders[work_orders["asset_id"] == selected_asset]
    asset_inspections = inspections[inspections["asset_id"] == selected_asset]
    asset_failures = failures[failures["asset_id"] == selected_asset]

    risk = asset_risk.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", int(risk["risk_score"]))
    col2.metric("Risk Level", risk["risk_level"])
    col3.metric("Criticality", risk["criticality"])

    st.subheader("Risk Explanation")
    st.write(risk["risk_reason"])

    st.subheader("Asset Information")
    st.dataframe(format_dates(asset_info), use_container_width=True, hide_index=True)

    st.subheader("Work Orders")
    st.dataframe(format_dates(asset_wos), use_container_width=True, hide_index=True)

    st.subheader("Inspections")
    st.dataframe(format_dates(asset_inspections), use_container_width=True, hide_index=True)

    st.subheader("Failure Events")
    st.dataframe(format_dates(asset_failures), use_container_width=True, hide_index=True)


def render_knowledge_graph(assets, work_orders, inspections, failures):
    render_page_intro(
        "Knowledge Graph",
        "NetworkX view of how assets connect to work orders, inspections, failures, and related equipment.",
    )

    graph, relationship_df = build_graph(assets, work_orders, inspections, failures)

    node_colors = {
        "Asset": "#1f77b4",
        "Work Order": "#ff7f0e",
        "Inspection": "#2ca02c",
        "Failure Event": "#d62728",
    }
    colors = [
        node_colors.get(graph.nodes[node].get("node_type"), "#7f7f7f")
        for node in graph.nodes
    ]
    labels = {node: node for node in graph.nodes}

    pos = nx.spring_layout(graph, seed=42, k=0.7)
    fig, ax = plt.subplots(figsize=(13, 8))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=1700, alpha=0.92, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=1.2, alpha=0.55, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)
    ax.set_axis_off()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Relationship Table")
    st.write("This table shows the explicit triples used to construct the graph.")
    st.dataframe(relationship_df, use_container_width=True, hide_index=True)


def render_ai_assistant(assets, work_orders, inspections, failures, risk_df):
    render_page_intro(
        "AI Assistant",
        "Ask deterministic, source-backed questions over the project CSV data.",
    )

    selected_example = st.selectbox("Example questions", SUPPORTED_QUESTIONS)
    question = st.text_input("Ask a question", value=selected_example)

    if st.button("Ask", type="primary"):
        st.subheader("Answer")
        st.markdown(
            answer_question(question, risk_df, assets, work_orders, inspections, failures)
        )

    st.subheader("Supported Questions")
    st.write("The assistant is intentionally small and transparent for this prototype.")
    st.dataframe(pd.DataFrame({"question": SUPPORTED_QUESTIONS}), hide_index=True, use_container_width=True)


def render_about_project():
    render_page_intro(
        "About Project",
        "A portfolio prototype for industrial asset intelligence and source-backed asset reasoning.",
    )

    st.subheader("Why Industrial Asset Intelligence Matters")
    st.write(
        "Asset-intensive industries such as mining, LNG, oil and gas, power, utilities, "
        "and process plants rely on equipment reliability, maintenance planning, and risk "
        "visibility. Asset data is often spread across registers, work orders, inspection "
        "notes, failure histories, and engineering context."
    )

    st.subheader("What This Prototype Demonstrates")
    st.write(
        "This app combines synthetic CSV data into a simple workflow for asset overview, "
        "risk prioritisation, asset drill-down, graph relationships, and source-backed "
        "assistant answers. The goal is to show practical data science thinking rather "
        "than a black-box model."
    )

    st.subheader("How It Relates to RAG and Knowledge Graphs")
    st.write(
        "The assistant answers from structured project data and cites the source records "
        "used in each answer. The knowledge graph expresses relationships between assets, "
        "work orders, inspections, failures, and related equipment. Together, these patterns "
        "mirror the foundations of retrieval-augmented generation and graph-based asset "
        "intelligence systems."
    )

    st.warning(
        "All data in this project is synthetic and used only for demonstration. It does not "
        "represent any real plant, company, or operational system."
    )


assets, work_orders, inspections, failures = get_data()
risk_df = get_risk_scores(assets, work_orders, inspections, failures)

st.sidebar.title("Industrial Asset Intelligence")
st.sidebar.caption("Synthetic maintenance analytics demo")

page = st.sidebar.radio(
    "Navigation",
    [
        "Asset Overview",
        "Risk Ranking",
        "Asset Detail",
        "Knowledge Graph",
        "AI Assistant",
        "About Project",
    ],
)

if page == "Asset Overview":
    render_asset_overview(assets, work_orders, inspections, failures, risk_df)
elif page == "Risk Ranking":
    render_risk_ranking(risk_df)
elif page == "Asset Detail":
    render_asset_detail(assets, work_orders, inspections, failures, risk_df)
elif page == "Knowledge Graph":
    render_knowledge_graph(assets, work_orders, inspections, failures)
elif page == "AI Assistant":
    render_ai_assistant(assets, work_orders, inspections, failures, risk_df)
else:
    render_about_project()
