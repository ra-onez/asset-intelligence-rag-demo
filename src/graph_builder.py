import networkx as nx
import pandas as pd


RELATED_ASSETS = [
    ("P-101", "M-101", "CONNECTED_TO"),
    ("C-201", "HX-401", "CONNECTED_TO"),
]


def build_graph(assets, work_orders, inspections, failures):
    graph = nx.Graph()
    relationships = []

    for _, row in assets.iterrows():
        graph.add_node(
            row["asset_id"],
            label=row["asset_name"],
            node_type="Asset",
        )

    for _, row in work_orders.iterrows():
        graph.add_node(
            row["work_order_id"],
            label=row["description"],
            node_type="Work Order",
        )
        graph.add_edge(row["asset_id"], row["work_order_id"], relationship="HAS_WORK_ORDER")
        relationships.append(
            {
                "source": row["asset_id"],
                "relationship": "HAS_WORK_ORDER",
                "target": row["work_order_id"],
            }
        )

    for _, row in inspections.iterrows():
        graph.add_node(
            row["inspection_id"],
            label=row["findings"],
            node_type="Inspection",
        )
        graph.add_edge(row["asset_id"], row["inspection_id"], relationship="HAS_INSPECTION")
        relationships.append(
            {
                "source": row["asset_id"],
                "relationship": "HAS_INSPECTION",
                "target": row["inspection_id"],
            }
        )

    for _, row in failures.iterrows():
        graph.add_node(
            row["failure_id"],
            label=row["failure_mode"],
            node_type="Failure Event",
        )
        graph.add_edge(row["asset_id"], row["failure_id"], relationship="HAD_FAILURE")
        relationships.append(
            {
                "source": row["asset_id"],
                "relationship": "HAD_FAILURE",
                "target": row["failure_id"],
            }
        )

    known_assets = set(assets["asset_id"])
    for source, target, relationship in RELATED_ASSETS:
        if source in known_assets and target in known_assets:
            graph.add_edge(source, target, relationship=relationship)
            relationships.append(
                {
                    "source": source,
                    "relationship": relationship,
                    "target": target,
                }
            )

    return graph, pd.DataFrame(relationships)
