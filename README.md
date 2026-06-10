# Industrial Asset Intelligence RAG Demo

A Streamlit portfolio project for industrial asset intelligence in asset-intensive industries such as mining, LNG, oil and gas, power, utilities, and process plants.

The app combines synthetic asset, maintenance, inspection, and failure data to demonstrate transparent risk scoring, maintenance analytics, knowledge graph thinking, and source-backed assistant answers.

## Why I Built It

Industrial teams often need to prioritise maintenance decisions using fragmented data from asset registers, work orders, inspections, and failure history. I built this prototype to show how data science and AI-style retrieval patterns can make asset risk easier to explain and investigate.

## Features

- Asset Overview: plant-level metrics, asset counts, risk summary, and asset register.
- Risk Ranking: explainable risk scores with Low, Medium, High, and Critical levels.
- Asset Detail: drill-down into work orders, inspections, failures, and risk reasons.
- Knowledge Graph: NetworkX graph linking assets to work orders, inspections, failures, and related equipment.
- AI Assistant: deterministic source-backed answers plus local RAG-style retrieval over the CSV data.
- About Project: context on industrial asset intelligence, maintenance analytics, RAG-style answering, and synthetic data.

## Risk Scoring

The `calculate_risk_scores()` function considers:

- Asset criticality
- Open high or critical work orders
- Latest inspection condition score
- Total downtime from work orders and failures
- Failure history
- Repeated failures on the same asset

Each asset receives a transparent `risk_reason` so the ranking can be explained without treating the score as a black box.

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- NetworkX
- Matplotlib
- Sentence Transformers
- ChromaDB
- CSV data

## Project Structure

```text
asset-intelligence-rag-demo/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- assets.csv
|   |-- work_orders.csv
|   |-- inspections.csv
|   `-- failure_events.csv
|-- src/
|   |-- data_loader.py
|   |-- risk_scoring.py
|   |-- assistant.py
|   |-- graph_builder.py
|   `-- rag_engine.py
`-- docs/
    `-- screenshots/
        |-- risk_ranking.png
        `-- ai_assistant.png
```

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app from the project root:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Example Questions

The AI Assistant supports questions such as:

- Which assets are high risk?
- Why is C-201 high risk?
- Why is P-101 high risk?
- Show maintenance history of P-101
- Which assets caused the most downtime?
- Which assets have open work orders?
- Which assets need urgent attention?
- What inspection findings exist for HX-401?

Answers include supporting sources such as `assets.csv: C-201`, `work_orders.csv: WO-005`, `inspections.csv: IN-002`, and `failure_events.csv: F-003`.

## Local RAG Retrieval

RAG means retrieval-augmented generation. In this project, the current implementation focuses on the retrieval part without using a paid API or external LLM.

The app converts each row from the CSV files into a readable text chunk:

- Asset rows become asset register descriptions.
- Work order rows become maintenance history descriptions.
- Inspection rows become condition, finding, and recommendation descriptions.
- Failure rows become failure mode, root cause, and downtime descriptions.

Each chunk keeps metadata for the source file, record ID, and asset ID. The `SimpleRAGEngine` in `src/rag_engine.py` embeds those chunks with the local Sentence Transformers model `all-MiniLM-L6-v2` and stores them in an in-memory ChromaDB collection.

When a user clicks **Ask with RAG Retrieval**, the app embeds the question, retrieves the top matching records, displays a short evidence-based answer, lists the sources, and shows the retrieved context chunks in expandable sections.

This version uses local retrieval only. A future improvement would be to add optional LLM generation over the retrieved context while keeping the same source citations.

## Screenshots

### Risk Ranking

![Risk Ranking](docs/screenshots/risk_ranking.png)

### AI Assistant

![AI Assistant](docs/screenshots/ai_assistant.png)

## Future Improvements

- Add more realistic asset hierarchy and equipment relationships.
- Add richer synthetic inspection narratives and maintenance notes.
- Add optional LLM generation over retrieved context.
- Add vector search over larger maintenance documents and inspection reports.
- Add trend charts for inspection condition and downtime over time.
- Add predictive maintenance modelling once more historical data exists.
- Add deployment packaging with Docker or Streamlit Community Cloud.

## Data Note

All data is synthetic and used only for demonstration. It does not represent any real company, plant, equipment fleet, or operational system.
