# Industrial Asset Knowledge Assistant

This project is a small AI/data science prototype for asset-intensive industries such as mining, LNG, power, and process plants.

The app connects asset registers, maintenance work orders, inspection findings, and failure history to create a simple asset intelligence dashboard.

## Why I built this

I built this project to explore how data science, knowledge graphs, and RAG-based AI assistants can support industrial asset management and maintenance decision-making.

## Main Features

- Asset overview dashboard
- Maintenance work order analysis
- Risk scoring for industrial equipment
- Asset-level maintenance history
- Simple knowledge graph connecting assets, inspections, failures, and work orders
- AI assistant for asking questions over asset data

## Tech Stack

- Python
- Pandas
- Streamlit
- NetworkX
- ChromaDB / FAISS
- LangChain / LlamaIndex
- SQL/CSV data modelling

## Example Questions

- Which assets are high risk?
- Why is Compressor C-201 high risk?
- Which assets have open critical work orders?
- What is the maintenance history of Pump P-101?
- Which assets caused the most downtime?

## Future Improvements

- Integrate Neo4j for graph database storage
- Add real-time maintenance scheduling data
- Improve RAG evaluation and source traceability
- Add predictive maintenance models
- Add shutdown planning optimisation
