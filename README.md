# 🤖 Agentic AI Chat Assistant

A unified **Agentic AI System** built with **Streamlit**, supporting:

- 🔍 SQL querying (PostgreSQL)
- 📄 RAG-based document Q&A (Pinecone)
- 📊 Data visualizations
- 📋 Comparison tables
- 📈 ML forecasting
- 🧾 Automated PDF report generation
- 💬 Clean chat-only UI

---

## 🧠 Architecture Overview

- `agentic_system.py` → Central router & agent orchestration
- `table_agent.py` → Comparison tables
- `analytics_visuals.py` → Chart generation
- `etl_agent.py` → CSV → PostgreSQL ingestion
- `pdf_embedding.py` → PDF → Pinecone embeddings
- `app.py` → Streamlit chat interface

---

## 📁 Project Structure

project/
│
├── app.py
├── agentic_system.py
├── table_agent.py
├── analytics_visuals.py
├── etl_agent.py
├── pdf_embedding.py
├── db_connection.py
├── pinecone_setup.py
├── requirements.txt
├── .gitignore
└── data/
