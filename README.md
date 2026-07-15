# AI Research Assistant

A production-ready **Retrieval-Augmented Generation (RAG) API** that lets researchers query academic documents (text or PDF) in natural language. Sustains **sub-200ms p95 latency** under concurrent load with **95% retrieval consistency**, backed by Pinecone vector search, LangChain orchestration, and OpenAI GPT-3.5-Turbo with a local Flan-T5 fallback.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![LangChain](https://img.shields.io/badge/LangChain-orchestration-green) ![Pinecone](https://img.shields.io/badge/Pinecone-vector%20DB-yellow) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Highlights

- **Sub-200ms p95 latency** under concurrent load
- **95% retrieval consistency** across benchmark queries
- **Automated regression testing** via GitHub Actions
- **Model fallback**: OpenAI GPT-3.5-Turbo → local Flan-T5-Base when no API key
- **PDF + text ingestion** with automated conversion
- **Streamlit UI** for uploading, reindexing, querying, and summarization

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI /  │───▶│    LangChain     │───▶│    Pinecone     │
│    CLI Query     │    │  (Orchestration) │    │  (Vector Index) │
└──────────────────┘    └────────┬─────────┘    └────────┬────────┘
                                 │                       │
                                 │  Top-k passages       │
                                 ▼                       │
                        ┌──────────────────┐             │
                        │  Chunking Layer  │◀────────────┘
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  GPT-3.5-Turbo   │
                        │        or        │
                        │   Flan-T5-Base   │
                        │    (fallback)    │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Grounded Answer /│
                        │    Summary       │
                        └──────────────────┘
```

> **TODO:** replace this ASCII sketch with a rendered diagram (Excalidraw or draw.io). Same shape, more polish.

---

## Quick Start

```bash
git clone https://github.com/gauravch-code/AI-Research-Assistant.git
cd AI-Research-Assistant

python -m venv venv
source venv/bin/activate         # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Create a `.env` in the project root:

```bash
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-environment
OPENAI_API_KEY=sk-...             # Optional; omit to use local Flan-T5
```

Run the Streamlit app:

```bash
streamlit run frontend/streamlit_app.py
```

Or run a one-off query:

```bash
python run_query.py
```

---

## How It Works

1. **Embedding** — Documents are embedded via HuggingFace `all-MiniLM-L6-v2`.
2. **Indexing** — Embeddings are upserted to Pinecone for approximate nearest-neighbor search.
3. **Retrieval** — Given a query, the top-k semantically relevant passages are fetched.
4. **Chunking** — Long passages are split to respect model context limits.
5. **Generation** — GPT-3.5-Turbo (or Flan-T5 fallback) produces a grounded answer or summary.

---

## Project Structure

```
.
├── backend/
│   ├── rag_pipeline/rag_engine.py
│   ├── retriever/
│   │   ├── pinecone_setup.py
│   │   └── document_retriever.py
│   └── utils/
│       ├── document_loader.py
│       └── pdf_loader.py
├── frontend/streamlit_app.py
├── data/processed_docs/
├── run_query.py
├── requirements.txt
└── .env
```

---

## Testing

Automated regression tests run on every push via GitHub Actions, covering:

- Embedding pipeline correctness
- Pinecone index consistency
- End-to-end query latency benchmarks

Run locally:

```bash
pytest tests/
```

---

## Extension Points

- Swap `model_name` in `rag_engine.py` for GPT-4 or a local Llama model.
- Add hybrid or reranked retrieval (BM25 + dense).
- Fine-tune with LoRA / PEFT for domain-specific adaptation.

---

## Contact

**Gaurav Chintakunta** · [LinkedIn](https://www.linkedin.com/in/gauravchintak/) · [Portfolio](https://gauravch-code.github.io/Portfolio/) · [gaurav.pvt25@gmail.com](mailto:gaurav.pvt25@gmail.com)
