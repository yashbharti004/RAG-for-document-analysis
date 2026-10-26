# University Policy RAG

A local question-answering assistant for university policy manuals, examination regulations, and student codes.

It uses a two-stage retrieval pipeline (**BM25 + FAISS embeddings** merged with reciprocal rank fusion, then reranked with a **cross-encoder**) to fetch relevant clauses, and generates answers with page citations using a local **Ollama** model.

---

## How It Works

```
PDFs (data/) ──► Text Cleaning (OCR fallback) ──► Chunks (1000 chars)
                                                      │
User Query ──► [FAISS Dense + BM25 Sparse] ──► Cross-Encoder ──► Ollama (llama3.2) ──► Answer + Citation
```

1. **Ingestion & OCR:** Reads PDFs using PyMuPDF. Low-text / scanned pages automatically trigger Tesseract OCR. Recurring headers, footers, and page numbers are stripped before chunking.
2. **Hybrid Retrieval:** FAISS (`all-MiniLM-L6-v2`) handles semantic intent, while BM25 captures exact policy numbers and terminology.
3. **Cross-Encoder Reranking:** The top 30 candidate chunks from hybrid retrieval are re-scored with `cross-encoder/ms-marco-MiniLM-L-6-v2`, passing the 6 most relevant chunks to the LLM.
4. **Local Generation:** Ollama generates the final answer grounded strictly in the retrieved context with explicit citations (`[Source: <doc>, Page: <page>]`).

---

## Quickstart

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) running locally

```bash
# 1. Pull the model
ollama run llama3.2

# 2. Setup environment
python -m venv venv
venv\Scripts\activate   # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# 3. Add PDFs into data/ and build index
python ingest.py

# 4. Start the API
python api.py           # http://localhost:8000

# 5. Start the chat UI (new terminal)
streamlit run app.py    # http://localhost:8501
```

Or run with Docker:
```bash
docker compose up --build
```

---

## Project Structure

- `ingest.py` — PDF parsing, OCR fallback, text cleaning, and FAISS indexing.
- `retriever.py` — Hybrid search (FAISS + BM25) and cross-encoder reranker.
- `rag_chain.py` — Prompt formatting, Ollama invocation, and citation handling.
- `api.py` — FastAPI service (`POST /ask`, `GET /health`, `POST /upload`).
- `app.py` — Streamlit chat interface with expandable source cards.
- `evaluate.py` — Offline benchmark script testing hit rate against `eval/questions.json`.
- `config.py` — Application configuration and environment variable defaults.

---

## Configuration

Settings can be customized via `.env` or directly in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Ollama model identifier |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text chunk size and overlap |
| `TOP_K_RETRIEVE` | `30` | Candidate pool before reranking |
| `TOP_K_FINAL` | `6` | Chunks injected into LLM context |
