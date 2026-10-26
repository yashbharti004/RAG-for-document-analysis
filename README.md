# University Policy Assistant (Local RAG)

A locally-hosted Retrieval-Augmented Generation (RAG) system for querying university policy manuals, examination regulations, and student codes with citations.

The pipeline combines dense vector embeddings with sparse BM25 keyword retrieval and cross-encoder reranking to ensure high precision when searching academic policies. Inference runs entirely locally using Ollama, requiring no external API keys or cloud dependencies.

---

## Architecture Overview

```
User Query
    │
    ├──► Dense Retrieval (FAISS + all-MiniLM-L6-v2) ──┐
    │                                                 ├──► Reciprocal Rank Fusion (Hybrid)
    └──► Sparse Keyword Search (BM25) ────────────────┘            │
                                                                   ▼
                                                       Cross-Encoder Reranker
                                                       (ms-marco-MiniLM-L-6-v2)
                                                                   │
                                                                   ▼
                                                          Top-k Context Chunks
                                                                   │
                                                                   ▼
                                                        Local LLM (Ollama / LLaMA)
                                                                   │
                                                                   ▼
                                                          Response with Citations
```

### Retrieval Design
1. **Document Ingestion & Preprocessing:** PDF documents in `data/` are parsed page-by-page using PyMuPDF. Pages with insufficient text trigger an OCR fallback via Tesseract. Repeating running headers/footers and isolated pagination markers are stripped before chunking.
2. **Hybrid Search:** Combines semantic vector similarity (FAISS with `all-MiniLM-L6-v2`) and exact keyword matching (BM25) to prevent terminology misses (e.g., specific rule numbers, course codes).
3. **Cross-Encoder Reranking:** Candidate chunks retrieved from the hybrid stage are scored with `cross-encoder/ms-marco-MiniLM-L-6-v2` to prioritize high-relevance passages before injecting into the context window.
4. **Grounded Generation:** A structured prompt instructs the local LLM to answer strictly from the retrieved context and attribute facts to document titles and page numbers.

---

## Project Structure

```
.
├── config.py             # Central application configuration & environment defaults
├── ingest.py             # PDF parsing, OCR fallback, chunking, and FAISS vector indexing
├── retriever.py          # Unified retriever (vector, hybrid, and reranking modes)
├── rag_chain.py          # Prompt formatting, Ollama LLM integration, and citations
├── api.py                # FastAPI REST endpoints (/ask, /health, /upload)
├── app.py                # Streamlit chat interface with citation inspector
├── evaluate.py           # Offline evaluation harness for benchmarking retrieval modes
├── check_chunks.py       # Helper script to inspect generated chunks and metadata
├── eval/
│   └── questions.json    # Labeled test queries, ground truth answers, and expected sources
├── data/                 # Input policy PDF documents
├── vectorstore/          # Serialized FAISS index and chunk metadata (gitignored)
├── Dockerfile            # Container definition for API and UI
├── docker-compose.yml    # Compose configuration bundling Ollama and RAG services
├── requirements.txt      # Python dependencies
└── .env.example          # Sample environment variables
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** (tested on 3.11)
- **Ollama** installed and running locally ([ollama.com](https://ollama.com))
- *(Optional)* **Tesseract OCR** if processing scanned PDF documents

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone <repository-url>
cd rag

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull the LLM Model

Ensure Ollama is running, then pull the target model (default: `llama3.2`):

```bash
ollama run llama3.2
```

*(You can also use `mistral` or any other model supported by Ollama by updating `LLM_MODEL` in `.env`).*

### 3. Add Documents and Build Index

Place university policy PDFs inside the `data/` directory:

```bash
# Ingest and index PDFs
python ingest.py
```

This processes the PDFs, applies header/footer cleaning, chunks the text, computes embeddings, and persists the FAISS vector index into `vectorstore/`.

---

## Running the Application

### Option A: Run Locally

1. Start the FastAPI backend:
```bash
python api.py
# Server runs on http://localhost:8000 (OpenAPI docs at http://localhost:8000/docs)
```

2. In a separate terminal, launch the Streamlit frontend:
```bash
streamlit run app.py
# Web interface available at http://localhost:8501
```

### Option B: Run with Docker Compose

To run the complete stack including Ollama in containers:

```bash
docker compose up --build -d

# Pull the model inside the Ollama container
docker exec -it rag-ollama ollama pull llama3.2

# Run ingestion inside the container
docker exec -it rag-app python ingest.py
```

- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

---

## Evaluation & Benchmarking

The repository includes an offline evaluation tool to test retrieval hit rates and answer quality across different pipeline configurations:

```bash
# Run full comparison across vector, hybrid, and rerank modes
python evaluate.py

# Evaluate a specific mode
python evaluate.py --mode rerank

# Run with a custom question dataset
python evaluate.py --questions ./eval/questions.json
```

Evaluation metrics computed:
- **Retrieval Hit Rate:** Percentage of test queries where the ground-truth document chunk was present in the retrieved candidates.
- **Answer Accuracy:** Keyword overlap and refusal correctness against ground-truth answers in `eval/questions.json`.

---

## Configuration Reference

Settings can be customized via environment variables or a `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data` | Directory containing source PDF manuals |
| `VECTORSTORE_DIR` | `./vectorstore` | Path for serialized FAISS index and chunk metadata |
| `LLM_MODEL` | `llama3.2` | Ollama model identifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face cross-encoder model |
| `CHUNK_SIZE` | `1000` | Target character size per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap character count between consecutive chunks |
| `TOP_K_RETRIEVE` | `30` | Candidate chunks retrieved during initial hybrid search |
| `TOP_K_FINAL` | `6` | Chunks passed to LLM context after reranking |
| `VECTOR_K` | `15` | Chunks pulled from FAISS dense retriever |
| `BM25_K` | `15` | Chunks pulled from BM25 sparse retriever |
