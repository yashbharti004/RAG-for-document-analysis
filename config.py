import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# File & directory paths
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectorstore"))
CHUNKS_FILE = os.getenv("CHUNKS_FILE", str(Path(VECTORSTORE_DIR) / "chunks.pkl"))
EVAL_DIR = os.getenv("EVAL_DIR", str(BASE_DIR / "eval"))

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval parameters
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "30"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "6"))
BM25_K = int(os.getenv("BM25_K", "15"))
VECTOR_K = int(os.getenv("VECTOR_K", "15"))
ENSEMBLE_WEIGHTS = [0.5, 0.5]
