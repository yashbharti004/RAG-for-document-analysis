"""
Retrieval layer supporting vector similarity, hybrid search (BM25 + FAISS),
and cross-encoder reranking.
"""

import os
import pickle
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


class PolicyRetriever:
    """Unified retriever that supports vector, hybrid, and reranked search."""

    def __init__(self, mode: str = "rerank"):
        if mode not in ("vector", "hybrid", "rerank"):
            raise ValueError(f"Unknown retriever mode: '{mode}'. Expected 'vector', 'hybrid', or 'rerank'.")

        self.mode = mode
        self.embeddings = None
        self.vectorstore = None
        self.vector_retriever = None
        self.chunks: List[Document] = []
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.reranker = None

        self._load_vectorstore()
        self._load_chunks()

        if mode in ("hybrid", "rerank"):
            self._build_bm25()

        if mode == "rerank":
            self._load_reranker()

        logger.info(f"PolicyRetriever initialized in '{mode}' mode")

    def _load_vectorstore(self):
        if not os.path.exists(config.VECTORSTORE_DIR):
            raise FileNotFoundError(
                f"Vectorstore not found at {config.VECTORSTORE_DIR}. Please run ingest.py first."
            )

        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"Loading FAISS index from {config.VECTORSTORE_DIR}")
        self.vectorstore = FAISS.load_local(
            config.VECTORSTORE_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.VECTOR_K},
        )

    def _load_chunks(self):
        if not os.path.exists(config.CHUNKS_FILE):
            raise FileNotFoundError(
                f"Chunks file not found at {config.CHUNKS_FILE}. Please run ingest.py first."
            )

        with open(config.CHUNKS_FILE, "rb") as fh:
            self.chunks = pickle.load(fh)
        logger.info(f"Loaded {len(self.chunks)} chunk(s) from {config.CHUNKS_FILE}")

    def _build_bm25(self):
        logger.info("Initializing BM25 index over chunks...")
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = config.BM25_K

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=config.ENSEMBLE_WEIGHTS,
        )

    def _load_reranker(self):
        logger.info(f"Loading cross-encoder reranker: {config.RERANKER_MODEL}")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def _rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)

        # Sort descending by cross-encoder relevance score
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieve most relevant chunks for a given query."""
        top_k = top_k or config.TOP_K_FINAL

        if self.mode == "vector":
            return self.vectorstore.similarity_search(query, k=top_k)

        if self.mode == "hybrid":
            docs = self.ensemble_retriever.invoke(query)
            return self._deduplicate(docs)[:top_k]

        # Mode: rerank
        # Query both dense and sparse retrievers first with wider pool
        self.vector_retriever.search_kwargs["k"] = config.VECTOR_K
        self.bm25_retriever.k = config.BM25_K

        candidates = self.ensemble_retriever.invoke(query)
        unique_candidates = self._deduplicate(candidates)
        return self._rerank(query, unique_candidates, top_k)

    @staticmethod
    def _deduplicate(docs: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in docs:
            # Deduplicate based on text content
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique.append(doc)
        return unique


def get_retriever(mode: str = "rerank") -> PolicyRetriever:
    return PolicyRetriever(mode=mode)
