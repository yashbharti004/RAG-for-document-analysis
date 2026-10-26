"""
RAG pipeline connecting retrieval with an Ollama LLM generator.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from retriever import get_retriever
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an assistant answering questions about university policies. Answer the question accurately using ONLY the provided context.

Rules:
1. Rely strictly on the context. If multiple sections or documents are relevant, combine them in your answer.
2. Cite all relevant sources and page numbers in your answer, e.g. [Source: <doc>, Page: <page>].
3. If the context does not contain enough information to answer, simply state:
   "I don't have enough information in the provided policy documents to answer this question."
4. Do not assume or extrapolate rules not stated in the context.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"],
)


def format_context(documents: List[Document]) -> str:
    """Format retrieved document chunks into structured text for the prompt."""
    parts: List[str] = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")

        header = f"[Doc {i}: {source}, Page {page}"
        if section:
            header += f" | {section}"
        header += "]"

        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def format_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """Extract distinct source references and text snippets from retrieved documents."""
    sources: List[Dict[str, Any]] = []
    seen: set[tuple] = set()

    for doc in documents:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)

        snippet = doc.page_content.strip()
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "section": doc.metadata.get("section", ""),
            "snippet": (snippet[:200] + "...") if len(snippet) > 200 else snippet,
        })
    return sources


class RAGChain:
    """End-to-end RAG orchestrator."""

    def __init__(
        self,
        retriever_mode: str = "rerank",
        llm_model: Optional[str] = None,
    ):
        self.retriever_mode = retriever_mode
        self.model_name = llm_model or config.LLM_MODEL
        self.retriever = get_retriever(mode=retriever_mode)
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
        )
        logger.info(f"RAGChain ready (mode={retriever_mode}, model={self.model_name})")

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query the knowledge base and return an answer with cited sources."""
        documents = self.retriever.retrieve(question, top_k=top_k)

        if not documents:
            return {
                "question": question,
                "answer": "No relevant policy documents could be found for this question.",
                "sources": [],
                "num_chunks_retrieved": 0,
            }

        context = format_context(documents)
        prompt = RAG_PROMPT.format(context=context, question=question)

        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as exc:
            err_msg = str(exc).lower()
            if any(k in err_msg for k in ("connection", "refused", "10061")):
                logger.warning("Ollama connection failed at %s", config.OLLAMA_BASE_URL)
                answer = (
                    "Could not connect to the local Ollama instance. "
                    "Make sure Ollama is running (`ollama serve`).\n\n"
                    "Top retrieved context:\n"
                    f"{context[:600]}..."
                )
            else:
                raise exc

        return {
            "question": question,
            "answer": answer,
            "sources": format_sources(documents),
            "num_chunks_retrieved": len(documents),
        }
