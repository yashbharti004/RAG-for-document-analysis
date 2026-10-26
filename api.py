"""
FastAPI application serving the RAG query pipeline and document upload endpoint.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_chain import RAGChain
import config

logger = logging.getLogger("rag.api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on application startup."""
    try:
        app.state.chain = RAGChain(retriever_mode="rerank")
        logger.info("RAG pipeline initialized successfully")
    except FileNotFoundError as err:
        app.state.chain = None
        logger.warning(
            "Vector index not found (%s). Run 'python ingest.py' to generate it.",
            err,
        )
    except Exception as exc:
        app.state.chain = None
        logger.error("Failed to initialize RAG pipeline: %s", exc)
    yield
    app.state.chain = None


app = FastAPI(
    title="University Policy RAG API",
    description="Query university policy documents with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_chain(request: Request) -> RAGChain:
    chain = getattr(request.app.state, "chain", None)
    if chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Run ingest.py first or verify vectorstore directory.",
        )
    return chain


# Request & Response schemas

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=config.TOP_K_FINAL, ge=1, le=20)


class SourceInfo(BaseModel):
    source: str
    page: Union[int, str]
    section: str = ""
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    num_chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    model: str
    retriever_mode: str


class UploadResponse(BaseModel):
    message: str
    filename: str


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    chain = getattr(request.app.state, "chain", None)
    return HealthResponse(
        status="healthy" if chain else "degraded_no_index",
        index_loaded=chain is not None,
        model=config.LLM_MODEL,
        retriever_mode=getattr(chain, "retriever_mode", "none"),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, chain: RAGChain = Depends(get_chain)):
    try:
        result = chain.ask(req.question, top_k=req.top_k)
        return AskResponse(**result)
    except Exception as exc:
        logger.exception("Error processing question: %s", req.question)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    dest_path = Path(config.DATA_DIR) / file.filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved uploaded PDF to %s", dest_path)
        return UploadResponse(
            message=f"File '{file.filename}' uploaded successfully. Run ingest.py to update index.",
            filename=file.filename,
        )
    except Exception as exc:
        logger.error("Failed to save file %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
