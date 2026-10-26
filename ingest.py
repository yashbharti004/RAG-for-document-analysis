import os
import re
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract page-level text and metadata from a PDF file."""
    doc = fitz.open(pdf_path)
    source_name = Path(pdf_path).stem
    pages: List[Dict[str, Any]] = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")

        # Scanned pages or vector graphics often yield little to no text;
        # fall back to OCR if PyTesseract is installed.
        if len(text.strip()) < 50:
            text = _ocr_page(page, source_name, page_num)

        pages.append({
            "text": text,
            "metadata": {
                "source": source_name,
                "page": page_num + 1,
                "file_path": pdf_path,
            },
        })

    doc.close()
    return pages


def _ocr_page(page, source_name: str, page_num: int) -> str:
    """Attempt OCR on a page when standard text extraction yields minimal text."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        logger.info(f"Ran OCR on {source_name} (page {page_num + 1})")
        return text
    except ImportError:
        # Tesseract / PIL not available, stick with whatever pymupdf returned
        return page.get_text("text")
    except Exception as exc:
        logger.warning(f"OCR failed for {source_name} page {page_num + 1}: {exc}")
        return page.get_text("text")


def clean_text(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove repeating running headers/footers and isolated page numbers."""
    if len(pages) < 3:
        return pages

    first_lines = []
    last_lines = []
    for p in pages:
        lines = p["text"].strip().split("\n")
        if lines:
            first_lines.append(lines[0].strip())
            last_lines.append(lines[-1].strip())

    threshold = len(pages) * 0.5
    repeating_headers = {
        line for line, cnt in Counter(first_lines).items()
        if cnt > threshold and len(line) > 2
    }
    repeating_footers = {
        line for line, cnt in Counter(last_lines).items()
        if cnt > threshold and len(line) > 2
    }

    cleaned: List[Dict[str, Any]] = []
    for page in pages:
        lines = page["text"].split("\n")

        if lines and lines[0].strip() in repeating_headers:
            lines = lines[1:]
        if lines and lines[-1].strip() in repeating_footers:
            lines = lines[:-1]

        # Strip standalone page numbering patterns
        lines = [
            l for l in lines
            if not re.match(r"^\s*[-—]?\s*\d+\s*[-—]?\s*$", l)
            and not re.match(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", l, re.IGNORECASE)
        ]

        text = "\n".join(lines).strip()
        page["metadata"]["section"] = _detect_section(text)
        page["text"] = text
        cleaned.append(page)

    return cleaned


def _detect_section(text: str) -> str:
    """Detect prominent heading or section titles near top of page."""
    for line in text.split("\n")[:8]:
        line = line.strip()
        if not line or len(line) < 4:
            continue
        if re.match(r"^(\d+|[ivxlcdm]+)$", line, re.IGNORECASE):
            continue
        if re.match(r"^(Chapter|Section|Article|Part)\s+\d+", line, re.IGNORECASE):
            return line
        if re.match(r"^\d+(\.\d+)*\s+[A-Za-z]", line):
            return line
        if line.isupper() and 5 < len(line) < 80:
            return line
    return ""


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Document]:
    """Split page text into chunks while preserving metadata and headings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: List[Document] = []
    current_section = ""

    for page in pages:
        section = page["metadata"].get("section", "")
        if section:
            current_section = section

        text = page["text"].strip()
        if not text:
            continue

        # If page is a section divider with minimal text, keep it as context heading
        non_empty_lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(text) < 80 and len(non_empty_lines) <= 2:
            current_section = " ".join(non_empty_lines)
            continue

        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            clean_chunk = chunk.strip()
            if len(clean_chunk) < 60 and len(chunks) > 1:
                continue

            meta = {
                **page["metadata"],
                "chunk_index": idx,
                "section": current_section or (non_empty_lines[0] if non_empty_lines else ""),
            }
            documents.append(Document(page_content=clean_chunk, metadata=meta))

    return documents


def build_vectorstore(documents: List[Document]) -> FAISS:
    """Create a FAISS vector index from documents."""
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"Indexing {len(documents)} chunks with FAISS...")
    return FAISS.from_documents(documents, embeddings)


def ingest(data_dir: str | None = None) -> None:
    """Read PDFs from data directory, chunk, embed, and persist index."""
    data_dir = data_dir or config.DATA_DIR
    pdf_files = sorted(Path(data_dir).glob("**/*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {data_dir}")

    all_pages: List[Dict[str, Any]] = []
    for pdf in pdf_files:
        logger.info(f"Processing {pdf.name}...")
        pages = extract_text_from_pdf(str(pdf))
        pages = clean_text(pages)
        all_pages.extend(pages)

    documents = chunk_pages(all_pages)
    logger.info(f"Generated {len(documents)} total chunks")

    vectorstore = build_vectorstore(documents)

    os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(config.VECTORSTORE_DIR)
    logger.info(f"Saved FAISS index to {config.VECTORSTORE_DIR}")

    with open(config.CHUNKS_FILE, "wb") as fh:
        pickle.dump(documents, fh)
    logger.info(f"Saved chunk records to {config.CHUNKS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into vector index")
    parser.add_argument("--data-dir", default=None, help="Directory containing input PDFs")
    args = parser.parse_args()
    ingest(data_dir=args.data_dir)
