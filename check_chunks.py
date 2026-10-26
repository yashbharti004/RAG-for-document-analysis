import os
import pickle
import config

if not os.path.exists(config.CHUNKS_FILE):
    print(f"No chunks file found at: {config.CHUNKS_FILE}")
    print("Run 'python ingest.py' first to process documents.")
    exit(1)

with open(config.CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks loaded: {len(chunks)}\n")

for i, chunk in enumerate(chunks[:8]):
    page = chunk.metadata.get("page", "?")
    source = chunk.metadata.get("source", "unknown")
    section = chunk.metadata.get("section", "N/A")

    print(f"[{i + 1}] Source: {source} (Page {page}) | Section: {section}")
    print(chunk.page_content.strip()[:300] + ("..." if len(chunk.page_content) > 300 else ""))
    print("-" * 60)
