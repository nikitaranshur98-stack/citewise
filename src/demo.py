from pathlib import Path
from src.ingest import ingest_pdf
from src.retriever import add_chunks, get_vector_store

DEMO_DIR = Path("demo_docs")


def ensure_demo_index():
    """Index the demo PDFs if the knowledge base is empty. Returns how many chunks were added."""
    if get_vector_store().get(limit=1)["ids"]:
        return 0
    total = 0
    for pdf in sorted(DEMO_DIR.glob("*.pdf")):
        total += add_chunks(ingest_pdf(pdf))
    return total