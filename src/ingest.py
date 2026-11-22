from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_pdf(path):
    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        page.metadata["source"] = Path(path).name
        page.metadata["page"] = page.metadata.get("page", 0) + 1
    return pages

def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)

def ingest_pdf(path):
    pages = load_pdf(path)
    return split_documents(pages)