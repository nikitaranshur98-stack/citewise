from src.ingest import load_pdf, ingest_pdf

PDF = "data/rag_paper.pdf"

pages = load_pdf(PDF)
chunks = ingest_pdf(PDF)

print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

lengths = [len(c.page_content) for c in chunks]
print(f"Chunk length: min={min(lengths)}, max={max(lengths)}, average={sum(lengths)//len(lengths)}")

print("\n--- Sample chunk (chunk #5) ---")
print(chunks[5].page_content)
print("\nMetadata:", chunks[5].metadata)