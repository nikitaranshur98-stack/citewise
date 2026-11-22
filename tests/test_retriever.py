from src.ingest import ingest_pdf
from src.retriever import add_chunks, search

chunks = ingest_pdf("data/rag_paper.pdf")
n = add_chunks(chunks)
print(f"Stored {n} chunks\n")

questions = [
    "What is the retriever used in RAG?",
    "How does RAG reduce hallucination?",
    "What is the capital of France?",  # off-topic control
]

for q in questions:
    print("Q:", q)
    for doc, score in search(q, k=3):
        preview = doc.page_content[:110].replace("\n", " ")
        print(f"  [{score:.3f}] p.{doc.metadata['page']}: {preview}")
    print()
    
