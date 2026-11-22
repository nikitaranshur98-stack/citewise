from src.tools import document_search

for q in [
    "What is the retriever used in RAG?",
    "How does RAG reduce hallucination?",
    "What is the capital of France?",
]:
    print("Q:", q)
    print(document_search.invoke({"query": q})[:500])
    print("-" * 60)