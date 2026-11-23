from src.agent import condense_question

history = """User: What retriever does the RAG paper use?
Assistant: The RAG paper uses DPR as its retriever. Claims: 1. The RAG paper uses Dense Passage Retriever (DPR) as its retrieval component. 2. DPR uses a bi-encoder architecture with BERT-base document and query encoders."""

for q in [
    "Explain the second claim in more detail",
    "What about the generator?",
    "What is the capital of France?",
]:
    print("Follow-up:", q)
    print("Rewritten:", condense_question(history, q))
    print()