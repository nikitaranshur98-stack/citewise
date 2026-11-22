from src.tools import web_search, arxiv_search, wikipedia_search

print("=== WEB ===")
print(web_search.invoke({"query": "what is retrieval augmented generation"}))

print("\n=== ARXIV ===")
print(arxiv_search.invoke({"query": "retrieval augmented generation"}))

print("\n=== WIKIPEDIA ===")
print(wikipedia_search.invoke({"query": "Transformer (deep learning architecture)"}))