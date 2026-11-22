import time

from src.retriever import search

# (question, [pages whose text states the answer]); page 1 = first page of the PDF
EVAL_SET = [
    ("What retriever does RAG use?", [2, 3]),
    ("What is Maximum Inner Product Search used for?", [2, 3, 4]),
    ("What generator model does RAG use?", [2, 3]),
    ("What is the difference between RAG-Sequence and RAG-Token?", [1, 2, 3]),
    ("How are the retriever and generator trained?", [3, 4]),
    ("What decoding method does RAG-Token use?", [4]),
    ("Which datasets were used for open-domain question answering?", [2, 4]),
    ("How did RAG perform on Natural Questions compared to DPR?", [5, 6]),
    ("What index of documents is used as the knowledge source?", [1, 2, 4]),
    ("How does RAG help avoid hallucination?", [10]),
    ("What are the ethical concerns the paper mentions?", [10]),
    ("How can the knowledge base be updated without retraining?", [2, 7, 8, 9]),
]

K = 5


def evaluate(search_fn, k=K):
    hit1 = hit3 = hit_k = 0
    reciprocal_ranks = []
    misses = []

    for question, gold_pages in EVAL_SET:
        docs = search_fn(question, k)
        pages = [d.metadata["page"] for d in docs]

        rank = next((i for i, p in enumerate(pages, start=1) if p in gold_pages), None)
        if rank is None:
            misses.append((question, gold_pages, pages))
            reciprocal_ranks.append(0)
            continue

        reciprocal_ranks.append(1 / rank)
        hit1 += rank <= 1
        hit3 += rank <= 3
        hit_k += rank <= k

    n = len(EVAL_SET)
    print(f"Questions: {n}")
    print(f"Hit@1: {hit1 / n:.0%}")
    print(f"Hit@3: {hit3 / n:.0%}")
    print(f"Hit@{k}: {hit_k / n:.0%}")
    print(f"MRR:   {sum(reciprocal_ranks) / n:.3f}")

    if misses:
        print(f"\nMissed ({len(misses)}):")
        for question, gold, found in misses:
            print(f"  - {question}\n    expected pages {gold}, got {found}")


def vector_only(question, k):
    time.sleep(1)  # stay under the free-tier rate limit
    return [doc for doc, _score in search(question, k=k)]


if __name__ == "__main__":
    print("=== BASELINE: vector search only ===")
    evaluate(vector_only)