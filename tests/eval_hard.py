import time

from src.retriever import search
import re

# (question, [phrases; a retrieved chunk is correct if it contains any of them])
HARD_SET = [
    ("How many passages are in the Wikipedia index?", ["21m documents"]),
    ("What extra approximation makes RAG-Sequence decoding cheaper?", ["further approximation"]),
    ("How many world leaders were used to test index swapping?", ["82 world leaders"]),
    ("What happened when the index and the leaders came from different years?", ["mismatched"]),
    ("How many trainable parameters does the full RAG model have?", ["626m"]),
    ("When does BM25 retrieval beat the learned retriever?", ["bm25 performs best"]),
    ("What GPUs was RAG trained on?", ["v100"]),
    ("What is retrieval collapse?", ["collapse"]),
    ("Did a null document mechanism help?", ["null document"]),
    ("How were the CuratedTrec answers preprocessed?", ["regular expressions"]),
    ("Which model is better for Jeopardy question generation?", ["performs better than rag-sequence"]),
    ("Why were some human annotators removed from the results?", ["two annotators"]),
]

K = 5


def normalize(text):
    # lowercase and drop spaces, hyphens and punctuation, so PDF extraction quirks cannot break matching
    return re.sub(r"\W+", "", text.lower())

def evaluate_hard(search_fn, k=K):
    hit1 = hit3 = hit_k = 0
    reciprocal_ranks = []
    misses = []

    for question, phrases in HARD_SET:
        docs = search_fn(question, k)
        rank = next(
            (
                i
                for i, d in enumerate(docs, start=1)
                if any(normalize(p) in normalize(d.page_content) for p in phrases)            ),
            None,
        )
        if rank is None:
            misses.append((question, phrases))
            reciprocal_ranks.append(0)
            continue

        reciprocal_ranks.append(1 / rank)
        hit1 += rank <= 1
        hit3 += rank <= 3
        hit_k += rank <= k

    n = len(HARD_SET)
    print(f"Questions: {n}")
    print(f"Hit@1: {hit1 / n:.0%}")
    print(f"Hit@3: {hit3 / n:.0%}")
    print(f"Hit@{k}: {hit_k / n:.0%}")
    print(f"MRR:   {sum(reciprocal_ranks) / n:.3f}")

    if misses:
        print(f"\nMissed ({len(misses)}):")
        for question, phrases in misses:
            print(f"  - {question}  (looking for {phrases})")


def vector_only(question, k):
    time.sleep(1)
    return [doc for doc, _score in search(question, k=k)]


if __name__ == "__main__":
    print("=== HARD SET BASELINE: vector search only ===")
    evaluate_hard(vector_only)
    
