import time

from src.retriever import hybrid_search
from tests.eval_hard import evaluate_hard, vector_only as hard_vector
from tests.eval_retrieval import evaluate, vector_only as easy_vector


def hybrid(question, k):
    time.sleep(1)
    return hybrid_search(question, k=k)


print("##### HARD SET: vector only #####")
evaluate_hard(hard_vector)

print("\n##### HARD SET: hybrid #####")
evaluate_hard(hybrid)

print("\n##### EASY SET: vector only #####")
evaluate(easy_vector)

print("\n##### EASY SET: hybrid #####")
evaluate(hybrid)