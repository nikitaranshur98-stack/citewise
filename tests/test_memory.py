from src.memory import backfill, search_history

print("Indexed answers:", backfill())

for q in [
    "how does the retriever encode queries",
    "capital city of a country",
    "best pasta recipe",
]:
    print("\nQuery:", q)
    results = search_history(q, max_distance=10)  # a huge cutoff, so we can see every score
    for r in results:
        print(f"  [{r['score']:.3f}] chat {r['chat_id']}: {r['question']}")