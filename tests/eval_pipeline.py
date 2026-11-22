import time

from src.agent import build_agent, run_research
from src.verify import verify_briefing

# (question, "document" if the paper answers it, "outside" if it must go beyond the paper)
PIPELINE_SET = [
    ("What retriever does the RAG paper use?", "document"),
    ("How many world leaders were used to test index hot-swapping?", "document"),
    ("What generator model does RAG use and how large is it?", "document"),
    ("What ethical concerns does the paper raise?", "document"),
    ("What is the capital of France?", "outside"),
    ("What are some recent research papers on agentic RAG?", "outside"),
]

PAUSE_SECONDS = 8


def main():
    agent = build_agent(structured=True)
    totals = {"supported": 0, "partial": 0, "unsupported": 0}
    routing_ok = 0
    ran = 0
    seconds = []
    problems = []

    for question, kind in PIPELINE_SET:
        print(f"\nQ: {question}")
        start = time.time()
        try:
            briefing, evidence, tools_used = run_research(agent, question)
            checks = verify_briefing(briefing, evidence)
        except Exception as e:
            print(f"  FAILED: {str(e)[:150]}")
            problems.append((question, "run failed", str(e)[:150]))
            time.sleep(PAUSE_SECONDS)
            continue

        elapsed = time.time() - start
        seconds.append(elapsed)
        ran += 1

        others = [t for t in tools_used if t != "document_search"]
        starts_right = bool(tools_used) and tools_used[0] == "document_search"
        routed = starts_right and (kind == "document" or bool(others))
        routing_ok += routed

        print(f"  tools: {tools_used}")
        print(f"  routing correct: {routed}   time: {elapsed:.0f}s   claims: {len(briefing.claims)}")

        for check in checks:
            totals[check.verdict] += 1
            if check.verdict != "supported" and 1 <= check.claim_number <= len(briefing.claims):
                statement = briefing.claims[check.claim_number - 1].statement
                problems.append((question, check.verdict, f"{statement[:90]} -> {check.reason}"))

        time.sleep(PAUSE_SECONDS)

    total_claims = sum(totals.values())
    print("\n" + "=" * 60)
    print(f"Questions completed: {ran} of {len(PIPELINE_SET)}")
    if ran:
        print(f"Routing accuracy:    {routing_ok}/{ran}")
        print(f"Average time:        {sum(seconds) / len(seconds):.0f}s per question")
    if total_claims:
        print(f"Claims checked:      {total_claims}")
        for verdict, count in totals.items():
            print(f"  {verdict:<12} {count}  ({count / total_claims:.0%})")
    if problems:
        print("\nNot fully supported or failed:")
        for question, verdict, detail in problems:
            print(f"  [{verdict}] {question}\n      {detail}")


if __name__ == "__main__":
    main()