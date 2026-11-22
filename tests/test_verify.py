from src.agent import build_agent, run_research
from src.schema import Claim, Source
from src.verify import verify_briefing

agent = build_agent(structured=True)
question = "What retriever does the RAG paper use and how does it work?"

briefing, evidence, tools_used = run_research(agent, question)
print("Tools called:", tools_used)
print("Titles:", {s.title for c in briefing.claims for s in c.sources})

# Trap: add a deliberately FALSE claim. A working checker must catch it.
briefing.claims.append(
    Claim(
        statement="The RAG paper uses a retriever based on GPT-4.",
        confidence="high",
        sources=[Source(title="rag_paper.pdf", location="rag_paper.pdf, page 3", source_type="document")],
    )
)

print("\n--- Verification ---")
for check in verify_briefing(briefing, evidence):
    claim = briefing.claims[check.claim_number - 1]
    print(f"[{check.verdict.upper()}] {claim.statement[:85]}")
    print(f"    {check.reason}\n")