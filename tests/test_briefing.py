from src.agent import build_agent, ask_briefing

agent = build_agent(structured=True)

for q in [
    "What retriever does the RAG paper use and how does it work?",
    "What are some recent research papers on agentic RAG?",
]:
    print("=" * 70)
    print("Q:", q)
    briefing, tools_used = ask_briefing(agent, q)

    print("\nTools called:", [name for name, _ in tools_used])
    print("\nType:", type(briefing).__name__)
    print(briefing.model_dump_json(indent=2))







