from src.agent import build_agent, ask

agent = build_agent()

questions = [
    "What retriever does the RAG paper use and how does it work?",
    "What is the capital of France?",
    "What are some recent research papers on agentic RAG?",
]

for q in questions:
    print("=" * 70)
    print("Q:", q)
    answer, tools_used = ask(agent, q)
    print("\nTools called:")
    for name, args in tools_used:
        print(f"  - {name}({args})")
    print("\nAnswer:\n", answer)
    
