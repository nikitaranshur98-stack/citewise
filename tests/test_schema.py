from pydantic import ValidationError
from src.schema import Briefing

# 1. A valid briefing
good = Briefing(
    question="What is RAG?",
    summary="RAG combines retrieval with generation.",
    claims=[
        {
            "statement": "RAG retrieves documents before generating an answer.",
            "confidence": "high",
            "sources": [
                {"title": "RAG paper", "location": "https://arxiv.org/abs/2005.11401", "source_type": "arxiv"}
            ],
        }
    ],
    limitations="None",
)
print("Valid briefing created:")
print(good.model_dump_json(indent=2))

# 2. An invalid briefing (bad confidence value)
try:
    Briefing(
        question="x",
        summary="x",
        claims=[{"statement": "x", "confidence": "pretty sure", "sources": []}],
        limitations="None",
    )
except ValidationError as e:
    print("\nCaught invalid data, as expected:")
    print(e)