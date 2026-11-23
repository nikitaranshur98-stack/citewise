from src.history import init_db, create_chat, add_message, list_chats, get_messages, delete_chat
from src.schema import Briefing
from src.verify import ClaimCheck

init_db()

chat_id = create_chat("What retriever does RAG use?")
add_message(chat_id, "user", "What retriever does RAG use?")

briefing = Briefing(
    question="What retriever does RAG use?",
    summary="RAG uses DPR.",
    claims=[
        {
            "statement": "RAG uses DPR.",
            "confidence": "high",
            "sources": [{"title": "rag_paper.pdf", "location": "rag_paper.pdf, page 3", "source_type": "document"}],
        }
    ],
    limitations="None",
)
checks = [ClaimCheck(claim_number=1, verdict="supported", reason="Stated on page 3.")]
add_message(chat_id, "assistant", briefing.summary, briefing, checks, ["document_search"])

print("Chats:", list_chats())

messages = get_messages(chat_id)
print("Messages stored:", len(messages))
for m in messages:
    print(" -", m["role"], "|", m["content"])

# Rebuild the briefing from the saved JSON to prove the round trip works
restored = Briefing.model_validate_json(messages[1]["briefing_json"])
print("Restored claim:", restored.claims[0].statement)

delete_chat(chat_id)
print("After delete, chats:", len(list_chats()))
print("After delete, messages:", len(get_messages(chat_id)))