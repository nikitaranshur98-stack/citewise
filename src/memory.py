import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL
from src.history import get_messages, list_chats
from src.retriever import PERSIST_DIR
from src.schema import Briefing

HISTORY_COLLECTION = "citewise_history"
HISTORY_MAX_DISTANCE = 0.7

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=HISTORY_COLLECTION,
            embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=PERSIST_DIR,
        )
    return _store

def index_answer(message_id, chat_id, briefing):
    text = "\n".join(
        [briefing.question, briefing.summary] + [c.statement for c in briefing.claims]
    )
    doc = Document(
        page_content=text,
        metadata={
            "chat_id": chat_id,
            "message_id": message_id,
            "question": briefing.question[:200],
        },
    )
    _get_store().add_documents([doc], ids=[f"msg-{message_id}"])
    

def search_history(query, k=8, max_distance=HISTORY_MAX_DISTANCE):
    results = _get_store().similarity_search_with_score(query, k=k)
    best = {}
    for doc, score in results:
        chat_id = doc.metadata["chat_id"]
        if score <= max_distance and (chat_id not in best or score < best[chat_id]["score"]):
            best[chat_id] = {
                "chat_id": chat_id,
                "question": doc.metadata["question"],
                "score": score,
            }
    return sorted(best.values(), key=lambda r: r["score"])


def delete_chat_vectors(chat_id):
    try:
        _get_store().delete(where={"chat_id": chat_id})
    except Exception:
        pass  # cleanup is best effort and must never block deleting the chat


def backfill():
    count = 0
    for chat in list_chats():
        for m in get_messages(chat["id"]):
            if m["role"] == "assistant" and m["briefing_json"]:
                index_answer(m["id"], chat["id"], Briefing.model_validate_json(m["briefing_json"]))
                count += 1
                time.sleep(1)
    return count

