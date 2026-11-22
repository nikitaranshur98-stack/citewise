import json
from pathlib import Path

import streamlit as st

from src.agent import build_agent, condense_question, format_history, run_research
from src.config import DEMO_MODE, MAX_QUESTIONS_PER_SESSION
from src.demo import ensure_demo_index
from src.history import add_message, create_chat, delete_chat, get_messages, init_db, list_chats
from src.ingest import ingest_pdf
from src.memory import delete_chat_vectors, index_answer, search_history
from src.retriever import add_chunks
from src.schema import Briefing
from src.verify import ClaimCheck, verify_briefing

st.set_page_config(page_title="CiteWise", page_icon="📚", layout="wide")
st.title("📚 CiteWise")
st.caption("A research assistant that cites its sources and checks its own claims.")

if not DEMO_MODE:
    init_db()


@st.cache_resource
def get_agent():
    return build_agent(structured=True)


@st.cache_resource
def prepare_demo():
    return ensure_demo_index()


if DEMO_MODE:
    try:
        with st.spinner("Preparing the demo knowledge base (first start only)..."):
            prepare_demo()
    except Exception as e:
        st.error(f"Could not prepare the demo knowledge base: {e}")
        st.stop()


# ---------- Display helpers ----------
VERDICT_ICON = {"supported": "✅", "partial": "⚠️", "unsupported": "❌"}


def render(briefing, checks, tools_used):
    st.caption(f"Question answered: {briefing.question}")
    verdicts = {c.claim_number: c for c in checks}

    st.subheader("Summary")
    st.write(briefing.summary)

    st.subheader("Claims")
    for i, claim in enumerate(briefing.claims, start=1):
        check = verdicts.get(i)
        icon = VERDICT_ICON.get(check.verdict, "❔") if check else "❔"
        st.markdown(f"**{icon} {claim.statement}**")
        if check:
            st.caption(
                f"Verification: {check.verdict}. {check.reason} "
                f"(the model's own confidence: {claim.confidence})"
            )
        for source in claim.sources:
            if source.source_type == "document":
                st.write(f"📄 {source.location}")
            else:
                st.write(f"🔗 [{source.title}]({source.location})")

    st.subheader("Limitations")
    st.write(briefing.limitations)
    st.caption("Tools used: " + (", ".join(tools_used) if tools_used else "none"))


def render_message(m):
    if m["role"] == "user":
        with st.chat_message("user"):
            st.write(m["content"])
        return

    with st.chat_message("assistant"):
        if m["briefing_json"]:
            briefing = Briefing.model_validate_json(m["briefing_json"])
            checks = [ClaimCheck(**c) for c in json.loads(m["checks_json"] or "[]")]
            tools = json.loads(m["tools_json"] or "[]")
            render(briefing, checks, tools)
        else:
            st.write(m["content"])


# ---------- Storage helpers (session-only in demo mode, SQLite otherwise) ----------
def load_messages():
    if DEMO_MODE:
        return st.session_state.setdefault("demo_messages", [])
    chat_id = st.session_state.get("chat_id")
    return get_messages(chat_id) if chat_id else []


def save_exchange(question, briefing, checks, tools_used):
    if DEMO_MODE:
        messages = st.session_state.setdefault("demo_messages", [])
        messages.append(
            {"role": "user", "content": question,
             "briefing_json": None, "checks_json": None, "tools_json": None}
        )
        messages.append(
            {"role": "assistant", "content": briefing.summary,
             "briefing_json": briefing.model_dump_json(),
             "checks_json": json.dumps([c.model_dump() for c in checks]),
             "tools_json": json.dumps(tools_used)}
        )
        return

    chat_id = st.session_state.get("chat_id")
    if chat_id is None:
        chat_id = create_chat(question)
        st.session_state["chat_id"] = chat_id
    add_message(chat_id, "user", question)
    message_id = add_message(chat_id, "assistant", briefing.summary, briefing, checks, tools_used)
    try:
        index_answer(message_id, chat_id, briefing)
    except Exception:
        pass  # saving the answer matters more than indexing it


# ---------- Sidebar ----------
if DEMO_MODE:
    used = st.session_state.get("questions_asked", 0)
    st.sidebar.header("Public demo")
    st.sidebar.info(
        "This demo answers questions about one paper (the original RAG paper) and the open web. "
        "The full version, with PDF upload and saved chats, runs locally. See the GitHub repository."
    )
    st.sidebar.markdown(
        "**Try these:**\n"
        "- What retriever does the RAG paper use?\n"
        "- What ethical concerns does the paper raise?\n"
        "- What is the capital of France?\n"
        "- What are some recent papers on agentic RAG?"
    )
    st.sidebar.caption(f"Questions used: {used} of {MAX_QUESTIONS_PER_SESSION}")
    if st.sidebar.button("🔄 Start over"):
        st.session_state["demo_messages"] = []
        st.rerun()
else:
    st.sidebar.header("Chats")

    search_text = st.sidebar.text_input("🔎 Search past research")
    if search_text.strip():
        found = search_history(search_text)
        if not found:
            st.sidebar.caption("No matching past research.")
        for r in found:
            if st.sidebar.button(f"{r['question'][:45]}", key=f"found_{r['chat_id']}"):
                st.session_state["chat_id"] = r["chat_id"]
                st.rerun()
        st.sidebar.divider()

    if st.sidebar.button("➕ New chat"):
        st.session_state["chat_id"] = None

    for chat in list_chats():
        col_open, col_delete = st.sidebar.columns([5, 1])
        is_current = chat["id"] == st.session_state.get("chat_id")

        if col_open.button(
            chat["title"],
            key=f"open_{chat['id']}",
            type="primary" if is_current else "secondary",
        ):
            st.session_state["chat_id"] = chat["id"]
            st.rerun()

        if col_delete.button("🗑", key=f"delete_{chat['id']}"):
            delete_chat(chat["id"])
            delete_chat_vectors(chat["id"])
            if is_current:
                st.session_state["chat_id"] = None
            st.rerun()

    st.sidebar.divider()
    Path("data").mkdir(exist_ok=True)

    st.sidebar.header("Your documents")
    uploaded = st.sidebar.file_uploader("Upload a PDF", type="pdf")

    if uploaded and st.sidebar.button("Add to knowledge base"):
        path = Path("data") / uploaded.name
        path.write_bytes(uploaded.getbuffer())
        try:
            with st.spinner("Reading and indexing your document..."):
                count = add_chunks(ingest_pdf(path))
            st.sidebar.success(f"Added {count} chunks from {uploaded.name}")
        except Exception as e:
            st.sidebar.error(f"Could not index the file: {e}")

# ---------- Main: the conversation ----------
messages = load_messages()

for m in messages:
    render_message(m)

if not messages:
    if DEMO_MODE:
        st.info("Ask a question below, or try one of the examples in the sidebar.")
    else:
        st.info("Ask a question below to start a new chat, or open a saved chat from the sidebar.")

limit_reached = DEMO_MODE and st.session_state.get("questions_asked", 0) >= MAX_QUESTIONS_PER_SESSION
if limit_reached:
    st.warning(
        "You have used all the questions for this demo session. "
        "Run the project locally (see the GitHub repository) to ask more."
    )

question = st.chat_input("Ask a research question", disabled=limit_reached)

if question:
    if DEMO_MODE:
        st.session_state["questions_asked"] = st.session_state.get("questions_asked", 0) + 1

    with st.chat_message("user"):
        st.write(question)

    try:
        with st.chat_message("assistant"):
            history = format_history(messages)
            with st.spinner("Researching your documents and the web..."):
                standalone = condense_question(history, question)
                briefing, evidence, tools_used = run_research(get_agent(), standalone)
            with st.spinner("Verifying each claim against its sources..."):
                checks = verify_briefing(briefing, evidence)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    else:
        save_exchange(question, briefing, checks, tools_used)
        st.rerun()
        
        