from pathlib import Path

import streamlit as st

from src.agent import build_agent, run_research
from src.ingest import ingest_pdf
from src.retriever import add_chunks
from src.verify import verify_briefing

st.set_page_config(page_title="CiteWise", page_icon="📚", layout="wide")
st.title("📚 CiteWise")
st.caption("A research assistant that cites its sources and checks its own claims.")


@st.cache_resource
def get_agent():
    return build_agent(structured=True)


# ---------- Sidebar: upload ----------
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


# ---------- Display ----------
VERDICT_ICON = {"supported": "✅", "partial": "⚠️", "unsupported": "❌"}


def render(briefing, checks, tools_used):
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


# ---------- Main: question and results ----------
question = st.text_input("Ask a research question")

if st.button("Research") and question.strip():
    try:
        with st.spinner("Researching your documents and the web..."):
            briefing, evidence, tools_used = run_research(get_agent(), question)
        with st.spinner("Verifying each claim against its sources..."):
            checks = verify_briefing(briefing, evidence)
        st.session_state["result"] = (briefing, checks, tools_used)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

if "result" in st.session_state:
    render(*st.session_state["result"])