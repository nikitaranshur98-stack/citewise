import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
TEMPERATURE = 0

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

RELEVANCE_MAX_DISTANCE = 0.7


# On Streamlit Community Cloud, copy secrets into the environment (a no-op locally)
try:
    import streamlit as st

    for _key in ("GOOGLE_API_KEY", "MODEL_NAME", "EMBEDDING_MODEL", "DEMO_MODE", "MAX_QUESTIONS_PER_SESSION"):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

