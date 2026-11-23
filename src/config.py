import os

from dotenv import load_dotenv

load_dotenv()

# On Streamlit Community Cloud there is no .env file, so copy secrets into the
# environment first. Locally this does nothing, because st.secrets raises an error.
try:
    import streamlit as st

    for _key in ("GOOGLE_API_KEY", "MODEL_NAME", "EMBEDDING_MODEL", "DEMO_MODE", "MAX_QUESTIONS_PER_SESSION"):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
TEMPERATURE = 0

RELEVANCE_MAX_DISTANCE = 0.7

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "5"))