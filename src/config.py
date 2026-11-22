import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
TEMPERATURE = 0

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

RELEVANCE_MAX_DISTANCE = 0.7
