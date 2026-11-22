import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL

emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "The cat sat on the mat",
]

vectors = emb.embed_documents(sentences)
print("Vector length:", len(vectors[0]))
print("First 5 numbers:", [round(x, 4) for x in vectors[0][:5]])


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\nPassword vs Login :", round(cosine(vectors[0], vectors[1]), 3))
print("Password vs Cat   :", round(cosine(vectors[0], vectors[2]), 3))