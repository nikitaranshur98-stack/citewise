import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL

import re

from rank_bm25 import BM25Okapi


PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "citewise"
_store = None


def get_vector_store():
    global _store
    if _store is None:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _store
    
    
def prepare(chunks):
    docs, ids = [], []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        docs.append(
            Document(
                page_content=chunk.page_content,
                metadata={"source": source, "page": page},
            )
        )
        ids.append(f"{source}-p{page}-c{i}")
    return docs, ids

def add_chunks(chunks, batch_size=20):
    store = get_vector_store()
    docs, ids = prepare(chunks)
    for start in range(0, len(docs), batch_size):
        store.add_documents(
            docs[start:start + batch_size],
            ids=ids[start:start + batch_size],
        )
        time.sleep(1)
    return len(docs)

def search(query, k=4):
    store = get_vector_store()
    return store.similarity_search_with_score(query, k=k)

_bm25_cache = {"count": -1, "bm25": None, "docs": []}


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def _keyword_index():
    data = get_vector_store().get()
    count = len(data["ids"])
    if _bm25_cache["count"] != count:
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        _bm25_cache["docs"] = docs
        _bm25_cache["bm25"] = BM25Okapi([tokenize(d.page_content) for d in docs])
        _bm25_cache["count"] = count
    return _bm25_cache


def keyword_search(query, k=20):
    index = _keyword_index()
    scores = index["bm25"].get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [index["docs"][i] for i in ranked[:k] if scores[i] > 0]

def rrf_fuse(rankings, k=60):
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered]


def hybrid_search(query, k=5, pool=20):
    vector_docs = [doc for doc, _score in search(query, k=pool)]
    keyword_docs = keyword_search(query, k=pool)
    return rrf_fuse([vector_docs, keyword_docs])[:k]