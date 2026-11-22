import requests
import arxiv
#import wikipedia
from ddgs import DDGS
from langchain_core.tools import tool
@tool
def web_search(query: str) -> str:
    """Search the live web for current events, recent developments, or general
    facts that may not be in the uploaded documents. Returns titles, URLs and snippets."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "No web results found."

    blocks = []
    for r in results:
        blocks.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}")
    return "\n\n".join(blocks)

@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic research papers. Use this for scientific or
    technical questions about machine learning, AI, physics, math or computer science.
    Pass a short keyword phrase (2 to 5 words), not a full sentence."""
    client = arxiv.Client()
    papers = []

    try:
        # Try an exact-phrase search first, then fall back to a loose search
        for q in (f'all:"{query}"', query):
            search = arxiv.Search(
                query=q,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = list(client.results(search))
            if papers:
                break
    except Exception as e:
        return f"arXiv search failed: {e}"

    if not papers:
        return "No arXiv papers found."

    blocks = []
    for p in papers:
        blocks.append(
            f"Title: {p.title}\nURL: {p.entry_id}\n"
            f"Published: {p.published.date()}\nAbstract: {p.summary[:600]}"
        )
    return "\n\n".join(blocks)


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "CiteWise/1.0 (learning project; nikitaranshur98@gmail.com)"}


@tool
def wikipedia_search(query: str) -> str:
    """Look up background information, definitions and general knowledge on
    Wikipedia. Use for well-established concepts, people, places and history."""
    try:
        # Step 1: find the best matching page title
        search = requests.get(
            WIKI_API,
            headers=WIKI_HEADERS,
            timeout=15,
            params={"action": "query", "list": "search", "srsearch": query,
                    "srlimit": 1, "format": "json"},
        )
        search.raise_for_status()
        hits = search.json()["query"]["search"]
        if not hits:
            return "No Wikipedia results found."
        title = hits[0]["title"]

        # Step 2: fetch that page's intro text and URL
        page = requests.get(
            WIKI_API,
            headers=WIKI_HEADERS,
            timeout=15,
            params={"action": "query", "prop": "extracts|info", "exintro": 1,
                    "explaintext": 1, "inprop": "url", "titles": title,
                    "redirects": 1, "format": "json"},
        )
        page.raise_for_status()
        data = next(iter(page.json()["query"]["pages"].values()))
        return f"Title: {data['title']}\nURL: {data['fullurl']}\nSummary: {data['extract'][:800]}"
    except Exception as e:
        return f"Wikipedia search failed: {e}"
    

from src.config import RELEVANCE_MAX_DISTANCE
from src.retriever import search as vector_search

@tool
def document_search(query: str) -> str:
    """Search the user's uploaded documents. Use this FIRST for any question that
    could be answered by the uploaded files. Returns passages with filename and
    page number. If it reports no relevant passages, use the web, arXiv or
    Wikipedia tools instead."""
    try:
        results = vector_search(query, k=4)
    except Exception as e:
        return f"Document search failed: {e}"

    relevant = [(doc, score) for doc, score in results if score <= RELEVANCE_MAX_DISTANCE]
    if not relevant:
        return "No relevant passages found in the uploaded documents."

    blocks = []
    for doc, score in relevant:
        blocks.append(
            f"Source: {doc.metadata['source']}, page {doc.metadata['page']}\n"
            f"Passage: {doc.page_content}"
        )
    return "\n\n".join(blocks)