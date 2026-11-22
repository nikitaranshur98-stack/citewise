from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import MODEL_NAME, TEMPERATURE
from src.retry import call_with_retry
from src.schema import Briefing
from src.tools import document_search, web_search, arxiv_search, wikipedia_search

TOOLS = [document_search, web_search, arxiv_search, wikipedia_search]
TOOL_NAMES = {t.name for t in TOOLS}

SYSTEM_PROMPT = """You are CiteWise, a careful research assistant.

Follow these rules:
1. Your FIRST tool call must ALWAYS be document_search, even if the question seems to need the web or research papers. Only after it returns may you call any other tool.
2. If it reports no relevant passages, or the passages do not fully answer the question, use the other tools: arxiv_search for research and science topics, wikipedia_search for background knowledge, web_search for recent or current information.
3. Answer ONLY with information returned by the tools. Never use facts from your own memory.
4. For every fact, state where it came from: the file name and page, or the URL.
5. If the tools do not contain the answer, say so clearly instead of guessing.
6. Keep search queries short: 2 to 5 keywords, not full sentences."""

BRIEFING_RULES = """

When you give your final answer, fill in the Briefing structure:
- summary: 2 to 3 sentences.
- claims: - Every fact stated in the summary must also appear as its own claim. The summary may not contain facts that are missing from the claims.
- one entry per distinct fact, each with at least one source.
- source_type must be "document", "web", "arxiv" or "wikipedia", matching the tool that returned the information.
- For documents, location is "filename, page N". For all other sources, location is the URL.
- For documents, title is the file name exactly as shown in the tool result. Never write a title, author or date that the tools did not return.
- confidence is "high" only when a source states the claim directly, "medium" when it is partly stated, and "low" when it is only implied.
- limitations: say what you could not find or verify, or "None"."""


def build_agent(structured=False):
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        timeout=60,
        max_retries=2,
    )
    prompt = SYSTEM_PROMPT + (BRIEFING_RULES if structured else "")
    options = {"response_format": ToolStrategy(Briefing)} if structured else {}
    return create_agent(model=llm, tools=TOOLS, system_prompt=prompt, **options)


def invoke_with_retry(agent, question):
    payload = {"messages": [{"role": "user", "content": question}]}
    return call_with_retry(lambda: agent.invoke(payload))


def ask(agent, question):
    result = invoke_with_retry(agent, question)
    messages = result["messages"]

    tools_used = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            tools_used.append((call["name"], call["args"]))

    return messages[-1].text, tools_used


def ask_briefing(agent, question):
    result = invoke_with_retry(agent, question)

    tools_used = []
    for m in result["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            if call["name"] != "Briefing":
                tools_used.append((call["name"], call["args"]))

    return result["structured_response"], tools_used


def collect_evidence(messages):
    return [
        m.content
        for m in messages
        if getattr(m, "type", "") == "tool"
        and getattr(m, "name", "") in TOOL_NAMES
        and isinstance(m.content, str)
    ]


def fix_titles(briefing):
    for claim in briefing.claims:
        for source in claim.sources:
            if source.source_type == "document":
                source.title = source.location.split(",")[0].strip()
    return briefing


def run_research(agent, question):
    result = invoke_with_retry(agent, question)
    messages = result["messages"]

    tools_used = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            if call["name"] != "Briefing":
                tools_used.append(call["name"])

    briefing = fix_titles(result["structured_response"])
    return briefing, collect_evidence(messages), tools_used


def format_history(messages, max_turns=3):
    lines = []
    for m in messages[-2 * max_turns:]:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
            continue

        text = m["content"]
        if m.get("briefing_json"):
            try:
                briefing = Briefing.model_validate_json(m["briefing_json"])
                claims = " ".join(
                    f"{i}. {c.statement}" for i, c in enumerate(briefing.claims, start=1)
                )
                text = f"{briefing.summary} Claims: {claims}"
            except Exception:
                pass
        lines.append(f"Assistant: {text[:700]}")
    return "\n".join(lines)

CONDENSE_PROMPT = """You rewrite follow-up questions.

Given the conversation below and a new question, rewrite the new question so it can be understood on its own, without the conversation.
- Replace references such as "it", "that", "the second claim" or "the paper" with the specific thing they refer to.
- Keep all names, numbers and terms exactly as written.
- If the new question already stands on its own, return it unchanged.
- Do NOT answer the question. Output only the rewritten question.

CONVERSATION:
{history}

NEW QUESTION: {question}

STANDALONE QUESTION:"""


def condense_question(history, question):
    if not history.strip():
        return question
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=0, timeout=60, max_retries=2
    )
    prompt = CONDENSE_PROMPT.format(history=history, question=question)
    try:
        reply = call_with_retry(lambda: llm.invoke(prompt))
        return reply.text.strip() or question
    except Exception:
        return question  # if rewriting fails, fall back to the question as typed
    
    