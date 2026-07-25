"""
research_agent.py

A minimal research agent that talks to Claude Haiku 4.5 through OpenRouter's
OpenAI-compatible function-calling API.

-----------------------------------------------------------------------------
Agentic AI concepts used in this file
-----------------------------------------------------------------------------
SKILLS
    A "skill" is a capability the agent can invoke to affect or query the
    outside world, exposed to the LLM as a "tool" (a JSON function schema).
    The model itself never runs the code — it only outputs a request like
    "call web_search(query='capital of France')", and our code executes the
    matching Python function and feeds the result back in. In this file the
    skills are: web_search, remember, recall, and read_file.

MEMORY
    Memory is anywhere the agent stores information across steps of a
    conversation so it doesn't have to look things up again. Real systems
    often use a vector database; the underlying idea is the same as what we
    do here: a plain Python list (`memory`) that facts get appended to via
    the `remember` skill and read back via the `recall` skill.

HOOKS
    A hook is code that automatically runs around an agent's actions,
    independent of which specific tool was called — used for cross-cutting
    concerns like logging, auditing, or rate limiting. `log_tool_call` below
    is a hook: every tool call passes through it before it executes, and it
    appends a timestamped line to agent_logs.txt.
-----------------------------------------------------------------------------
"""

import os
import json
import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup: load OPENROUTER_API_KEY from .env
# ---------------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found — check your .env file")

MODEL = "anthropic/claude-haiku-4.5"  # Claude Haiku 4.5 via OpenRouter
LOG_FILE = Path(__file__).resolve().parent.parent / "agent_logs.txt"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ---------------------------------------------------------------------------
# MEMORY: facts learned during this session live in a plain Python list.
# ---------------------------------------------------------------------------
memory: list[str] = []


def remember(fact: str) -> str:
    memory.append(fact)
    return f"Saved to memory: {fact}"


def recall() -> str:
    if not memory:
        return "Memory is empty — nothing saved yet."
    return "\n".join(f"- {fact}" for fact in memory)


# ---------------------------------------------------------------------------
# HOOK: fires on every tool call, before it runs, to log it.
# ---------------------------------------------------------------------------
def log_tool_call(tool_name: str, arguments: dict) -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] TOOL CALL: {tool_name}({json.dumps(arguments)})\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# SKILL: web search via the DuckDuckGo Instant Answer API
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    """
    Queries https://api.duckduckgo.com. This is DuckDuckGo's free Instant
    Answer API (no key required) — it returns summary/infobox-style
    knowledge rather than a full list of ranked web pages, which is enough
    for factual lookups in this demo.
    """
    response = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    parts = []
    if data.get("AbstractText"):
        parts.append(data["AbstractText"])
    if data.get("Answer"):
        parts.append(data["Answer"])
    if data.get("Definition"):
        parts.append(data["Definition"])
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            parts.append(topic["Text"])
        if len(parts) >= 5:
            break

    if not parts:
        return f"No results found for '{query}'."
    return "\n".join(parts[:5])


# ---------------------------------------------------------------------------
# SKILL (plugin): read local .txt and .pdf files
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f"File not found: {path}"

    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text or "(PDF contained no extractable text)"

    return f"Unsupported file type: {suffix} (only .txt and .pdf are supported)"


# ---------------------------------------------------------------------------
# Tool schemas — how the skills are described to the model for function
# calling. The model reads these to decide *when* and *how* to call them.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (via DuckDuckGo) for facts, definitions, or general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a fact learned during this session to memory so it can be reused later without searching again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Retrieve all facts saved to memory earlier in this session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a local .txt or .pdf file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the .txt or .pdf file"},
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_IMPLS = {
    "web_search": lambda args: web_search(args["query"]),
    "remember": lambda args: remember(args["fact"]),
    "recall": lambda args: recall(),
    "read_file": lambda args: read_file(args["path"]),
}


def call_tool(name: str, arguments: dict) -> str:
    log_tool_call(name, arguments)  # hook: runs for every tool, before execution
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"Unknown tool: {name}"
    try:
        return str(impl(arguments))
    except Exception as e:
        return f"Error running {name}: {e}"


# ---------------------------------------------------------------------------
# Agent loop: the model decides which skill(s) to call, we execute them and
# feed results back, until it produces a final answer with no more tool calls.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a research agent with four tools: web_search, remember, recall, "
    "and read_file. Use recall first to check what you already know, use "
    "web_search for facts you don't have, use remember to save useful facts "
    "you discover so you don't need to search for them again, and use "
    "read_file to read local .txt/.pdf files when asked. Give a clear final "
    "answer once you have enough information."
)


def run_agent(user_question: str, max_turns: int = 6) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    # Claude sometimes writes its answer in the same message as a tool call
    # (e.g. "Paris is... Let me save this." + a remember() call), leaving the
    # *next* turn with nothing left to say. Track the last non-empty
    # assistant text so we can fall back to it if the final turn is empty.
    last_assistant_text = None

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            last_assistant_text = msg.content

        if not msg.tool_calls:
            return msg.content or last_assistant_text or "(no answer produced)"

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            print(f"  -> calling skill: {name}({args})")
            result = call_tool(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "Reached max turns without a final answer."


# ---------------------------------------------------------------------------
# Demo: a multi-hop question that needs two lookups (capital, then what it's
# famous for) and should exercise both the web_search and memory skills.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    question = "What is the capital of France and what is it famous for?"
    print(f"Question: {question}\n")

    answer = run_agent(question)

    print("\nFinal answer:")
    print(answer)
    print(f"\nMemory at end of session: {memory}")
    print(f"Tool call log written to: {LOG_FILE}")
