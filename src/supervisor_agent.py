"""
supervisor_agent.py

A minimal multi-agent system using the supervisor/worker pattern, with
Claude Haiku 4.5 (via OpenRouter) powering both the supervisor and the
workers.

-----------------------------------------------------------------------------
Supervisor / worker pattern
-----------------------------------------------------------------------------
Instead of one agent trying to be good at everything (search the web, do
math, write code, ...), a SUPERVISOR agent's only job is to look at an
incoming task and decide which specialized WORKER agent should handle it.
Each worker is narrowly scoped — it gets only the tools and system prompt it
needs for its specialty — and the supervisor never answers the task itself,
it just routes.

Why use multiple agents instead of one?
  - Separation of concerns: each worker's prompt and toolset stays small and
    focused, which makes it more reliable at its one job than a single agent
    juggling every tool would be.
  - Easier to extend: adding a new capability means adding a new worker and
    teaching the supervisor about it, without touching existing workers.
  - Easier to debug/observe: routing decisions and each worker's tool use
    can be traced independently (see the tracing layer below).
  - Mirrors how human teams work: a dispatcher routes a request to the right
    specialist rather than one generalist doing everything.
-----------------------------------------------------------------------------
Tracing
-----------------------------------------------------------------------------
The `trace()` function is a logging hook that fires on two kinds of events
across the whole system: agent HANDOFFs (supervisor routing a task to a
worker) and TOOL CALLs (a worker invoking one of its tools). Every event is
timestamped and appended to trace_logs.txt, giving a full audit trail of how
a task moved through the system.
-----------------------------------------------------------------------------
"""

import os
import json
import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found — check your .env file")

MODEL = "anthropic/claude-haiku-4.5"  # Claude Haiku 4.5 via OpenRouter
TRACE_FILE = Path(__file__).resolve().parent.parent / "trace_logs.txt"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ---------------------------------------------------------------------------
# TRACING LAYER: logs every agent handoff and tool call, system-wide.
# ---------------------------------------------------------------------------
def trace(event: str) -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {event}\n")


def log_tool_call(agent_name: str, tool_name: str, arguments: dict) -> None:
    trace(f"TOOL CALL: {agent_name}.{tool_name}({json.dumps(arguments)})")


def log_handoff(from_agent: str, to_agent: str, task: str) -> None:
    trace(f"HANDOFF: {from_agent} -> {to_agent} (task: {task!r})")


# ---------------------------------------------------------------------------
# WORKER 1: research worker — answers factual questions using web search.
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    """Calls the DuckDuckGo Instant Answer API (no key required)."""
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


RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (via DuckDuckGo) for facts and general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    },
]
RESEARCH_TOOL_IMPLS = {"web_search": lambda args: web_search(args["query"])}
RESEARCH_SYSTEM_PROMPT = (
    "You are a research worker agent. Use the web_search tool to look up "
    "facts you don't already know, then answer the question clearly and "
    "concisely based on what you find (and your own knowledge if search "
    "comes up short)."
)


# ---------------------------------------------------------------------------
# WORKER 2: math worker — solves calculation problems with a safe evaluator.
# ---------------------------------------------------------------------------
import ast
import operator as op

_SAFE_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError("Unsupported or unsafe expression")


def calculate(expression: str) -> float:
    """Safely evaluates a numeric expression (+, -, *, /, **) without using eval()."""
    tree = ast.parse(expression, mode="eval").body
    return _safe_eval_node(tree)


MATH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a numeric arithmetic expression, e.g. '15 * 47'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A numeric expression using +, -, *, /, **",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]
MATH_TOOL_IMPLS = {"calculate": lambda args: calculate(args["expression"])}
MATH_SYSTEM_PROMPT = (
    "You are a math worker agent. Use the calculate tool to evaluate any "
    "arithmetic in the task, then state the final numeric answer clearly."
)


# ---------------------------------------------------------------------------
# Generic worker loop: any worker is just a system prompt + a toolset.
# ---------------------------------------------------------------------------
def run_worker(agent_name: str, system_prompt: str, tools: list, tool_impls: dict, task: str, max_turns: int = 5) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    last_text = None

    for _ in range(max_turns):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            last_text = msg.content

        if not msg.tool_calls:
            return msg.content or last_text or "(no answer produced)"

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            log_tool_call(agent_name, name, args)  # tracing hook
            try:
                result = str(tool_impls[name](args))
            except Exception as e:
                result = f"Error running {name}: {e}"
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return last_text or "Reached max turns without a final answer."


def research_worker(task: str) -> str:
    return run_worker("research_worker", RESEARCH_SYSTEM_PROMPT, RESEARCH_TOOLS, RESEARCH_TOOL_IMPLS, task)


def math_worker(task: str) -> str:
    return run_worker("math_worker", MATH_SYSTEM_PROMPT, MATH_TOOLS, MATH_TOOL_IMPLS, task)


WORKERS = {
    "research": research_worker,
    "math": math_worker,
}


# ---------------------------------------------------------------------------
# SUPERVISOR: routes each incoming task to the correct worker.
# The supervisor never answers the task itself — it only decides *who*
# should, by calling the `route` tool exactly once.
# ---------------------------------------------------------------------------
ROUTE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "route",
            "description": "Route the task to the correct specialized worker agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "worker": {
                        "type": "string",
                        "enum": ["research", "math"],
                        "description": "'research' for factual/knowledge questions, 'math' for calculations.",
                    },
                    "reason": {"type": "string", "description": "Brief reason for this choice"},
                },
                "required": ["worker", "reason"],
            },
        },
    },
]
SUPERVISOR_SYSTEM_PROMPT = (
    "You are a supervisor agent. You do not answer tasks yourself. For each "
    "task you receive, decide whether it should go to the 'research' worker "
    "(factual/knowledge questions) or the 'math' worker (calculations), then "
    "call the route tool exactly once with your decision."
)


def supervisor_route(task: str) -> str:
    """Asks the supervisor LLM to pick a worker, then hands the task off."""
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=ROUTE_TOOL, tool_choice="required"
    )
    msg = response.choices[0].message
    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments or "{}")
    worker_name = args["worker"]
    reason = args.get("reason", "")

    print(f"  Supervisor decision: route to '{worker_name}' worker ({reason})")
    log_handoff("supervisor", f"{worker_name}_worker", task)

    worker_fn = WORKERS[worker_name]
    return worker_fn(task)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tasks = [
        "What is machine learning?",
        "What is 15 multiplied by 47?",
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        answer = supervisor_route(task)
        print(f"Answer: {answer}")

    print(f"\nFull trace written to: {TRACE_FILE}")
