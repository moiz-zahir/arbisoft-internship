# Week 4 Prompt Log

A record of every prompt given during Week 4, what was built in response, and the outcome.

---

### 1. "hi"

**Built:** Nothing — a greeting exchange to start the session.

**Outcome:** N/A.

---

### 2. "Set up a Python project in this folder using uv. Run uv init, then add these dependencies: openai, python-dotenv, chromadb, requests. Also create a .gitignore that ignores .env, .venv/, and __pycache__. Show each step as you run it."

**Built:** Initialized a uv-managed Python project (`pyproject.toml`, `main.py`, `README.md`, git repo). Discovered the target folder was read-only and fixed permissions before `uv init` could run. Discovered the system default Python (3.9) was incompatible with `chromadb`'s `onnxruntime` dependency, so pinned the project to Python 3.12. Also discovered this Intel Mac needed an `onnxruntime<1.24` constraint since newer versions dropped x86_64 macOS wheels. Installed `openai`, `python-dotenv`, `chromadb`, `requests` via `uv add`. Wrote `.gitignore` covering `.env`, `.venv/`, `__pycache__/`.

**Outcome:** Project set up successfully — 87 packages installed, `requires-python` tightened to `>=3.10` to match reality, `uv sync` verified clean.

---

### 3. "Check the background uv add task output; if it succeeded, update .gitignore to include .env and report final project status to the user."

**Built:** Nothing new — this work was already completed in the prior turn.

**Outcome:** Confirmed `uv add` had succeeded and `.gitignore` already included `.env`; reported final status back to the user.

---

### 4. "Create a .env file with these placeholders: OPENROUTER_API_KEY=your_openrouter_key_here, ANTHROPIC_API_KEY=your_anthropic_key_here. Then open .env and I will fill in the real keys manually."

**Built:** Created `.env` with the two placeholder lines and opened it in the default text editor.

**Outcome:** File created and opened; user filled in real API keys manually afterward.

---

### 5. "done i filled it and saved"

**Built:** Nothing new — verified the existing setup rather than building anything.

**Outcome:** Confirmed `.env` was correctly listed in `.gitignore` (`git check-ignore -v .env`), so the real keys wouldn't be committed.

---

### 6. "Create src/research_agent.py - a research agent that: loads OPENROUTER_API_KEY from .env, has a web search skill using requests against the DuckDuckGo API, has a memory system storing facts in a Python list, has a hook logging every tool call with a timestamp to agent_logs.txt, has a file-read plugin for .txt/.pdf files, uses claude-haiku-4.5 via OpenRouter with function calling, and runs a demo answering 'What is the capital of France and what is it famous for?' using web search and memory. Add clear comments explaining skills, hooks, and memory. Then run it and show me the output."

**Built:** `src/research_agent.py` — an agent with four tools (`web_search`, `remember`, `recall`, `read_file`) exposed to Claude Haiku 4.5 via OpenRouter's OpenAI-compatible function-calling API, a `memory` list, and a `log_tool_call` hook writing timestamped entries to `agent_logs.txt`. Added `pypdf` as a dependency for PDF reading. Extensive comments explain what skills, memory, and hooks mean conceptually.

**Outcome:** Ran the demo successfully. The agent called `recall`, tried `web_search` twice (DuckDuckGo's Instant Answer API returned nothing useful for those queries — a known limitation of that endpoint), fell back to its own knowledge, and called `remember` to save the fact. Along the way, hit and fixed a real bug: Claude sometimes puts its answer text in the same message as a tool call, leaving the next turn empty — added a fallback that tracks the last non-empty assistant text. Final answer correctly identified Paris and its landmarks; `agent_logs.txt` recorded all 4 tool calls with timestamps.

---

### 7. "Create src/mcp_server.py - a simple MCP server that exposes one resource (knowledge_base.txt with 5 AI facts) and one tool (a calculator taking two numbers and an operation), uses FastMCP (install with uv add fastmcp), with comments explaining MCP/resources/tools. Also create mcp_client.py that connects and demonstrates calling the calculator and reading the resource. Run the server and client and show me the output."

**Built:** Installed `fastmcp`. Created `src/knowledge_base.txt` (5 AI facts), `src/mcp_server.py` (FastMCP server exposing resource `kb://facts` and tool `calculator`), and `src/mcp_client.py` (connects over stdio, lists tools/resources, calls the calculator, reads the resource). Comments explain what MCP is and what resources vs. tools mean in that context.

**Outcome:** Ran the client (which launches the server as a subprocess). Tool/resource discovery worked (`tools=['calculator']`, `resources=['kb://facts']`), `calculator(12, 4, 'multiply') → 48.0`, a divide-by-zero call correctly raised and was caught client-side, and all 5 facts were read back successfully through the resource.

---

### 8. "Check the background uv add fastmcp task output; once it completes, inspect the installed fastmcp API and continue building src/mcp_server.py and src/mcp_client.py."

**Built:** Nothing new — this work was already completed in the prior turn.

**Outcome:** Confirmed `fastmcp` had installed and both files were already built and verified working; summarized status back to the user.

---

### 9. "Create src/supervisor_agent.py - a multi-agent system with a supervisor that routes tasks to a research worker (web search) or a math worker (calculations), a tracing layer logging every tool call and handoff with timestamps to trace_logs.txt, both workers using claude-haiku-4.5 via OpenRouter, with comments explaining the supervisor/worker pattern. Demo: route 'What is machine learning?' and 'What is 15 multiplied by 47?' Then run it and show me the output."

**Built:** `src/supervisor_agent.py` — a supervisor that uses a forced tool call (`tool_choice="required"`) on a single `route` tool to pick between a `research` worker (`web_search` skill) and a `math` worker (a safe AST-based `calculate` tool, no `eval()`). A generic `run_worker()` loop is shared by both workers. A tracing layer (`trace`, `log_handoff`, `log_tool_call`) timestamps every handoff and tool call to `trace_logs.txt`. Comments explain the supervisor/worker pattern and why multi-agent systems are useful.

**Outcome:** Ran the demo. "What is machine learning?" routed to the research worker, which answered correctly from its own knowledge (didn't need to call `web_search` this time). "What is 15 multiplied by 47?" routed to the math worker, which called `calculate("15 * 47")` and correctly returned 705. `trace_logs.txt` recorded both handoffs and the one tool call.

---

### 10. "Create prompts.md in the root of this project. Add an entry for every prompt I gave you during Week 4..."

**Built:** This file (`prompts.md`) — a log of every Week 4 prompt plus a "Concepts Learned" section.

**Outcome:** You're reading it.

---

## Concepts Learned

### AI agents vs. chatbots

A chatbot takes a message and returns a message — it's a single request/response loop with no ability to act on the world beyond generating text. An **agent** is a chatbot given the ability to *act*: it can decide, on its own, to call a tool (search the web, run a calculation, read a file), look at the result, and decide what to do next — possibly calling more tools — before giving a final answer. The defining feature of an agent isn't the model itself, it's the **loop**: observe → decide → act → observe again, repeated until the task is done. `research_agent.py`'s `run_agent()` function is exactly this loop.

### Skills / tools in agentic AI

A "skill" (also called a "tool" in function-calling terminology) is a specific capability exposed to the model as a callable function with a defined schema (name, description, parameters). The model never runs the code itself — it outputs a *request* to call a function with certain arguments, and the surrounding program executes the real Python function and feeds the result back in. This is how an LLM, which can only generate text, gets to interact with the outside world: `web_search`, `read_file`, and `calculate` in this project are all skills defined this way.

### Hooks and why they matter

A hook is code that runs automatically around an agent's actions, regardless of *which* action it is — used for cross-cutting concerns that shouldn't have to be duplicated inside every tool. In this project, `log_tool_call()` in `research_agent.py` and `trace()` in `supervisor_agent.py` are hooks: every tool call passes through them before/after execution, giving a consistent audit log without cluttering the tool implementations themselves. Hooks matter because they're where you put logging, rate-limiting, safety checks, or auditing once, instead of scattering that logic through every skill.

### Memory types: in-context, vector, key-value

- **In-context memory** is the simplest kind: everything the model "remembers" is just text sitting in its current prompt/conversation history. It disappears once the conversation ends or falls out of the context window. This project's `memory` list in `research_agent.py` is a step above pure in-context memory — it's an explicit list the agent writes to and reads from on purpose, but it still only lives for the process's lifetime.
- **Vector memory** stores information as embeddings (numeric vectors capturing meaning) in a database, so the agent can later retrieve facts by *semantic similarity* rather than exact match — "find things related to this," not just "find this exact string." `chromadb`, installed in this project, is a vector database built for exactly this purpose, though it wasn't wired into an agent yet.
- **Key-value memory** stores facts as explicit key → value pairs (like a dictionary or a database row), retrieved by exact lookup rather than similarity. It's simpler and more precise than vector memory when you know exactly what you're looking for (e.g. `user_id -> preferences`).

The right choice depends on retrieval needs: exact lookups favor key-value, "find something related" favors vector, and short-lived session facts are fine as plain in-context/list memory.

### What MCP is and why it exists

MCP (Model Context Protocol) is an open standard for connecting AI applications to external context and capabilities, so that servers exposing tools/resources and clients (agents, chat apps) that consume them can interoperate without custom one-off integrations for every pairing. It exists because, without a shared protocol, every AI app would need bespoke code to talk to every data source or tool — MCP standardizes that into one interface any client can speak. In `mcp_server.py`, **resources** are read-only data the client can fetch by URI (like `knowledge_base.txt`), and **tools** are functions the client can invoke to perform an action or computation (like `calculator`) — the MCP equivalent of function calling, but decoupled from any specific model provider.

### Supervisor/worker pattern and when to use it

Rather than one agent trying to handle every kind of task with every possible tool, a **supervisor** agent's only job is to look at an incoming task and route it to the right specialized **worker** agent, which has just the tools and prompt it needs for its narrow job. `supervisor_agent.py` demonstrates this: the supervisor never answers a task itself, it just calls a `route` tool to pick `research` or `math`, then hands off.

Use this pattern when:
- Tasks fall into clearly distinguishable categories that need different tools or expertise.
- A single agent's prompt/toolset would otherwise grow large and unfocused, hurting reliability.
- You want to add new capabilities (a new worker) without touching or risking existing ones.
- You want independent observability — tracing exactly which worker handled what, and what tools it used, as this project's `trace_logs.txt` does.

It's overkill for simple, single-domain tasks where one focused agent is already enough.
