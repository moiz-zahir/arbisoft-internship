"""
mcp_server.py

A minimal MCP (Model Context Protocol) server built with FastMCP.

-----------------------------------------------------------------------------
What is MCP?
-----------------------------------------------------------------------------
MCP is an open protocol that standardizes how AI applications (like an LLM
agent or a chat client) connect to external context and capabilities. Instead
of every app inventing its own bespoke integration for "give the model a
file" or "let the model run a calculation," an MCP *server* exposes a fixed
set of capabilities over a standard interface, and any MCP *client* can talk
to it the same way — regardless of what language or framework the server was
built with. Think of it like a USB port for AI apps: one plug shape, many
devices.

An MCP server can expose three main kinds of capabilities; this file uses two
of them:

RESOURCES
    Read-only pieces of context the client can fetch by URI — e.g. a file, a
    database row, an API response. Resources are for *data the model reads*,
    not actions it performs. Here, `knowledge_base` is a resource that
    returns the contents of knowledge_base.txt.

TOOLS
    Functions the client (and, through it, the model) can *call* to perform
    an action or computation and get a result back — the MCP equivalent of
    function calling. Here, `calculator` is a tool that performs arithmetic.

(The third kind, not used here, is "prompts" — reusable prompt templates a
server can offer to clients.)
-----------------------------------------------------------------------------
"""

from pathlib import Path

from fastmcp import FastMCP

# Create the MCP server. The name shows up in client-side introspection.
mcp = FastMCP("research-mcp-server")

KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.txt"


# ---------------------------------------------------------------------------
# RESOURCE: exposes knowledge_base.txt for clients to read.
# Resources are addressed by URI — here we invent the scheme "kb://facts".
# ---------------------------------------------------------------------------
@mcp.resource("kb://facts")
def knowledge_base() -> str:
    """Returns the contents of knowledge_base.txt — 5 facts about AI."""
    return KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# TOOL: a calculator the client (or an LLM through the client) can invoke.
# FastMCP turns this function's type hints/docstring into the tool's schema
# automatically, the same way function-calling schemas work for LLM tools.
# ---------------------------------------------------------------------------
@mcp.tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        a: The first number.
        b: The second number.
        operation: One of "add", "subtract", "multiply", "divide".
    """
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    raise ValueError(f"Unknown operation: {operation!r}")


if __name__ == "__main__":
    # Runs the server over stdio by default, the standard transport for
    # local MCP servers launched as a subprocess by a client.
    mcp.run()
