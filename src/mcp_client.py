"""
mcp_client.py

A minimal MCP client that connects to mcp_server.py and demonstrates:
  1. Calling the `calculator` tool.
  2. Reading the `kb://facts` resource (knowledge_base.txt).

An MCP *client* is the counterpart to an MCP server: it starts/connects to
the server, discovers what resources/tools/prompts it offers, and lets an
application (or, in a real agent, an LLM via function calling) use them
through one standard interface. Here we point the FastMCP `Client` straight
at the server script, which launches it as a subprocess and talks to it over
stdio — no network setup required for local use.
"""

import asyncio
from pathlib import Path

from fastmcp import Client

SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"


async def main() -> None:
    # Pointing the client at the server script launches it as a subprocess
    # and speaks MCP to it over stdio.
    client = Client(str(SERVER_SCRIPT))

    async with client:
        print("Connected to MCP server.\n")

        # --- Discover what the server offers -------------------------------
        tools = await client.list_tools()
        resources = await client.list_resources()
        print("Available tools:", [t.name for t in tools])
        print("Available resources:", [str(r.uri) for r in resources])
        print()

        # --- Call the calculator tool ---------------------------------------
        result = await client.call_tool(
            "calculator", {"a": 12, "b": 4, "operation": "multiply"}
        )
        print("calculator(12, 4, 'multiply') ->", result.data)

        try:
            await client.call_tool("calculator", {"a": 10, "b": 0, "operation": "divide"})
        except Exception as e:
            print("calculator(10, 0, 'divide') ->", f"error: {e}")

        # --- Read the knowledge_base resource -------------------------------
        contents = await client.read_resource("kb://facts")
        print("\nknowledge_base.txt contents (via MCP resource):")
        print(contents[0].text)


if __name__ == "__main__":
    asyncio.run(main())
