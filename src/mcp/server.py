import json

from fastmcp import FastMCP

from src.agents.pattern_detector import build_report, format_report
from src.agents.query_agent import ask_question
from src.tools.transaction_store import get_all, get_by_category

# MCP turns this project's finance logic into something any compatible
# client (Claude Desktop, an IDE agent, another automated system) can use
# without knowing our internals - the CSV pipeline, the ChromaDB schema, the
# OpenRouter wiring. A client just sees "resources" (data it can read) and
# "tools" (actions it can invoke) over a standard protocol, so this same
# categorized transaction data and these same agents become reusable outside
# this codebase instead of being locked behind the CLI.
#
# Resources vs. tools here is deliberate: the two resources below expose
# already-computed data cheaply and read-only (a client can inspect spending
# without triggering an LLM call just to look). The two tools are for
# requests that require actual computation - answering a novel question, or
# aggregating a category on demand.
mcp = FastMCP("Personal Finance Assistant")


@mcp.resource("finance://summary")
def finance_summary() -> str:
    """The latest spending pattern report (top categories, anomalies, recurring charges, totals) as text."""
    records = get_all()
    if not records:
        return "No transactions stored yet - run the ingestion pipeline first."
    report = build_report(records)
    return format_report(report)


@mcp.resource("finance://transactions")
def finance_transactions() -> str:
    """Every stored transaction (id, description, date, amount, category, type) as JSON."""
    return json.dumps(get_all(), indent=2)


@mcp.tool()
def query_finances(question: str) -> dict:
    """Answer a natural-language question about the user's transactions using RAG over ChromaDB."""
    return ask_question(question).model_dump()


@mcp.tool()
def get_category_total(category: str) -> dict:
    """Total amount spent (debits only) in a given category, e.g. 'FOOD' or 'SHOPPING'."""
    category = category.strip().upper()
    records = [r for r in get_by_category(category) if r["metadata"]["type"] == "debit"]
    total = sum(r["metadata"]["amount"] for r in records)
    return {"category": category, "total_spent": total, "transaction_count": len(records)}


if __name__ == "__main__":
    mcp.run()
