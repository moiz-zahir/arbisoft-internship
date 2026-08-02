# week5

An AI-powered bank statement analyzer. Upload a CSV of bank transactions and use
LLM-backed agents to categorize spending, detect patterns, and answer natural
language questions about your finances.

## What it does

- **Ingestion** — parses uploaded bank statement CSVs into structured transaction records.
- **Categorization** — an agent classifies each transaction (food, transport, shopping, subscriptions, entertainment, etc.).
- **Pattern detection** — an agent surfaces recurring charges, spending trends, and anomalies.
- **Query agent** — ask questions about your transactions in plain English (e.g. "how much did I spend on food last month?").
- **Storage** — transactions and embeddings are persisted with Chroma for retrieval.
- **MCP server** — exposes the project's tools/agents over the Model Context Protocol so they can be used from MCP-compatible clients.
- **CLI** — a command-line entry point for ingesting statements and running queries directly.

## Project structure

```
src/
  agents/
    categorizer.py       # classifies transactions into categories
    pattern_detector.py  # finds recurring charges & spending patterns
    query_agent.py        # answers natural-language questions
  tools/
    csv_ingestion.py      # parses uploaded bank statement CSVs
    transaction_store.py  # persists/retrieves transactions (Chroma)
  models/
    transaction.py        # Pydantic models for transaction data
  mcp/
    server.py             # MCP server exposing tools/agents
  cli.py                  # command-line interface

data/            # uploaded bank statements go here (gitignored contents)
sample_data/     # sample_transactions.csv for testing
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:

- `OPENROUTER_API_KEY`
- `ANTHROPIC_API_KEY`

## Running

Run the CLI:

```bash
uv run src/cli.py
```

Run the MCP server:

```bash
uv run src/mcp/server.py
```

Try it out with the bundled sample data at `sample_data/sample_transactions.csv`,
or drop your own bank statement CSV into `data/`.
