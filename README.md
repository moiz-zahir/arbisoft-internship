# Personal Finance Assistant

An AI-powered bank statement analyzer. Feed it a CSV of bank transactions and it
categorizes your spending, detects patterns and anomalies, answers plain-English
questions, and writes friendly spending summaries — all backed by Claude Haiku 4.5
(via OpenRouter), with optional local categorization through Ollama.

📖 **[Usage Guide](docs/USAGE.md)** — installation, running the CLI, example
questions, using your own CSV, known limitations.

🏗️ **[Architecture](docs/ARCHITECTURE.md)** — system diagram, what each file does,
how data flows through the pipeline, and why each design decision was made.

## Quick start

```bash
uv sync
cp .env.example .env        # then add your OPENROUTER_API_KEY
uv run python -m src.cli
```

That's it — the CLI walks you through categorizing the bundled sample statement,
shows a spending report, and drops you into a menu to ask questions or get a
summary. See the [Usage Guide](docs/USAGE.md) for details and the CSV format
needed to use your own bank statement.

## Project structure

```
src/
  agents/
    categorizer.py        # classifies transactions (local Ollama or cloud OpenRouter)
    model_router.py        # detects whether Ollama is available and routes accordingly
    pattern_detector.py     # spending totals, top categories, anomalies (Python + LLM narrative)
    query_agent.py           # RAG-based natural-language Q&A over transactions
    summarizer.py             # natural-language time-period summaries
  tools/
    csv_ingestion.py          # parses bank statement CSVs
    transaction_store.py       # persists/retrieves transactions (ChromaDB)
    validation_guards.py        # batch-level sanity checks
  models/
    transaction.py               # Pydantic models for transaction data
  mcp/
    server.py                     # FastMCP server exposing resources/tools
  cli.py                           # interactive command-line interface

data/            # your own bank statements go here (gitignored contents)
sample_data/     # sample_transactions.csv for testing
docs/            # USAGE.md and ARCHITECTURE.md
tests/           # pytest suite
```

## Running the test suite

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

## Running the MCP server

```bash
uv run python -m src.mcp.server
```

Exposes `finance://summary` and `finance://transactions` as resources, and
`query_finances` / `get_category_total` as tools. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what each one does and why.
