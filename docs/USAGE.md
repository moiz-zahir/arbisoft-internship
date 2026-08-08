# Usage Guide

## What this project does

This is a personal finance assistant that reads a bank statement CSV and turns it
into something you can actually talk to. Under the hood it:

- Parses your transactions and validates them
- Categorizes each one (food, transport, shopping, subscriptions, entertainment,
  income, etc.) using an LLM
- Stores them in a local vector database so you can search them semantically
- Detects spending patterns: top categories, unusually large transactions, and
  recurring charges
- Answers plain-English questions like *"how much did I spend on food?"*
- Writes friendly, advisor-style summaries for a time period ("July", "last month")
- Exposes all of the above over MCP, so other tools (Claude Desktop, an IDE agent)
  can use it too

Everything runs from a single interactive CLI, or as an MCP server.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **An OpenRouter API key** ([openrouter.ai](https://openrouter.ai)) — this is what
  powers the categorization, pattern detection, query, and summary agents by default
  (via Claude Haiku 4.5)
- **Ollama (optional)** — if you want categorization to run locally instead of via
  OpenRouter. Install from [ollama.com](https://ollama.com) and pull a model:
  ```bash
  ollama pull llama3
  ```
  If Ollama isn't installed or isn't running, the project automatically falls back
  to OpenRouter — nothing breaks either way.

## Installation from scratch

```bash
# 1. Clone/enter the project directory
cd week5

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Set up your environment file
cp .env.example .env
```

Open `.env` and fill in your key:

```
OPENROUTER_API_KEY=your_openrouter_key_here
```

(`ANTHROPIC_API_KEY` is present in `.env.example` but not currently required by
anything in this project — it's fine to leave it blank.)

## Running with the sample data

The repo includes a ready-to-use sample statement at
`sample_data/sample_transactions.csv` (20 transactions). Just run:

```bash
uv run python -m src.cli
```

On startup you'll be asked:

```
Use local model (Ollama) or cloud model (OpenRouter) for categorization? (local/cloud):
```

Type `cloud` (or `local` if you have Ollama running) and press Enter. The CLI will
then:

1. Load and validate the CSV
2. Categorize every transaction
3. Run validation guards and print any warnings
4. Store everything in ChromaDB
5. Run pattern detection and print a full spending report

...and drop you into a menu:

```
Options: (1) Ask a question  (2) Get spending summary  (3) Exit
```

## Example questions you can ask

Pick option `1` from the menu and try things like:

- `how much did I spend on food?`
- `what are my biggest expenses?`
- `any suspicious transactions?`
- `how much did I earn this month?`
- `what did I spend on subscriptions?`

Pick option `2` for a narrative summary, and when asked for a time period try:

- `July`
- `this month`
- `last month`
- `2026-07`

(Note: "this month" / "last week" are resolved against today's real date, so they'll
correctly return "no transactions found" unless your data actually falls in that
window — the sample data is all from July 2026.)

## Using your own bank statement

Run the CLI with a path to your own CSV:

```bash
uv run python -m src.cli path/to/your_statement.csv
```

Your CSV needs exactly these four columns (any extra columns are ignored):

| Column        | Type   | Notes                                  |
|----------------|--------|-----------------------------------------|
| `date`         | string | `YYYY-MM-DD` format, e.g. `2026-07-01`  |
| `description`  | string | merchant/transaction description        |
| `amount`       | number | positive number, e.g. `45.21`           |
| `type`         | string | exactly `debit` or `credit`             |

Example:

```csv
date,description,amount,type
2026-07-01,STARBUCKS COFFEE #4521,5.75,debit
2026-07-02,SALARY DEPOSIT ACME CORP,3200.00,credit
```

Rows that don't parse (bad amount, unrecognized `type`, etc.) are skipped and
logged rather than failing the whole import — you'll see how many rows were
skipped in the warnings. Drop statement files into `data/` if you want a
consistent place to keep them (its contents aren't committed to git).

## Known limitations

- **LLM arithmetic in narrative text.** Every number your code actually reads
  (spending totals, savings rate, category averages) is computed in Python, never
  by the LLM. But the free-text narrative an LLM writes (a summary paragraph, an
  answer to a question) can still restate a number slightly wrong when it's just
  talking about the data rather than being asked to return it as a field. The
  query agent post-processes its answer text to correct dollar figures against the
  real transaction amounts, but this is a targeted fix for that one agent — the
  pattern detector's and summarizer's prose summaries aren't corrected the same way,
  so treat their narrative paragraphs as informational, not authoritative.
- **Local model quality and tool-calling reliability.** `llama3` via Ollama is
  much smaller than Claude Haiku 4.5. It may categorize less accurately, and its
  support for forced tool calling is less consistent — that's exactly why
  local-model categorization automatically falls back to OpenRouter on any error.
- **Semantic retrieval isn't exhaustive.** The query agent retrieves a fixed number
  of "most relevant" transactions (10 by default) before answering. For a small
  statement like the sample data this covers everything, but on a much larger
  statement, a broad question could miss transactions that didn't rank in the
  top matches.
- **Time period parsing is deliberately limited.** `summarizer.py` understands
  `today`, `yesterday`, `this/last week`, `this/last month`, month names, and
  `YYYY-MM` — not arbitrary phrasing like "the first two weeks of July" or "Q3".
  Anything it can't parse raises a clear error rather than guessing.
- **No real bank integration.** This project only reads CSV exports you provide
  yourself; it doesn't connect to any bank, aggregator, or live transaction feed.
