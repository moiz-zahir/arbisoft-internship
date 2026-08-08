# Architecture

## System diagram

```
+-------------------------------+
|  sample_data/*.csv             |
|  data/*.csv  (your own)        |
+---------------+-----------------+
                |
                v
+---------------------------------------------+
|  csv_ingestion.py                             |
|  CSV rows -> Transaction (Pydantic-validated)  |
+---------------------+-------------------------+
                      | TransactionBatch
                      v
+----------------------------------------------------------+
|  categorizer.py                                            |
|                                                              |
|    model_router.py --ping--> is Ollama running?              |
|          |                                                     |
|      yes |                                       no             |
|          v                                       v               |
|    Ollama (llama3)                        OpenRouter               |
|    localhost:11434/v1                    (Claude Haiku 4.5)          |
|          |                                       ^                    |
|          +------------ error? fall back ---------+                    |
+---------------------------+--------------------------------------------+
                            | categorized TransactionBatch
                            v
+------------------------------------------------+
|  validation_guards.py                            |
|  batch-level sanity checks -> warnings only        |
|  (never blocks the pipeline)                        |
+---------------------+------------------------------+
                      v
+------------------------------------------------+
|  transaction_store.py                            |
|  embeds + persists -> ChromaDB                     |
|  (data/chroma_db/, on disk)                          |
+-----------+----------------------+--------------------+
            |                      |
            v                      v
+---------------------+   +----------------------------+
|  pattern_detector.py  |   |  query_agent.py             |
|  Python: totals,       |   |  RAG: Chroma retrieval +    |
|  top categories,        |   |  forced tool call,           |
|  >2x anomalies           |   |  Python-corrected             |
|  LLM: narrative summary  |   |  dollar amounts in answer      |
+-----------+-------------+   +--------------+------------------+
            |                                |
            +----------------+----------------+
                             v
                  +----------------------+
                  |  summarizer.py         |
                  |  Python: period parsing |
                  |  + period totals         |
                  |  LLM: insight/recs/prose  |
                  +-----------+-------------+
                              |
              +----------------+-----------------+
              v                                  v
   +-------------------+              +-----------------------------+
   |     cli.py          |              |     mcp/server.py           |
   |  interactive          |              |  FastMCP resources & tools    |
   |  terminal menu:        |              |  for external MCP clients      |
   |  ask / summarize /      |              |  (Claude Desktop, IDE agents,    |
   |  exit                    |              |  other automated systems)         |
   +-------------------+              +-----------------------------+
```

## What each file does

| File | One-sentence description |
|---|---|
| `src/models/transaction.py` | Defines `TransactionType`, `Transaction`, and `TransactionBatch` — the Pydantic models that validate every transaction at the boundary before it can reach storage or an LLM prompt. |
| `src/tools/csv_ingestion.py` | Parses a bank statement CSV into a `TransactionBatch`, skipping and logging malformed rows and raising clear errors for missing/empty files. |
| `src/agents/model_router.py` | Pings Ollama to decide whether categorization should run locally or in the cloud, and logs which backend actually served each transaction. |
| `src/agents/categorizer.py` | Classifies each transaction into a `TransactionType` via a forced LLM tool call (local Ollama or cloud OpenRouter), with retries and a low-confidence/`OTHER` fallback. |
| `src/tools/validation_guards.py` | Runs four batch-level sanity checks (negative amounts, date format, low confidence, `OTHER`-category overuse) and turns any failure into a warning instead of a crash. |
| `src/tools/transaction_store.py` | Embeds and persists transactions in a disk-backed ChromaDB collection, and exposes semantic search, category filtering, and anomaly-marking. |
| `src/agents/pattern_detector.py` | Computes spending totals, top categories, and >2x-average anomalies in Python, then asks the LLM only to narrate the findings. |
| `src/agents/query_agent.py` | Answers natural-language questions via RAG (Chroma retrieval + a forced tool call), then corrects any dollar figures in the LLM's answer against the real transaction amounts. |
| `src/agents/summarizer.py` | Resolves a natural-language time period into a date range in Python, computes that period's totals in Python, and asks the LLM only for the insight, recommendations, and narrative. |
| `src/mcp/server.py` | Exposes the spending report, raw transactions, question-answering, and category totals over MCP so external MCP clients can use them. |
| `src/cli.py` | The interactive terminal entry point: asks for a model backend, runs the full pipeline once, then loops on a menu (ask a question / get a summary / exit). |

## How data flows through the pipeline

1. **Ingest** — `csv_ingestion.py` reads the CSV and produces a `TransactionBatch`
   of validated `Transaction` objects. Bad rows are dropped and logged, not fatal.
2. **Categorize** — `cli.py` asks the user for a model backend preference;
   `model_router.py` checks whether Ollama is actually reachable, and
   `categorizer.py` classifies each transaction accordingly, with the local path
   falling back to the cloud path on any error.
3. **Validate** — `validation_guards.py` runs cross-transaction sanity checks
   against the now-categorized batch and surfaces any issues as warnings.
4. **Store** — `transaction_store.py` embeds each transaction (description +
   category + type) and persists it to ChromaDB on disk, so the data survives
   between runs.
5. **Analyze** — `pattern_detector.py` reads everything back out of ChromaDB,
   computes the spending report in Python, and calls the LLM once for a narrative
   summary. This runs automatically on every CLI startup.
6. **Interact** — from the CLI menu (or an MCP client), `query_agent.py` answers
   ad-hoc questions and `summarizer.py` answers period-based summary requests, both
   retrieving from the same ChromaDB store.
7. **Serve** — `mcp/server.py` wraps steps 5 and 6 (plus raw transaction access) as
   MCP resources and tools, so the same logic is reachable from outside the CLI.

## Why each design decision was made

**Python does the math, the LLM does the language.** Every number a downstream
consumer actually reads — spending totals, savings rate, category averages, the
>2x anomaly ratio, the dollar amounts in a query answer — is computed in plain
Python, never by the LLM. This isn't a hypothetical concern: this project hit a
real, reproducible case where an LLM correctly cited five transaction amounts and
then added them up wrong by about a dollar. LLMs are used only where they're
actually good: classifying a transaction into a category, and writing narrative
text (a summary paragraph, an insight, a recommendation) that a human reads
directly and isn't parsed by other code.

**Two-stage validation wherever an LLM produces structured output.** A constrained
tool schema (an enum of valid categories, a `minimum`/`maximum` on a confidence
score) is the first line of defense — but it's a hint providers are expected to
follow, not an enforced guarantee. Every tool-call response is re-validated with
Pydantic before it's trusted, so a hallucinated category or an out-of-range score
fails loudly instead of corrupting stored data.

**Four separate agents instead of one large one.** `categorizer.py`,
`pattern_detector.py`, `query_agent.py`, and `summarizer.py` each have one
narrow job — classify a single transaction, summarize a whole batch, answer a
question about retrieved transactions, or summarize a period. That separation
means each agent can retry, validate, and fall back independently: a categorizer
failure doesn't take down pattern detection, and a bad query doesn't corrupt the
stored data.

**Local/cloud model routing for categorization specifically.** Categorization is
the highest-volume LLM call in the whole pipeline — one call per transaction,
versus one call for an entire pattern report or query answer. That makes it the
one place where a local model's speed, cost, and privacy tradeoff actually pays
off. `model_router.py` does a *live* availability check rather than trusting a
cached assumption, since Ollama can be started or stopped between runs, and every
local attempt has an automatic, transparent fallback to cloud so a categorization
always gets produced one way or another.

**ChromaDB as the single store.** Transactions need both exact filtering (get me
everything in category FOOD) and semantic search (find transactions relevant to
"coffee and food purchases"). Chroma provides both from one store without needing
a separate search index layered on top of a regular database.

**Validation guards as a separate pipeline stage, not just Pydantic fields.**
Pydantic validates that a single `Transaction` is structurally legal (a debit is a
number, a type is "debit" or "credit"). Guards validate something Pydantic can't:
whether the *batch as a whole* is plausible — is more than 20% of it falling back
to `OTHER` (a sign categorization is failing broadly, not on one bad row)? Are
confidence scores trending low? Those are cross-transaction questions that only
make sense to ask once the whole batch exists.

**An MCP server alongside the CLI.** The CLI is one interface into this project's
logic; MCP makes the same categorized data, pattern report, and question-answering
available to any other MCP-compatible client (Claude Desktop, an IDE agent, another
automated system) without re-implementing the CSV pipeline, the ChromaDB schema, or
the OpenRouter wiring elsewhere. Splitting resources (cheap, read-only data) from
tools (anything requiring computation) lets a client inspect existing data without
triggering an LLM call just to look.
