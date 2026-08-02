# Week 5 Prompt Log

A record of every prompt given during this project's build, what was built in response, and the outcome.

---

## 1. Project setup

**Prompt:** Set up a Python project in this folder using uv. Run `uv init`, then add dependencies: openai, python-dotenv, chromadb, pydantic, pypdf, fastmcp, requests. Create a `.gitignore` ignoring `.env`, `.venv/`, and `__pycache__`.

**Built:** Fixed a read-only permission on the project folder that was blocking `uv init`. Ran `uv init`, then `uv add` for all requested packages. Merged `.env`/`.venv/`/`__pycache__` entries into the `.gitignore` uv had already scaffolded.

**Outcome:** Hit one real obstacle: this is an Intel (x86_64) Mac, and `onnxruntime` 1.28.0 (a transitive dependency of `chromadb`) no longer ships x86_64 macOS wheels. Pinned `onnxruntime==1.23.2`, the newest version that still does. All 125 packages installed cleanly.

---

## 2. Environment files

**Prompt:** Create `.env.example` with placeholders for `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY`. Create a real `.env` with the same structure but empty values.

**Built:** Both files, `.env` left blank (later filled in by the user directly).

**Outcome:** Done. `.env` is covered by `.gitignore` so the real keys never get committed.

---

## 3. Open `.env` in a text editor

**Prompt:** Open it on text editor.

**Built:** Opened `.env` with `open -e` (TextEdit).

**Outcome:** Done.

---

## 4. Folder structure and sample data

**Prompt:** Create the `src/` folder structure (agents, tools, models, mcp, cli.py), a `data/` folder for uploaded statements, a `sample_data/sample_transactions.csv` with 20 realistic transactions across food/transport/shopping/subscriptions/entertainment, and a `README.md`.

**Built:** All directories and empty placeholder files. A 20-row sample CSV mixing debits (coffee, groceries, gas, rideshare, streaming, electronics) and credits (salary, freelance payment, a refund). A `.gitkeep` in `data/` so the empty folder is tracked by git. A README explaining the project's purpose and how to run it.

**Outcome:** Done.

---

## 5. Transaction models and CSV ingestion

**Prompt:** Create Pydantic models (`TransactionType`, `Transaction`, `TransactionBatch`) in `src/models/transaction.py` with comments explaining why Pydantic validation matters. Create `src/tools/csv_ingestion.py` to parse a CSV into a `TransactionBatch`, handling malformed rows gracefully. Run it against the sample CSV.

**Built:** The three models, with `type` constrained to `Literal["debit", "credit"]` and `confidence_score` bounded `[0, 1]`. `load_transactions_csv()` checks for required columns up front and skips (logs, doesn't crash on) any row that fails validation. Added `__init__.py` files under `src/` so the modules import as a package.

**Outcome:** All 20 sample rows parsed cleanly; totals matched hand-calculated debit/credit sums.

---

## 6. Categorizer agent

**Prompt:** Create `src/agents/categorizer.py`: load `OPENROUTER_API_KEY`, categorize each transaction via Claude Haiku 4.5 (OpenRouter) using function calling, return a confidence score, flag anything under 0.7 confidence as "low confidence - needs review", validate every result with Pydantic, retry up to 3 times on validation failure. Run against the sample CSV.

**Built:** `categorize_transaction()` forces a tool call (`categorize_transaction`) constrained to the `TransactionType` enum plus a `[0,1]` confidence score, validated through a `CategorizationResult` Pydantic model with a 3-attempt retry loop on parse/validation failure.

**Outcome:** All 20 transactions categorized correctly (Starbucks → FOOD, Netflix → SUBSCRIPTION, salary → INCOME, etc.) with confidence scores between 0.95 and 0.99 — no row needed a retry or fell below the review threshold.

---

## 7. Vector store and end-to-end flow

**Prompt:** Create `src/tools/transaction_store.py` to embed and persist transactions in ChromaDB (disk-persisted at `data/chroma_db`), with functions for semantic search, category filtering, and clearing the DB. Update the flow to load → categorize → store → run 3 test semantic queries ("coffee and food purchases", "transport and travel", "subscriptions and streaming"). Run it.

**Built:** `store_batch()`, `query_similar()`, `get_by_category()`, `clear_db()`. Each transaction's embedded text combines description + category (e.g. `"NETFLIX.COM | category: SUBSCRIPTION | type: debit"`) so category-level semantic queries work even when the merchant name alone wouldn't match. Rewired `src/cli.py` to run the full pipeline and print query results.

**Outcome:** First run downloaded Chroma's default embedding model (~79MB, one-time). All 3 test queries returned correctly ranked, relevant results (e.g. all 5 FOOD transactions for the coffee/food query, Starbucks ranked first).

---

## 8. Pattern detector agent

**Prompt:** Create `src/agents/pattern_detector.py`: load all transactions from ChromaDB, use Claude Haiku 4.5 to analyze spending patterns (top 3 categories, transactions >2x their category average, recurring charges, total spent vs. received), return a `SpendingReport` Pydantic model, mark anomalies `is_anomaly=True` back in ChromaDB, print a clean summary. Run it.

**Built:** All aggregation (category totals, the >2x anomaly check, recurring-charge grouping, spent/received totals) computed deterministically in Python — the LLM is used only to write the narrative `summary` field, so the numeric findings can't be corrupted by model arithmetic errors. Added `get_all()` and `mark_anomalies()` to `transaction_store.py` to support this.

**Outcome:** Report matched hand-verified numbers exactly: Shopping ($318.45) > Food ($175.43) > Transport ($133.85); one anomaly detected (Whole Foods, $87.32, 2.5x the FOOD category average) and correctly flagged in ChromaDB; no recurring charges (all descriptions in the sample data are unique); net +$3198.80.

---

## 9. Query agent and interactive CLI

**Prompt:** Create `src/agents/query_agent.py`: take a natural-language question, retrieve relevant transactions from ChromaDB, answer via Claude Haiku 4.5 using only the retrieved context, return the answer plus which transactions were used plus a confidence score, with comments explaining the RAG + agentic design. Wire everything into `src/cli.py` as an interactive CLI: run the full pipeline and report on startup, then loop on user questions until `exit`. Demo with 3 questions.

**Built:** `ask_question()` retrieves top-10 semantically relevant transactions, then forces a tool call so the model must return an answer, the ids of transactions it actually used, and a confidence score — ids are resolved defensively against the retrieved set rather than trusted, since a model can still hallucinate an id even under a constrained schema. `src/cli.py` now runs the pipeline once on startup and then loops on `input()` until `exit`/`quit`.

**Outcome:** Demo answered "how much did I spend on food?", "show me my biggest expenses", and "any suspicious transactions?" correctly overall, but surfaced two real gaps, both fixed in-session:
- The query agent's retrieval context didn't include the `is_anomaly` flag `pattern_detector` had already set, so it initially missed flagging the Whole Foods anomaly when asked about suspicious activity — fixed by including `is_anomaly` in the context passed to the model.
- The food-spending answer had a $1 arithmetic slip ($176.43 vs. the correct $175.43), since that agent lets the LLM sum retrieved amounts itself rather than computing totals in code like `pattern_detector` does. Flagged as a known limitation — not fixed, since fixing it changes the agent's answer-generation design (computing sums over `used_transactions` in code) and wasn't asked for at the time.

---

## 10. MCP server

**Prompt:** Create `src/mcp/server.py` using FastMCP exposing: resource `finance://summary` (latest spending report as text), resource `finance://transactions` (all transactions as JSON), tool `query_finances` (natural-language Q&A via the query agent), tool `get_category_total` (total spent in a category) — with comments on why exposing financial data via MCP is useful. Create this `prompts.md`. Run the server and confirm it starts without errors.

**Built:** `src/mcp/server.py` with the two resources and two tools described above. Refactored `pattern_detector.py` to split `format_report()` (returns a string) out of `print_report()` (prints it), so the MCP resource could reuse the same formatting logic instead of duplicating it. This file.

**Outcome:** Server started cleanly in the background (stdio transport, stayed running with no errors or traceback) and was shut down after verification.
