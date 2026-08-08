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

---

# Week 6 Prompt Log

## 11. Test suite

**Prompt:** Create a `tests/` folder with `test_csv_ingestion.py`, `test_transaction_models.py`, and `test_transaction_store.py`, plus a `conftest.py` with shared fixtures for sample transactions. Run `pytest tests/ -v` and show the coverage report. Add `pytest-cov` via `uv add --dev pytest-cov` and run with `--cov=src --cov-report=term-missing`.

**Built:** `conftest.py` with `sample_csv_path`, `sample_transactions`, `sample_batch`, and a `chroma_test_dir` fixture that monkeypatches `transaction_store.CHROMA_DB_PATH` to a `tmp_path` so store tests never touch the real `data/chroma_db`. 15 tests total across the three files, covering batch parsing/totals, malformed-row and missing-column handling, model validation edges (bad `type`, out-of-range `confidence_score`, missing required fields), and the store's save/query/filter/clear operations. Added `[tool.pytest.ini_options]` (`testpaths`, `pythonpath`) to `pyproject.toml` for reliable import resolution.

**Outcome:** All 15 tests passed. Coverage came back at 100% for `models/transaction.py`, 79%/74% for `transaction_store.py`/`csv_ingestion.py`, and 0% for the agent/CLI/MCP modules (expected — those need live LLM calls and weren't in scope for this suite). 23% overall, consistent with testing exactly the three modules asked for.

---

## 12. Error handling

**Prompt:** Add proper error handling throughout: `csv_ingestion.py` (clear `FileNotFoundError`/`ValueError` for missing/empty files, log skipped-row counts), `categorizer.py` (catch unreachable-API errors and retry exhaustion, fall back to `OTHER`/0.0 confidence instead of crashing, add a progress indicator), `query_agent.py` (helpful message on empty retrieval, fallback answer if the LLM call fails), `cli.py` (reprompt on a bad CSV path, skip empty questions, catch Ctrl+C gracefully). Add 3 more tests for the new `csv_ingestion.py` error cases and confirm the full suite still passes.

**Built/verified:** By the time this prompt was being worked, an external process had already applied every one of these changes to `csv_ingestion.py`, `categorizer.py`, `cli.py`, and the 3 new tests in `test_csv_ingestion.py` (`test_missing_file_raises_file_not_found_error`, `test_empty_file_raises_value_error`, `test_skipped_row_count_is_logged`) — confirmed by reading each file in full against the request. The one piece not yet done, `query_agent.py`'s error handling (empty-retrieval message, LLM-failure fallback), was also found already implemented on inspection. No new code was needed; the work was to verify each requirement against the actual file contents rather than assume.

**Outcome:** All 18 tests (15 previous + 3 new) passed.

---

## 13. Full CLI demo and prompt log update

**Prompt:** Run the full CLI end-to-end against the sample CSV with 5 demo questions ("how much did I spend on food?", "what are my biggest expenses?", "any suspicious transactions?", "how much did I earn this month?", "what did I spend on subscriptions?"), show the full output including the spending report, then update this file with the Week 6 prompts.

**Built:** Ran `src/cli.py` piping the 5 questions plus `exit` into stdin.

**Outcome:** Full pipeline (load → categorize → store → pattern report) ran cleanly, then all 5 questions were answered with sources and confidence scores. Two pre-existing rough edges resurfaced, both informational rather than newly introduced:
- The food-spending answer repeated the earlier-known ~$1 LLM arithmetic slip ($176.43 vs. the correct $175.43) — the query agent still sums retrieved amounts itself instead of computing totals in code.
- The "biggest expenses" answer's numbered list put Whole Foods ($87.32) at #1 and Best Buy ($199.99) at #2, an ordering inconsistency, even though the same answer's prose correctly called Best Buy "the highest single transaction."
"How much did I earn this month?" correctly excluded the $34.50 Zara refund from earnings (a sensible semantic call, not an error) and summed the two INCOME transactions to $3,850. Subscriptions total ($29.97) was exact.

---

## 14. Summarizer agent and CLI menu

**Prompt:** Add `src/agents/summarizer.py`: take a natural-language time period ("last week", "this month", "July"), retrieve matching transactions from ChromaDB, use Claude Haiku 4.5 to write a friendly financial-advisor-style summary, return a `SummaryReport` (`period`, `total_spent`, `total_earned`, `savings_rate` computed in Python, `top_insight`, `recommendations` (3), `full_summary` from the LLM), with comments on why totals are computed in Python. Add a CLI menu after the spending report: "(1) Ask a question (2) Get spending summary (3) Exit", where option 2 prompts for a period and runs the summarizer. Demo it and update this file.

**Built:** `resolve_period()` parses the period into a concrete date range using plain `datetime`/`calendar` arithmetic (handles "today", "this/last week", "this/last month", month names, and "YYYY-MM") rather than asking the LLM to interpret it — a wrong date boundary would silently drop a transaction from the summary instead of raising an error, the same reasoning already applied to money totals in `pattern_detector.py`. `generate_summary()` filters transactions by that range, computes `total_spent`/`total_earned`/`savings_rate` in Python, then forces a tool call so the LLM returns only `top_insight`, exactly 3 `recommendations`, and `full_summary` — never the numbers themselves. Falls back to a totals-only `SummaryReport` if the LLM call fails, matching the fallback pattern already used in `query_agent.py`. Replaced the old single-purpose question loop in `src/cli.py` with a `menu_loop()` offering the 3 options, backed by `ask_a_question()` and `get_spending_summary()`.

**Outcome:** Ran the full CLI, selected option 2, entered "July" (the month the sample data is actually in — "this month"/"last week" would resolve against today's real date and correctly return zero transactions, since the sample statement predates them). The Python-computed fields were exact: $685.70 spent, $3,884.50 earned, 82.3% savings rate. The LLM's free-text `full_summary`, however, restated the FOOD and TRANSPORT category subtotals slightly wrong ("$175.63" and "$134.85" vs. the actual $175.43 and $133.85) — a small arithmetic slip in prose that didn't touch any of the trustworthy numeric fields, which is exactly the failure mode the Python-totals design is meant to contain.

---

## 15. Multi-model routing (Ollama + OpenRouter)

**Prompt:** Add multi-model routing: `categorize_with_local_model()` in `categorizer.py` using Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`, `api_key="ollama"`, model `llama3`) with automatic fallback to Claude Haiku 4.5/OpenRouter on any Ollama error, with a comment on why local categorization matters (speed, cost, privacy). At CLI startup, ask "local or cloud" and route accordingly, printing "Ollama not available, falling back to OpenRouter" if local was requested but Ollama isn't running. Create `src/agents/model_router.py`: pings `http://localhost:11434` to decide `"local"` vs `"cloud"`, and logs which model served each categorization. Demo the router detecting availability and routing, then update this file.

**Built:** `model_router.py` — `is_ollama_available()` does a live GET ping (not a cached check, since Ollama can start/stop between runs) with a 2-second timeout, catching `requests.RequestException`; `get_available_backend()` returns `"local"`/`"cloud"`; `log_model_used()` logs which backend handled each transaction. In `categorizer.py`: extracted the message-building shared by both paths into `_build_messages()`; added `_get_local_client()` (Ollama's OpenAI-compatible client) and `categorize_with_local_model()`, which tries the local model first and falls through to the existing `categorize_transaction()` (cloud, with its own 3-retry/OTHER-fallback logic) on any `OpenAIError`, missing tool call, or validation failure. `categorize_batch()` gained a `use_local` flag routing each transaction accordingly, and both the local and cloud success paths call `log_model_used()`. In `cli.py`: `choose_model_backend()` asks the user's preference at startup and only honors "local" if `model_router.get_available_backend()` confirms Ollama is actually reachable, printing the fallback message and returning to cloud otherwise; `run_pipeline()` now takes and reports which model is in use.

**Outcome:** Ollama was not running in this environment (confirmed via a direct ping before implementing, so the demo's fallback path is a real result, not a simulated one). Standalone check: `is_ollama_available()` → `False`, `get_available_backend()` → `"cloud"`. Full CLI run choosing "local" at the prompt correctly printed "Ollama not available, falling back to OpenRouter" and completed categorization on the cloud path, producing the same spending report as prior runs ($685.70 spent, $3,884.50 received). All 18 existing tests still passed after the refactor.

---

## 16. Output validation and schema guards

**Prompt:** In `query_agent.py`, fix the known arithmetic bug by computing the actual sum of used-transaction amounts in Python and post-processing the LLM's answer text to replace any dollar amounts with the correct value. Create `src/tools/validation_guards.py` with `validate_no_negative_amounts`, `validate_date_format`, `validate_confidence_scores` (logs, doesn't raise), and `validate_category_coverage` (raises above 20% OTHER). Wire the guards into `cli.py`'s pipeline after categorization, printing a warnings summary. Add 4 tests in `tests/test_validation_guards.py`. Confirm the full suite passes, then update this file.

**Built:** `_correct_dollar_amounts()` in `query_agent.py` computes `sum(t.amount for t in used_transactions)` in Python, then regex-scans the LLM's answer for `$X.XX`-style figures: any figure matching a known per-transaction amount is left alone (a legitimate citation), and anything else is overwritten with the correct computed total — this targets the actual failure mode already seen in this project (individual amounts cited correctly, but the LLM's own addition wrong), without touching correct line-item citations. `validation_guards.py`: 3 guards raise `ValueError` (negative debit amounts, malformed dates, >20% `OTHER` category coverage), 1 logs via `logger.warning` (confidence below 0.7) per the spec's explicit distinction; `run_guards(batch)` runs all four and catches any raised `ValueError` into a returned warnings list rather than propagating, so a data-quality issue never crashes the pipeline. `cli.py`'s `run_pipeline()` calls `run_guards()` right after `categorize_batch()` and prints either "All validation checks passed." or a bulleted warnings list.

**Outcome:** Verified the arithmetic fix directly against the exact case that was previously wrong: "how much did I spend on food?" now answers **$175.43** (the correct sum) instead of the earlier $176.43, with every individual line item ($45.21, $12.85, $87.32, $24.30, $5.75) untouched. All 22 tests passed (18 previous + 4 new guard tests).

---

## 17. Documentation: usage guide, architecture, and README

**Prompt:** Create `docs/USAGE.md` (what the project does, prerequisites, installation, running with sample data, example questions, own-CSV format, known limitations including LLM arithmetic in narrative text and "DuckDuckGo limitations etc."), `docs/ARCHITECTURE.md` (ASCII system diagram, one-sentence-per-file, data flow, design-decision rationale), and a polished `README.md` linking both. Run the full test suite one final time, then update this file.

**Built:** `docs/USAGE.md` and `docs/ARCHITECTURE.md` covering exactly what was asked, plus a rewritten `README.md` with a quick-start and links to both docs. One deliberate deviation: the prompt's example limitation "DuckDuckGo limitations etc." doesn't apply — this project has no DuckDuckGo integration anywhere in the codebase — so it was omitted rather than fabricated, and replaced with limitations that actually exist here: LLM arithmetic in narrative text (with the caveat that the query agent's fix is targeted, not universal — `pattern_detector` and `summarizer` prose isn't corrected the same way), local-model (`llama3`) quality and tool-calling reliability, non-exhaustive semantic retrieval (fixed top-10), limited time-period parsing in `summarizer.py`, and no real bank integration.

**Outcome:** All 22 tests passed on the final run. Documentation written to accurately reflect the actual current codebase (verified by reading every source file and the live `cli.py`/`mcp/server.py` behavior before writing, rather than describing an idealized version).
