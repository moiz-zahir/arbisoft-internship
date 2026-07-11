# Week 3 Prompt Log

A record of every prompt given during Week 3, what was built in response, and the outcome.

---

### 1. Set up the project

**Prompt:** "Set up a Python project in this folder using uv. Run uv init, then add these dependencies: chromadb, openai, python-dotenv, pydantic, pypdf. Also create a .gitignore that ignores .env, .venv/, and __pycache__. Show each step as you run it."

**Built:** Installed `uv` (not present on the machine), ran `uv init` to scaffold the project (`pyproject.toml`, `main.py`, `.python-version`, git repo), then `uv add chromadb openai python-dotenv pydantic pypdf`. Extended the `.gitignore` that `uv init` generated to add `.env`.

**Outcome:** Project initialized successfully. Hit one snag: `uv add` failed to hardlink packages because the project lives inside a OneDrive-synced folder; fixed by setting `UV_LINK_MODE=copy`. All 5 dependencies plus 49 transitive packages installed into `.venv`.

---

### 2. Create the `.env` file

**Prompt:** "Create a .env file with these placeholders: OPENROUTER_API_KEY=your_openrouter_key_here. Then copy the real OpenRouter API key from the week2 .env file into this .env file. Make sure .env is gitignored."

**Built:** Located `../week2/.env`, read the real `OPENROUTER_API_KEY` value, and wrote it into a new `.env` in this project (leaving out the unrelated `ANTHROPIC_API_KEY` that was also present in the week2 file, since it wasn't requested).

**Outcome:** `.env` created with the real key. Verified with `git check-ignore -v .env` that it's correctly excluded from version control.

---

### 3. Build the RAG demo

**Prompt:** "Create src/rag_demo.py that does the following: 1. Loads all PDFs from the docs/ folder using pypdf 2. Splits the text into chunks of 500 characters with 50 character overlap 3. Creates a ChromaDB collection and stores the chunks with embeddings using OpenAI's text-embedding-3-small model via OpenRouter 4. Takes a user question, finds the 3 most relevant chunks from ChromaDB 5. Sends those chunks plus the question to claude-haiku-4.5 via OpenRouter to generate an answer 6. Prints the answer and which chunks were used to answer it. Add clear comments explaining what RAG means and why each step exists. Then run it with the question: 'What is attention mechanism in transformers?'"

**Built:** [`src/rag_demo.py`](src/rag_demo.py) — loads PDFs from `docs/` with `pypdf`, chunks text (500 chars, 50 overlap), embeds chunks with `openai/text-embedding-3-small` via OpenRouter and stores them in an in-memory ChromaDB collection, retrieves the top-3 chunks for a question, and sends them plus the question to `anthropic/claude-haiku-4.5` via OpenRouter to generate a grounded answer. Verified OpenRouter's embeddings endpoint and the correct Claude Haiku 4.5 model ID via its docs before writing the code.

**Outcome:** Ran successfully against `docs/machinelearning1.pdf` (the "Attention Is All You Need" paper) and `docs/machinelearning2.pdf`. Correctly retrieved 3 relevant chunks about self-attention/multi-head attention and produced an accurate, context-grounded answer, printing both the answer and the source chunks used.

---

### 4. Compare embedding models

**Prompt:** "Create src/compare_embeddings.py that: 1. Loads the same PDFs from docs/ folder 2. Takes 3 test questions about the content 3. Embeds the chunks using two different embedding models: openai/text-embedding-3-small and a second model available on OpenRouter 4. For each question, retrieves the top 3 chunks from each embedding model 5. Prints a comparison showing which chunks each model retrieved and how relevant they seem 6. Saves a summary to embedding_comparison.md explaining which model performed better and why. Run it and show me the results."

**Built:** [`src/compare_embeddings.py`](src/compare_embeddings.py) — reuses `load_pdfs`/`chunk_text` from `rag_demo.py`, embeds the same chunks with `openai/text-embedding-3-small` and `baai/bge-large-en-v1.5` (chosen after checking OpenRouter's available embedding models), stores each in its own ChromaDB collection, runs 3 test questions relevant to the actual PDF content (attention mechanism, multi-head attention, dilated convolutions), and scores retrieved chunks with a model-agnostic keyword-overlap relevance proxy since raw distances aren't comparable across embedding spaces. Writes [`embedding_comparison.md`](embedding_comparison.md).

**Outcome:** Ran successfully. `baai/bge-large-en-v1.5` scored marginally higher (0.62 vs 0.60 average relevance) and surfaced one chunk (the `MultiHead(Q,K,V)` formula) that Model A missed. Also surfaced and fixed a Windows console encoding bug (crashed printing `Ł` from a paper author's name) in both this script and `rag_demo.py`.

---

### 5. Structured output with validation

**Prompt:** "Create src/structured_output.py that: 1. Loads OPENROUTER_API_KEY from .env 2. Defines a Pydantic model called ResearchSummary with these fields: title: str, key_points: list[str] (minimum 3 points), confidence_score: float (between 0 and 1), limitations: str 3. Sends a prompt to claude-haiku-4.5 via OpenRouter asking it to summarize the attention mechanism as JSON matching that schema 4. Parses and validates the response with Pydantic 5. If validation fails, retries up to 3 times with an error message telling the model what went wrong 6. Saves the validated output to output.json 7. Prints the validated result clearly. Add comments explaining why we validate LLM output and what could go wrong without it. Then run it and show me the output."

**Built:** [`src/structured_output.py`](src/structured_output.py) — defines the `ResearchSummary` Pydantic model with the required field constraints, prompts Claude Haiku 4.5 via OpenRouter for a JSON summary of the attention mechanism, parses/validates the response, and on failure feeds the exact validation error back to the model and retries (up to 3 retries, 4 attempts total). Saves the validated result to [`output.json`](output.json).

**Outcome:** Ran successfully — validation passed on the first attempt (confidence_score 0.92, 6 key points), so the retry path exists in the code but wasn't exercised in this run. Result printed clearly and saved to `output.json`.

---

### 6. Tests for structured output

**Prompt:** "Create tests/test_structured_output.py with these pytest tests: 1. Test that a valid ResearchSummary object passes validation 2. Test that validation fails when key_points has fewer than 3 items 3. Test that validation fails when confidence_score is greater than 1 4. Test that validation fails when confidence_score is less than 0 5. Test that output.json exists and contains valid JSON after running structured_output.py. Run pytest tests/ -v and show me the results."

**Built:** [`tests/test_structured_output.py`](tests/test_structured_output.py) with the 5 requested tests. Added `pytest` as a dev dependency (`uv add --dev pytest`) and configured `pythonpath = ["src"]` / `testpaths = ["tests"]` in `pyproject.toml` so tests can import from `src/` without packaging it. Test 5 runs `structured_output.py` as a real subprocess (live API call) rather than mocking it.

**Outcome:** All 5 tests passed (`5 passed in 17.98s`).

---

### 7. Hallucination log

**Prompt:** "Create hallucination_log.md that documents: 1. What hallucination means in LLMs 2. At least 2 examples of where the LLM could hallucinate in our RAG demo 3. How our structured output validation catches bad output 4. How the retry loop handles it 5. What we did to reduce hallucination risk in this project"

**Built:** [`hallucination_log.md`](hallucination_log.md), covering the definition of hallucination, three concrete risks specific to `rag_demo.py` (ungrounded elaboration from the model's memorized training data on a famous paper, confident answers built on weak retrieval — citing the actual 0.33-0.38 relevance scores from `embedding_comparison.md`, and a retrieval/generation traceability gap), how `ResearchSummary` validation catches structural problems (while being explicit that it can't catch content hallucination), how the retry loop feeds back the exact validation error, and a summary of the concrete mitigations used in this project plus an honest "not addressed" list (no entailment checking, no temperature tuning, self-reported confidence score).

**Outcome:** Written and delivered directly (no code run for this step).

---

### 8. This prompt log

**Prompt:** "Create prompts.md in the root of this project. Add an entry for every prompt I gave you during Week 3. Each entry should have the prompt, what you built, and the outcome. Also add a section summarizing what RAG is, what vector databases do, and why structured outputs matter."

**Built:** This file, plus the concepts summary below.

**Outcome:** Written and delivered directly.

---

## Concepts

### What is RAG?

**Retrieval-Augmented Generation** is a technique for getting accurate,
grounded answers out of an LLM about content it wasn't trained on (or
might have forgotten/misremembered) — your own documents, recent data,
private data, etc. Instead of relying purely on what the model memorized
during training, RAG retrieves the most relevant pieces of your actual
source material at query time and hands them to the model as context,
then asks it to answer *from that context*. This is what `rag_demo.py`
does: chunk documents → embed and store them → retrieve the top-k
relevant chunks for a question → generate an answer grounded in those
chunks. It matters because it lets an LLM answer correctly about content
outside its training data, reduces (though doesn't eliminate — see
`hallucination_log.md`) hallucination by anchoring answers to real text,
and makes answers auditable, since you can point to exactly which source
chunks were used.

### What do vector databases do?

A vector database (ChromaDB, in this project) stores **embeddings** —
numeric vectors that represent the meaning of a piece of text — alongside
the original text and metadata, and is built to efficiently find the
vectors closest in meaning to a given query vector ("nearest neighbor
search"). This is what makes retrieval in RAG fast and semantic rather
than a slow, literal keyword search: a question about "attention" can
retrieve a chunk that says "self-attention mechanism" even without an
exact word match, because their embeddings land near each other in vector
space. `compare_embeddings.py` demonstrated why the choice of embedding
model matters here — different models place text differently in vector
space, so retrieval quality (and therefore the quality of the final
answer) depends directly on which embedding model generated the vectors
the database is searching.

### Why do structured outputs matter?

LLMs generate free-form text by default, which is a poor fit for feeding
into other code — an application can't safely do `data["confidence_score"]`
on an unvalidated string. Structured output means constraining and then
verifying that a model's response matches an exact schema (types, required
fields, value ranges) before anything downstream trusts it. In this
project, `structured_output.py` uses a Pydantic model (`ResearchSummary`)
to guarantee that whatever gets used or saved is well-formed: the right
fields, the right types, `key_points` with at least 3 entries,
`confidence_score` actually between 0 and 1. Combined with a retry loop
that feeds validation errors back to the model, this turns an LLM from a
producer of "probably fine" text into a reliable component in a larger
program — while still being honest that schema validation guarantees
*shape*, not *truth* (see `hallucination_log.md` §3).
