# Week 2 Prompt Log

## 1. Project setup with uv

**Prompt:**
> Set up a Python project in this folder using uv. Run uv init, then uv add openai python-dotenv. Also create a .gitignore file that ignores .env, .venv/, and __pycache__. Show me each step as you run it.

**What was built:** Initialized the project with `uv init` (creating `pyproject.toml`, `README.md`, `.venv`), installed `openai` and `python-dotenv` with `uv add`, and updated the auto-generated `.gitignore` to also ignore `.env`.

**Outcome:** `uv` wasn't on PATH but was already installed as a Python package, so `python -m uv` was used instead. Installing dependencies initially failed due to a OneDrive hardlink conflict (the project lives in a synced OneDrive folder); fixed by re-running with `--link-mode=copy`. Project set up successfully.

## 2. .env and .env.example

**Prompt:**
> Create a .env.example file with these two lines: OPENROUTER_API_KEY=..., ANTHROPIC_API_KEY=... Then create a real .env file with the same structure but leave the values empty for now.

**What was built:** Flagged that the request had the two files backwards — `.env.example` is meant to be committed to git and should only hold placeholders, while `.env` (gitignored) should hold real secrets. After confirming with the user, created `.env.example` with empty placeholder values and `.env` with the real API keys.

**Outcome:** Files created correctly with secrets kept out of the committable file.

## 3. Command-line chat app

**Prompt:**
> Create a src folder. Inside it create chat.py - a command line chat app that: 1. Loads OPENROUTER_API_KEY from .env 2. Uses the openai Python client pointed at https://openrouter.ai/api/v1 3. Uses model anthropic/claude-3.5-haiku 4. Keeps the full conversation history in a list so the model remembers earlier messages 5. Lets me type 'exit' to quit. Add comments explaining how the conversation memory works and why we resend history every time.

**What was built:** `src/chat.py` — loads the API key, points the OpenAI client at OpenRouter, maintains a `history` list of every user/assistant message, resends the full list on each call, and exits on `exit`. Comments explain that the chat API is stateless and why the full history must be resent each turn.

**Outcome:** `anthropic/claude-3.5-haiku` returned a 404 (retired from OpenRouter's catalog). Checked OpenRouter's live model list and swapped to `anthropic/claude-haiku-4.5` with user confirmation. Verified end-to-end with a live test (told it a name, asked it back) — the model correctly recalled it, confirming memory worked.

## 4. Model comparison script

**Prompt:**
> Create src/compare_models.py that: 1. Loads OPENROUTER_API_KEY from .env 2. Sends this exact prompt to 3 different models via OpenRouter: "Explain what a neural network is in 3 sentences for a first-year CS student" 3. Times how long each model takes to respond 4. Prints each model name, time taken in seconds, and the full response. Use these 3 models: anthropic/claude-haiku-4.5, openai/gpt-4o-mini, meta-llama/llama-3.1-8b-instruct. Then run it and show me all 3 responses.

**What was built:** `src/compare_models.py` — sends the same prompt to all three models, times each call with `time.perf_counter()`, and prints the model name, elapsed time, and full response.

**Outcome:** Verified all three model IDs existed on OpenRouter before running. Ran successfully: `anthropic/claude-haiku-4.5` (5.41s), `openai/gpt-4o-mini` (2.74s), `meta-llama/llama-3.1-8b-instruct` (26.19s). All three responses printed in full.

## 5. Model comparison table

**Prompt:**
> Create model_comparison.md with a table comparing the 3 models we just tested. Include these columns: Model, Speed, Quality Rating out of 5, Best Use Case. Fill in the speed from our test results. Leave quality rating and best use case for me to fill in manually.

**What was built:** `model_comparison.md` with a table pre-filled with the measured speeds; Quality Rating and Best Use Case left blank.

**Outcome:** File created as requested. User later filled in the Quality Rating and Best Use Case columns themselves.

## 6. LLM concepts Q&A

**Prompt:**
> Ask me these 4 questions one at a time and wait for my answer before asking the next: 1. What do transformers and attention mean intuitively, no math? 2. What is tokenization, context window, temperature and sampling? 3. What are embeddings, give a simple analogy? 4. What is the difference between system prompt, few-shot prompting and chain of thought, with a tiny example of each?

**What was built:** Asked each question one at a time, waiting for an answer before moving to the next.

**Outcome:** User answered all four questions accurately and with good examples, demonstrating solid understanding of the concepts.

---

## Concepts Learned

**Transformers & Attention**
Attention lets a model weigh which parts of the input matter most for understanding a given word or token — e.g., resolving that "bank" means something different depending on whether "river" or "money" appears nearby. Transformers are the neural network architecture built around this mechanism, processing an entire sequence in parallel (rather than word-by-word) and using attention to let every token look at every other token to build context-aware meaning.

**Tokenization, Context Window, Temperature & Sampling**
- *Tokenization*: models don't operate on whole words, but on sub-word chunks called tokens (e.g., "unhappy" → "un" + "happy"), roughly 1 token ≈ 0.75 words.
- *Context window*: the maximum number of tokens the model can attend to at once — its working memory. Once a conversation exceeds it, the earliest content drops out.
- *Temperature*: controls randomness in output. Low (near 0) = deterministic, repetitive, best for coding/factual tasks. High (near 1+) = more varied and creative, better for creative writing.
- *Sampling*: the method used to actually pick the next token from the model's output probability distribution; temperature reshapes that distribution before sampling occurs.

**Embeddings**
Embeddings represent words, sentences, or concepts as vectors (lists of numbers) positioned in a high-dimensional space, such that semantically similar items land close together (e.g., "king" near "queen", both far from "rocket"). This numeric "map of meaning" is what lets models compare, search, and reason about similarity.

**Prompting Styles**
- *System prompt*: sets the model's overall behavior/persona before the conversation starts (e.g., "You are a helpful assistant that only answers in bullet points.").
- *Few-shot prompting*: gives the model a handful of input/output examples so it infers the pattern before answering a new case (e.g., showing English→French translation pairs before asking for one more).
- *Chain of thought*: instructs the model to reason step by step before giving a final answer, improving accuracy on multi-step problems (e.g., working through an arithmetic word problem step by step before stating the answer).
