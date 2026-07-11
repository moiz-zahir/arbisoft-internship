# Hallucination Log

## 1. What "hallucination" means in LLMs

A hallucination is output that is fluent and confident-sounding but not
actually supported by the source material (or reality) - the model fills a
gap with the most *statistically plausible* continuation rather than a
*verified* one, because that's fundamentally what a language model does:
predict likely next tokens, not look up facts. Hallucinations are
dangerous precisely because they're stylistically indistinguishable from
correct answers - nothing in the model's tone signals "I made this part
up."

Two flavors show up in this project:

- **Content hallucination** - a fact, number, or claim that isn't true or
  isn't in the given source.
- **Structural hallucination** - output that doesn't match the format it
  was asked for (wrong type, missing field, malformed JSON), which is a
  smaller-scale version of the same problem: the model produced something
  plausible-looking instead of something correct.

## 2. Where the LLM could hallucinate in our RAG demo

[`src/rag_demo.py`](src/rag_demo.py) asks the model to answer "using only
the context below," but that's an instruction, not a guarantee - the model
is still free to draw on its own training data. Two concrete ways this
could go wrong:

1. **Ungrounded elaboration from memorized training data.** "Attention Is
   All You Need" (`docs/machinelearning1.pdf`) is one of the most-cited
   papers in ML and near-certainly appeared in the model's training data.
   Nothing stops `generate_answer()` from quietly blending details it
   remembers about the paper - a specific hyperparameter, a claim from a
   section we didn't retrieve - into an answer that reads as if it came
   from the 3 supplied chunks. For an obscure or private document this
   failure mode is less likely (the model has nothing to fall back on),
   which is actually an argument *for* testing RAG on non-famous documents.

2. **Confident answers from weak retrieval.** [`embedding_comparison.md`](embedding_comparison.md)
   shows real retrieved chunks scoring as low as 0.33-0.38 relevance for
   some questions - i.e. the top-3 chunks ChromaDB returns aren't always
   strongly on-topic. `generate_answer()` has no check for "is this
   context actually sufficient" before generating; if the retrieved
   chunks are only tangentially related, the model can still produce a
   fluent, confident-sounding answer that papers over the gap rather than
   saying it doesn't know. (The prompt does ask the model to say when the
   answer isn't in the context, which mitigates but does not eliminate
   this - see §5.)

3. **Retrieval/generation traceability gap.** `answer_question()` prints
   the chunks ChromaDB *retrieved*, not the chunks the model actually
   *used* to write its answer. If the model ignores the weakest of the
   3 chunks and instead answers from its own knowledge, the printed
   "chunks used" list would still look like solid grounding when it isn't
   - the demo has no mechanism to verify the answer actually traces back
   to the cited text.

## 3. How structured output validation catches bad output

[`src/structured_output.py`](src/structured_output.py) defines
`ResearchSummary` as a Pydantic model with `title: str`,
`key_points: list[str]` (`min_length=3`), `confidence_score: float`
(`ge=0.0, le=1.0`), and `limitations: str`. Every model response is parsed
with `json.loads()` and then run through `ResearchSummary.model_validate()`
before it's trusted anywhere else in the program. This catches:

- Malformed or non-JSON output (`json.JSONDecodeError`)
- Missing or renamed fields
- Wrong types (e.g. `confidence_score` returned as a string)
- `key_points` with fewer than 3 items
- `confidence_score` outside `[0, 1]` (e.g. `1.5`, `-0.1`)

These exact failure modes are what [`tests/test_structured_output.py`](tests/test_structured_output.py)
asserts against.

**Important limitation:** validation checks *shape*, not *truth*. A
`confidence_score` of `0.92` is only guaranteed to be a float between 0
and 1 - Pydantic has no way to know whether 0.92 reflects genuine
calibrated certainty or is itself a plausible-looking number the model
invented. Likewise, a `key_points` entry that is a well-formed string can
still be a fabricated fact. Schema validation is necessary but not
sufficient protection against hallucination - it guarantees the output is
usable, not that it's correct.

## 4. How the retry loop handles it

`request_research_summary()` in `src/structured_output.py` wraps the
call/validate step in a loop (`MAX_RETRIES = 3`, so up to 4 attempts
total). On `JSONDecodeError` or `ValidationError`:

1. The model's bad response is appended to the conversation as an
   `assistant` message (so it has its own failed attempt in context).
2. A `user` message is appended containing the **exact validation error
   text** (e.g. "List should have at least 3 items after validation,
   not 2") along with a repeated instruction to return corrected JSON
   only.
3. The loop calls the model again with this extended conversation.

This is targeted self-correction, not a blind retry: telling the model
specifically what broke ("your `key_points` array was too short") gives
it the exact information needed to fix that one thing, which is far more
likely to succeed than resending the same prompt unchanged. If all 4
attempts fail validation, the function raises `RuntimeError` with the
last error rather than letting bad data through - the program fails loudly
instead of silently accepting malformed output.

In the run recorded in this project, the first attempt already passed
validation, so the retry path exists but wasn't exercised - see attempt
log in the terminal output from running the script.

## 5. What we did to reduce hallucination risk in this project

- **Grounding via retrieval (RAG):** instead of asking the model to
  answer from memory, `generate_answer()` supplies the top-3 retrieved
  chunks as context and explicitly instructs it to answer only from that
  context and to say so if the answer isn't present - narrowing the
  space in which the model can improvise.
- **Traceable output:** `answer_question()` prints which chunks were
  retrieved and used alongside the answer, so a human can spot-check
  whether the answer is actually supported by the cited text rather than
  trusting it blindly.
- **Overlapping chunks:** `chunk_text()` uses a 50-character overlap
  between 500-character chunks specifically so a sentence or idea that
  falls on a chunk boundary isn't split and left incomplete - incomplete
  context is a common trigger for the model "filling in" the missing
  piece itself.
- **Empirically checking retrieval quality:** `src/compare_embeddings.py`
  measures retrieval relevance across two embedding models rather than
  assuming the first model chosen retrieves good context. Since weak
  retrieval is one of the main hallucination triggers in RAG (§2.2), this
  gives us actual data on how often that risk shows up instead of
  guessing.
- **Schema validation with a self-correcting retry loop:** `structured_output.py`
  guarantees that whatever gets written to `output.json` is at minimum
  well-formed and within the declared constraints (see §3-4), so
  malformed output can't silently propagate downstream even if it can't
  catch every kind of content hallucination.

**Not addressed by this project (worth flagging, not solved here):** we
don't verify that generated answers are *entailed* by the retrieved
chunks (no fact-checking pass), we don't lower `temperature` to reduce
generation variance, and `confidence_score` in `ResearchSummary` is
self-reported by the model with no external calibration - all three would
be reasonable next steps if this moved beyond a demo.
