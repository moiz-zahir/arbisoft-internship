import json
import logging
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from src.tools.transaction_store import query_similar

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRIEVAL_COUNT = 10


class UsedTransaction(BaseModel):
    id: str
    date: str
    description: str
    amount: float
    category: str


class QueryAnswer(BaseModel):
    question: str
    answer: str
    used_transactions: list[UsedTransaction]
    confidence_score: float = Field(ge=0.0, le=1.0)


# What the model's tool call actually returns: ids, not full transaction
# records. Validated here before we resolve those ids back to real
# transaction data, mirroring the two-stage validation used elsewhere in
# this project (constrained schema first, Pydantic second).
class _RawAnswer(BaseModel):
    answer: str
    used_transaction_ids: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": (
            "Answer the user's question using only the retrieved transactions "
            "provided in context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A plain-English answer to the user's question.",
                },
                "used_transaction_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The ids (from the provided list) of transactions actually used to compute the answer.",
                },
                "confidence_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": (
                        "Confidence that the retrieved transactions fully and "
                        "correctly answer the question, from 0 to 1."
                    ),
                },
            },
            "required": ["answer", "used_transaction_ids", "confidence_score"],
        },
    },
}


_DOLLAR_AMOUNT_RE = re.compile(r"\$([\d,]+\.\d{2})")


def _correct_dollar_amounts(answer: str, used_transactions: list[UsedTransaction]) -> str:
    """
    Replaces any dollar figure in the LLM's answer that isn't one of the
    actual retrieved transaction amounts with the correct, Python-computed
    sum of those transactions.

    This project has already hit a real case (see prompts.md) where an LLM
    asked to total several transactions in prose got the arithmetic wrong by
    about a dollar, even though every individual amount it cited was
    correct - it just added them up incorrectly. The fix isn't to ask the
    model to be more careful; it's to never trust it with the addition at
    all. A dollar amount that matches one of the known per-transaction
    amounts is a legitimate citation and is left alone; any other dollar
    figure in the text is assumed to be an LLM-computed aggregate (right or
    wrong) and is overwritten with the real total computed here in Python.
    """
    if not used_transactions:
        return answer

    computed_total = sum(t.amount for t in used_transactions)
    known_amounts = {round(t.amount, 2) for t in used_transactions}
    correct_str = f"{computed_total:.2f}"

    def _replace(match: re.Match) -> str:
        value = float(match.group(1).replace(",", ""))
        if round(value, 2) in known_amounts:
            return match.group(0)
        return f"${correct_str}"

    return _DOLLAR_AMOUNT_RE.sub(_replace, answer)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set - check your .env file")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def ask_question(question: str, n_results: int = RETRIEVAL_COUNT) -> QueryAnswer:
    """
    Answers a natural-language question about the user's transactions.

    This combines RAG with an agentic layer on top of it:
      - Retrieval: ChromaDB's semantic search pulls the transactions whose
        embedded text is most relevant to the question, so the model never
        has to see (or guess from memory) the entire transaction history -
        it only reasons over grounded, real data actually in the store.
      - Generation: Claude Haiku 4.5 answers using only that retrieved
        context, via a forced tool call so the response has to include the
        answer, its sources, and a confidence score in a fixed shape.
      - Agentic behavior: the model doesn't just summarize whatever came
        back from retrieval - it decides *which* of the retrieved
        transactions actually support its answer (used_transaction_ids) and
        self-assesses whether the retrieved evidence is sufficient
        (confidence_score). A low score is itself useful output: it signals
        that retrieval likely missed relevant transactions and the answer
        shouldn't be fully trusted.
    """
    retrieved = query_similar(question, n_results=n_results)
    if not retrieved:
        return QueryAnswer(
            question=question,
            answer="No relevant transactions found for that question.",
            used_transactions=[],
            confidence_score=0.0,
        )

    by_id = {r["id"]: r for r in retrieved}
    context_lines = [
        f"id={r['id']} | date={r['metadata']['date']} | {r['document']} | "
        f"amount={r['metadata']['amount']} | "
        f"is_anomaly={r['metadata'].get('is_anomaly', False)}"
        for r in retrieved
    ]

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a personal finance assistant. Answer the user's "
                        "question about their transactions using ONLY the "
                        "transactions listed below - do not assume or invent any "
                        "transaction that isn't listed. Each transaction has an "
                        "is_anomaly flag already computed by a separate pattern-"
                        "detection step (True means it was unusually large for its "
                        "category) - treat is_anomaly=True as a strong signal when "
                        "asked about suspicious or unusual activity. If the listed "
                        "transactions don't fully answer the question, say so and "
                        "lower your confidence score accordingly.\n\nTransactions:\n"
                        + "\n".join(context_lines)
                    ),
                },
                {"role": "user", "content": question},
            ],
            tools=[ANSWER_TOOL],
            tool_choice={"type": "function", "function": {"name": "answer_question"}},
        )

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("model did not return a tool call")

        raw = _RawAnswer.model_validate(json.loads(tool_calls[0].function.arguments))
    except (OpenAIError, RuntimeError, json.JSONDecodeError, ValidationError) as e:
        # Covers an unreachable/erroring OpenRouter API, a missing tool
        # call, and a response that fails validation. Retrieval already
        # succeeded at this point, but without a trustworthy generation step
        # we'd rather hand back an honest "couldn't answer" than crash the
        # whole interactive session over one bad question.
        logger.error("Failed to answer question %r: %s", question, e)
        return QueryAnswer(
            question=question,
            answer="Sorry, I couldn't generate an answer right now - please try again in a moment.",
            used_transactions=[],
            confidence_score=0.0,
        )

    # Defensive resolution: only accept ids that were actually part of the
    # retrieved set. Tool calling constrains the *shape* of the response but
    # not its *content* - the model could still cite an id it made up, so we
    # never trust it to describe a transaction; we look up every claimed
    # source in what we actually retrieved and build the record ourselves.
    used_transactions = [
        UsedTransaction(
            id=r["id"],
            date=r["metadata"]["date"],
            description=r["document"].split(" | category:")[0],
            amount=r["metadata"]["amount"],
            category=r["metadata"]["category"],
        )
        for tid in raw.used_transaction_ids
        if (r := by_id.get(tid)) is not None
    ]

    return QueryAnswer(
        question=question,
        answer=_correct_dollar_amounts(raw.answer, used_transactions),
        used_transactions=used_transactions,
        confidence_score=raw.confidence_score,
    )


def print_answer(result: QueryAnswer) -> None:
    print(f"\nQ: {result.question}")
    print(f"A: {result.answer}")
    print(f"Confidence: {result.confidence_score:.2f}")
    if result.used_transactions:
        print("Sources:")
        for t in result.used_transactions:
            print(f"  - {t.date}  {t.description}  ${t.amount:.2f}  [{t.category}]")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how much did I spend on food?"
    print_answer(ask_question(question))
