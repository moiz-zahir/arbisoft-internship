import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from src.models.transaction import Transaction, TransactionBatch, TransactionType
from src.tools.csv_ingestion import load_transactions_csv

load_dotenv()

MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOW_CONFIDENCE_THRESHOLD = 0.7
MAX_RETRIES = 3

# Tool schema the model must fill in. Forcing a function call (rather than
# asking for free-text JSON) is itself a first line of defense: the API
# rejects malformed arguments before they even reach us. Pydantic validation
# below is the second, stricter line of defense.
CATEGORIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "categorize_transaction",
        "description": "Classify a bank transaction into a category and report confidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [t.value for t in TransactionType],
                    "description": "The category that best fits the transaction.",
                },
                "confidence_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "How confident you are in this categorization, from 0 (guessing) to 1 (certain).",
                },
            },
            "required": ["category", "confidence_score"],
        },
    },
}


# The model's tool-call arguments are just untrusted JSON text until proven
# otherwise. We parse them into this Pydantic model before a Transaction is
# ever touched, so a hallucinated category string or an out-of-range
# confidence value fails validation instead of silently corrupting stored data.
class CategorizationResult(BaseModel):
    category: TransactionType
    confidence_score: float = Field(ge=0.0, le=1.0)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set - check your .env file")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def categorize_transaction(client: OpenAI, transaction: Transaction) -> CategorizationResult:
    """
    Calls Claude Haiku 4.5 (via OpenRouter) with a forced tool call to
    categorize a single transaction.

    Confidence scoring matters because a categorizer that's just "right or
    wrong" gives no way to prioritize human review. The confidence_score lets
    us flag borderline calls (ambiguous merchant names, unusual amounts)
    instead of silently trusting every guess - transactions below
    LOW_CONFIDENCE_THRESHOLD get flagged in `notes` for a human to check.

    Retries up to MAX_RETRIES times if the model's response fails Pydantic
    validation, since an LLM can still return an out-of-range confidence or
    malformed arguments despite the constrained tool schema.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a bank transaction categorizer. Given a transaction's "
                "date, description, amount, and type (debit/credit), call "
                "categorize_transaction with the best-fit category and your "
                "confidence in that choice."
            ),
        },
        {
            "role": "user",
            "content": (
                f"date: {transaction.date}\n"
                f"description: {transaction.description}\n"
                f"amount: {transaction.amount}\n"
                f"type: {transaction.type}"
            ),
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[CATEGORIZE_TOOL],
            tool_choice={"type": "function", "function": {"name": "categorize_transaction"}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            last_error = RuntimeError("model did not return a tool call")
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: no tool call returned", file=sys.stderr)
            continue
        try:
            args = json.loads(tool_calls[0].function.arguments)
            return CategorizationResult.model_validate(args)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            print(f"  attempt {attempt}/{MAX_RETRIES} failed validation: {e}", file=sys.stderr)

    raise RuntimeError(
        f"Failed to get a valid categorization for {transaction.description!r} "
        f"after {MAX_RETRIES} attempts: {last_error}"
    )


def categorize_batch(batch: TransactionBatch) -> TransactionBatch:
    """Categorizes every transaction in a batch in place and returns it."""
    client = _get_client()
    for transaction in batch.transactions:
        result = categorize_transaction(client, transaction)
        transaction.category = result.category
        transaction.confidence_score = result.confidence_score
        if result.confidence_score < LOW_CONFIDENCE_THRESHOLD:
            transaction.notes = "low confidence - needs review"
    return batch


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_transactions.csv"
    batch = load_transactions_csv(csv_path)
    categorize_batch(batch)

    for t in batch.transactions:
        flag = f"  [{t.notes}]" if t.notes else ""
        print(
            f"{t.date}  {t.description:<32}  {t.amount:>8.2f}  {t.type:<6}  "
            f"{t.category.value:<13}  conf={t.confidence_score:.2f}{flag}"
        )
