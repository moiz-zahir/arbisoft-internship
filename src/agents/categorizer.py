import json
import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from src.agents.model_router import log_model_used
from src.models.transaction import Transaction, TransactionBatch, TransactionType
from src.tools.csv_ingestion import load_transactions_csv

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3"

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


def _get_local_client() -> OpenAI:
    # Ollama's OpenAI-compatible endpoint doesn't check the api_key value at
    # all - it just has to be present for the client library to accept it.
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def _build_messages(transaction: Transaction) -> list[dict]:
    return [
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


def categorize_with_local_model(transaction: Transaction) -> CategorizationResult:
    """
    Categorizes a transaction with a local Ollama model (llama3) instead of
    OpenRouter.

    Routing to a local model trades a bit of accuracy for three things a
    cloud API can't offer: speed (no network round trip per transaction),
    cost (a whole statement can be categorized for free instead of billed
    per token), and privacy (raw transaction descriptions - merchant names,
    amounts - never leave the machine, which matters for financial data).
    If Ollama isn't running or the local model errors, we fall straight
    through to categorize_transaction (the cloud path) rather than failing
    the transaction outright - the whole point of routing is that a
    categorization always gets produced somehow.
    """
    try:
        response = _get_local_client().chat.completions.create(
            model=OLLAMA_MODEL,
            messages=_build_messages(transaction),
            tools=[CATEGORIZE_TOOL],
            tool_choice={"type": "function", "function": {"name": "categorize_transaction"}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("local model did not return a tool call")
        result = CategorizationResult.model_validate(json.loads(tool_calls[0].function.arguments))
    except (OpenAIError, RuntimeError, json.JSONDecodeError, ValidationError) as e:
        logger.warning(
            "Local model (Ollama) unavailable or failed for %r: %s - falling back to OpenRouter",
            transaction.description, e,
        )
        return categorize_transaction(_get_client(), transaction)

    log_model_used(transaction.description, "local")
    return result


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
    malformed arguments despite the constrained tool schema. If every retry
    fails - including because OpenRouter is unreachable - falls back to
    OTHER/0.0 rather than raising, so one bad transaction can't crash the
    whole batch. The 0.0 confidence makes the fallback obvious downstream:
    it's always below LOW_CONFIDENCE_THRESHOLD, so it gets flagged for
    review just like a genuinely low-confidence model answer would.
    """
    messages = _build_messages(transaction)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[CATEGORIZE_TOOL],
                tool_choice={"type": "function", "function": {"name": "categorize_transaction"}},
            )
        except OpenAIError as e:
            last_error = e
            logger.warning(
                "Attempt %d/%d: OpenRouter request failed for %r: %s",
                attempt, MAX_RETRIES, transaction.description, e,
            )
            continue

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            last_error = RuntimeError("model did not return a tool call")
            logger.warning(
                "Attempt %d/%d: no tool call returned for %r",
                attempt, MAX_RETRIES, transaction.description,
            )
            continue
        try:
            args = json.loads(tool_calls[0].function.arguments)
            result = CategorizationResult.model_validate(args)
            log_model_used(transaction.description, "cloud")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning(
                "Attempt %d/%d: validation failed for %r: %s",
                attempt, MAX_RETRIES, transaction.description, e,
            )

    logger.error(
        "Giving up on %r after %d attempts (%s) - falling back to OTHER/0.0",
        transaction.description, MAX_RETRIES, last_error,
    )
    return CategorizationResult(category=TransactionType.OTHER, confidence_score=0.0)


def categorize_batch(batch: TransactionBatch, use_local: bool = False) -> TransactionBatch:
    """Categorizes every transaction in a batch in place and returns it."""
    client = None if use_local else _get_client()
    total = len(batch.transactions)
    for i, transaction in enumerate(batch.transactions, start=1):
        print(f"Categorizing transaction {i} of {total}...", file=sys.stderr)
        try:
            if use_local:
                result = categorize_with_local_model(transaction)
            else:
                result = categorize_transaction(client, transaction)
        except Exception as e:
            # categorize_transaction already handles the failure modes we
            # anticipate (network errors, bad model output) internally. This
            # is a last-resort net for anything unforeseen, so a single
            # transaction can never take down the rest of the batch.
            logger.error(
                "Unexpected error categorizing %r: %s - falling back to OTHER/0.0",
                transaction.description, e,
            )
            result = CategorizationResult(category=TransactionType.OTHER, confidence_score=0.0)

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
