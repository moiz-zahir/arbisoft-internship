import calendar
import json
import logging
import os
import re
import sys
from datetime import date, timedelta

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from src.tools.transaction_store import get_all

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTH_ABBRS = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}


class SummaryReport(BaseModel):
    period: str
    total_spent: float
    total_earned: float
    savings_rate: float
    top_insight: str
    recommendations: list[str]
    full_summary: str


# What the model's tool call actually returns - validated before being copied
# into the SummaryReport that also carries the Python-computed totals above.
class _RawSummary(BaseModel):
    top_insight: str
    recommendations: list[str] = Field(min_length=3, max_length=3)
    full_summary: str


SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": (
            "Write a friendly, financial-advisor-style summary of a user's "
            "spending for a period, using only the totals and transactions provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "top_insight": {
                    "type": "string",
                    "description": "The single most important, specific insight about this period's spending.",
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Exactly 3 concrete, actionable recommendations based on the data provided.",
                },
                "full_summary": {
                    "type": "string",
                    "description": "A friendly 3-5 sentence paragraph summarizing spending for the period.",
                },
            },
            "required": ["top_insight", "recommendations", "full_summary"],
        },
    },
}


def _month_range(year: int, month: int) -> tuple[date, date]:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def resolve_period(period: str, today: date | None = None) -> tuple[date, date]:
    """
    Resolves a natural-language period ("last week", "this month", "July",
    "2026-07") into a concrete (start, end) date range.

    This is plain date arithmetic, not an LLM call, for the same reason
    totals are computed in Python below: a wrong date boundary is a silent
    failure - a transaction one day outside the range just quietly vanishes
    from the summary instead of raising an error. Deterministic parsing
    means the same period string always resolves to the same range, and any
    period we can't confidently parse raises immediately instead of
    guessing.
    """
    today = today or date.today()
    normalized = period.strip().lower()

    if normalized == "today":
        return today, today
    if normalized == "yesterday":
        y = today - timedelta(days=1)
        return y, y
    if normalized == "this week":
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=6)
    if normalized == "last week":
        this_week_start = today - timedelta(days=today.weekday())
        start = this_week_start - timedelta(days=7)
        return start, start + timedelta(days=6)
    if normalized == "this month":
        return _month_range(today.year, today.month)
    if normalized == "last month":
        year, month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        return _month_range(year, month)

    match = re.fullmatch(r"(\d{4})-(\d{1,2})", normalized)
    if match:
        return _month_range(int(match.group(1)), int(match.group(2)))

    match = re.fullmatch(r"([a-z]+)\s*(\d{4})?", normalized)
    if match:
        month_word, year_word = match.group(1), match.group(2)
        month = _MONTH_NAMES.get(month_word) or _MONTH_ABBRS.get(month_word)
        if month:
            year = int(year_word) if year_word else today.year
            return _month_range(year, month)

    raise ValueError(
        f"Could not understand time period {period!r}. Try 'last week', "
        f"'this month', 'last month', a month name like 'July', or 'YYYY-MM'."
    )


def _transactions_in_range(start: date, end: date) -> list[dict]:
    return [
        r for r in get_all()
        if start <= date.fromisoformat(r["metadata"]["date"]) <= end
    ]


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set - check your .env file")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def generate_summary(period: str) -> SummaryReport:
    """
    Builds a natural-language spending summary for a time period.

    total_spent, total_earned, and savings_rate are computed here in plain
    Python, never by the LLM. This project has already hit a real case
    (query_agent.py) where letting a model sum retrieved amounts itself
    produced a $1 arithmetic error - money math has to be exact every
    single time, and an LLM asked to total a list of transactions is prone
    to exactly this kind of slip. The LLM's job here is narrower and better
    suited to a language model: turn already-correct numbers into an
    insight, recommendations, and a friendly narrative a financial advisor
    might write, without ever being trusted to compute the numbers themselves.
    """
    start, end = resolve_period(period)
    records = _transactions_in_range(start, end)

    if not records:
        return SummaryReport(
            period=period,
            total_spent=0.0,
            total_earned=0.0,
            savings_rate=0.0,
            top_insight="No transactions found for this period.",
            recommendations=[],
            full_summary=f"No transactions were found for {period!r} ({start} to {end}).",
        )

    total_spent = sum(r["metadata"]["amount"] for r in records if r["metadata"]["type"] == "debit")
    total_earned = sum(r["metadata"]["amount"] for r in records if r["metadata"]["type"] == "credit")
    savings_rate = ((total_earned - total_spent) / total_earned * 100) if total_earned > 0 else 0.0

    context_lines = [
        f"{r['metadata']['date']} | {r['document']} | amount={r['metadata']['amount']}"
        for r in records
    ]

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly personal financial advisor. You are given "
                        "a user's transactions for a period and totals already computed "
                        "for you. Call generate_summary with the single most important "
                        "insight, exactly 3 concrete and actionable recommendations, and "
                        "a warm 3-5 sentence summary paragraph. Only reference the "
                        "numbers given below - do not invent or recompute figures.\n\n"
                        f"Period: {period} ({start} to {end})\n"
                        f"Total spent: {total_spent:.2f}\n"
                        f"Total earned: {total_earned:.2f}\n"
                        f"Savings rate: {savings_rate:.1f}%\n\n"
                        "Transactions:\n" + "\n".join(context_lines)
                    ),
                },
                {"role": "user", "content": f"Summarize my spending for {period}."},
            ],
            tools=[SUMMARY_TOOL],
            tool_choice={"type": "function", "function": {"name": "generate_summary"}},
        )

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("model did not return a tool call")

        raw = _RawSummary.model_validate(json.loads(tool_calls[0].function.arguments))
    except (OpenAIError, RuntimeError, json.JSONDecodeError, ValidationError) as e:
        # Retrieval and the Python-computed totals above already succeeded -
        # only the narrative generation failed, so we still return a fully
        # valid, useful SummaryReport instead of crashing the CLI over one
        # bad summary request.
        logger.error("Failed to generate summary for %r: %s", period, e)
        return SummaryReport(
            period=period,
            total_spent=total_spent,
            total_earned=total_earned,
            savings_rate=savings_rate,
            top_insight="Summary generation failed - showing computed totals only.",
            recommendations=[],
            full_summary=(
                f"You spent ${total_spent:.2f} and earned ${total_earned:.2f} during "
                f"{period}, a savings rate of {savings_rate:.1f}%. A friendly narrative "
                "summary couldn't be generated right now - please try again."
            ),
        )

    return SummaryReport(
        period=period,
        total_spent=total_spent,
        total_earned=total_earned,
        savings_rate=savings_rate,
        top_insight=raw.top_insight,
        recommendations=raw.recommendations,
        full_summary=raw.full_summary,
    )


def print_summary(report: SummaryReport) -> None:
    print(f"\n{'=' * 60}")
    print(f"SPENDING SUMMARY - {report.period}")
    print("=" * 60)
    print(f"Total spent:   ${report.total_spent:.2f}")
    print(f"Total earned:  ${report.total_earned:.2f}")
    print(f"Savings rate:  {report.savings_rate:.1f}%")
    print(f"\nTop insight: {report.top_insight}")
    if report.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(report.recommendations, start=1):
            print(f"  {i}. {rec}")
    print(f"\n{report.full_summary}")
    print("=" * 60)


if __name__ == "__main__":
    period_arg = " ".join(sys.argv[1:]) or "this month"
    print_summary(generate_summary(period_arg))
