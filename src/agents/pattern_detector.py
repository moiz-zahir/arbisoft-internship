import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.tools.transaction_store import get_all, mark_anomalies

load_dotenv()

MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANOMALY_MULTIPLIER = 2.0


class CategoryTotal(BaseModel):
    category: str
    total: float
    count: int


class LargeTransaction(BaseModel):
    id: str
    date: str
    description: str
    amount: float
    category: str
    category_average: float
    ratio: float


class RecurringTransaction(BaseModel):
    description: str
    count: int
    total_amount: float
    average_amount: float


class SpendingReport(BaseModel):
    top_categories: list[CategoryTotal]
    large_transactions: list[LargeTransaction]
    recurring_transactions: list[RecurringTransaction]
    total_spent: float
    total_received: float
    net: float
    summary: str


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set - check your .env file")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _description(record: dict) -> str:
    # The document text is "<description> | category: X | type: Y" - the
    # description is whatever precedes the first " | ".
    return record["document"].split(" | category:")[0]


def _generate_summary(
    client: OpenAI,
    top_categories: list[CategoryTotal],
    large_transactions: list[LargeTransaction],
    recurring_transactions: list[RecurringTransaction],
    total_spent: float,
    total_received: float,
) -> str:
    # The category totals, anomaly ratios, and recurring counts below are all
    # computed with plain arithmetic, not the LLM - money math has to be
    # exact, and an LLM asked to "analyze spending" from a raw transaction
    # list is prone to arithmetic slips. Claude Haiku's job here is narrower
    # and better suited to a language model: turn already-correct numbers
    # into a short, readable narrative.
    stats = {
        "top_categories": [c.model_dump() for c in top_categories],
        "large_transactions": [t.model_dump() for t in large_transactions],
        "recurring_transactions": [r.model_dump() for r in recurring_transactions],
        "total_spent": total_spent,
        "total_received": total_received,
    }
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a personal finance analyst. You are given precomputed "
                    "spending statistics as JSON. Write a concise 3-5 sentence "
                    "summary highlighting the most important patterns and any "
                    "notable anomalies. Only reference the numbers provided - do "
                    "not invent or recompute figures."
                ),
            },
            {"role": "user", "content": json.dumps(stats, indent=2)},
        ],
    )
    return response.choices[0].message.content.strip()


def build_report(records: list[dict]) -> SpendingReport:
    """
    Analyzes all stored transactions and returns a validated SpendingReport.

    Spending totals, anomaly detection, and recurring-charge grouping are all
    deterministic aggregations over the ChromaDB metadata - the only place
    the LLM is involved is generating the human-readable `summary`, which
    keeps the numeric findings trustworthy regardless of model behavior.
    """
    debit_records = [r for r in records if r["metadata"]["type"] == "debit"]
    credit_records = [r for r in records if r["metadata"]["type"] == "credit"]

    total_spent = sum(r["metadata"]["amount"] for r in debit_records)
    total_received = sum(r["metadata"]["amount"] for r in credit_records)

    category_amounts: dict[str, list[float]] = defaultdict(list)
    for r in debit_records:
        category_amounts[r["metadata"]["category"]].append(r["metadata"]["amount"])

    category_totals = [
        CategoryTotal(category=cat, total=sum(amts), count=len(amts))
        for cat, amts in category_amounts.items()
    ]
    category_totals.sort(key=lambda c: c.total, reverse=True)
    top_categories = category_totals[:3]

    category_avg = {cat: sum(amts) / len(amts) for cat, amts in category_amounts.items()}

    large_transactions = []
    for r in debit_records:
        cat = r["metadata"]["category"]
        avg = category_avg[cat]
        amount = r["metadata"]["amount"]
        if avg > 0 and amount > ANOMALY_MULTIPLIER * avg:
            large_transactions.append(
                LargeTransaction(
                    id=r["id"],
                    date=r["metadata"]["date"],
                    description=_description(r),
                    amount=amount,
                    category=cat,
                    category_average=avg,
                    ratio=amount / avg,
                )
            )

    desc_groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        desc_groups[_description(r)].append(r["metadata"]["amount"])
    recurring_transactions = [
        RecurringTransaction(
            description=desc,
            count=len(amts),
            total_amount=sum(amts),
            average_amount=sum(amts) / len(amts),
        )
        for desc, amts in desc_groups.items()
        if len(amts) > 1
    ]

    summary = _generate_summary(
        _get_client(), top_categories, large_transactions, recurring_transactions,
        total_spent, total_received,
    )

    return SpendingReport(
        top_categories=top_categories,
        large_transactions=large_transactions,
        recurring_transactions=recurring_transactions,
        total_spent=total_spent,
        total_received=total_received,
        net=total_received - total_spent,
        summary=summary,
    )


def format_report(report: SpendingReport) -> str:
    lines = ["=" * 60, "SPENDING PATTERN REPORT", "=" * 60]

    lines.append("\nTop Spending Categories:")
    for c in report.top_categories:
        lines.append(f"  {c.category:<15} ${c.total:>10.2f}  ({c.count} transactions)")

    lines.append("\nUnusually Large Transactions (>2x category average):")
    if report.large_transactions:
        for t in report.large_transactions:
            lines.append(
                f"  {t.date}  {t.description:<32}  ${t.amount:>8.2f}  "
                f"(category avg ${t.category_average:.2f}, {t.ratio:.1f}x)"
            )
    else:
        lines.append("  None detected")

    lines.append("\nRecurring Transactions:")
    if report.recurring_transactions:
        for r in report.recurring_transactions:
            lines.append(
                f"  {r.description:<32}  x{r.count}  "
                f"total ${r.total_amount:.2f}  avg ${r.average_amount:.2f}"
            )
    else:
        lines.append("  None detected")

    lines.append("\nTotals:")
    lines.append(f"  Total spent:    ${report.total_spent:.2f}")
    lines.append(f"  Total received: ${report.total_received:.2f}")
    lines.append(f"  Net:            ${report.net:+.2f}")

    lines.append("\nSummary:")
    lines.append(f"  {report.summary}")
    lines.append("=" * 60)

    return "\n".join(lines)


def print_report(report: SpendingReport) -> None:
    print(format_report(report))


if __name__ == "__main__":
    records = get_all()
    if not records:
        raise SystemExit("No transactions found in ChromaDB - run the ingestion/categorizer flow first.")

    report = build_report(records)
    mark_anomalies([t.id for t in report.large_transactions])
    print_report(report)
