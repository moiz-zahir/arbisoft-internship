import logging

import pytest

from src.models.transaction import Transaction, TransactionBatch, TransactionType
from src.tools.validation_guards import (
    validate_category_coverage,
    validate_confidence_scores,
    validate_date_format,
    validate_no_negative_amounts,
)


def _batch(transactions: list[Transaction]) -> TransactionBatch:
    debits = sum(t.amount for t in transactions if t.type == "debit")
    credits = sum(t.amount for t in transactions if t.type == "credit")
    return TransactionBatch(
        transactions=transactions,
        total_count=len(transactions),
        total_debits=debits,
        total_credits=credits,
        date_range="",
    )


def test_validate_no_negative_amounts_raises_on_negative_debit():
    batch = _batch([
        Transaction(date="2026-07-01", description="Negative debit", amount=-5.00, type="debit"),
    ])

    with pytest.raises(ValueError):
        validate_no_negative_amounts(batch)


def test_validate_date_format_raises_on_malformed_date():
    batch = _batch([
        Transaction(date="07/01/2026", description="Bad date format", amount=5.00, type="debit"),
    ])

    with pytest.raises(ValueError):
        validate_date_format(batch)


def test_validate_confidence_scores_logs_warning_for_low_confidence(caplog):
    batch = _batch([
        Transaction(
            date="2026-07-01", description="Ambiguous Charge", amount=10.00, type="debit",
            category=TransactionType.OTHER, confidence_score=0.4,
        ),
    ])

    with caplog.at_level(logging.WARNING):
        validate_confidence_scores(batch)

    assert any("Ambiguous Charge" in record.message for record in caplog.records)


def test_validate_category_coverage_raises_when_too_many_other():
    transactions = [
        Transaction(
            date="2026-07-01", description=f"Unknown {i}", amount=10.00, type="debit",
            category=TransactionType.OTHER,
        )
        for i in range(3)
    ] + [
        Transaction(
            date="2026-07-01", description="Known", amount=10.00, type="debit",
            category=TransactionType.FOOD,
        )
    ]
    batch = _batch(transactions)  # 3/4 = 75% OTHER, exceeds the 20% threshold

    with pytest.raises(ValueError):
        validate_category_coverage(batch)
