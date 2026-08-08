import pytest
from pydantic import ValidationError

from src.models.transaction import Transaction, TransactionBatch


def test_valid_transaction_passes_validation():
    t = Transaction(date="2026-07-01", description="Coffee", amount=5.75, type="debit")

    assert t.date == "2026-07-01"
    assert t.amount == 5.75
    assert t.type == "debit"
    assert t.category is None
    assert t.confidence_score is None
    assert t.is_anomaly is False
    assert t.notes is None


def test_invalid_type_fails_validation():
    with pytest.raises(ValidationError):
        Transaction(date="2026-07-01", description="Coffee", amount=5.75, type="withdrawal")


def test_confidence_score_above_one_fails_validation():
    with pytest.raises(ValidationError):
        Transaction(
            date="2026-07-01", description="Coffee", amount=5.75, type="debit",
            confidence_score=1.5,
        )


def test_confidence_score_below_zero_fails_validation():
    with pytest.raises(ValidationError):
        Transaction(
            date="2026-07-01", description="Coffee", amount=5.75, type="debit",
            confidence_score=-0.1,
        )


def test_missing_required_fields_fail_validation():
    with pytest.raises(ValidationError):
        Transaction(description="Coffee", amount=5.75, type="debit")  # missing date

    with pytest.raises(ValidationError):
        Transaction(date="2026-07-01", amount=5.75, type="debit")  # missing description

    with pytest.raises(ValidationError):
        Transaction(date="2026-07-01", description="Coffee", type="debit")  # missing amount


def test_transaction_batch_totals_are_correct(sample_transactions):
    debits = sum(t.amount for t in sample_transactions if t.type == "debit")
    credits = sum(t.amount for t in sample_transactions if t.type == "credit")

    batch = TransactionBatch(
        transactions=sample_transactions,
        total_count=len(sample_transactions),
        total_debits=debits,
        total_credits=credits,
        date_range="2026-07-01 to 2026-07-05",
    )

    assert batch.total_count == len(sample_transactions)
    assert batch.total_debits == pytest.approx(debits)
    assert batch.total_credits == pytest.approx(credits)
