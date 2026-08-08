from pathlib import Path

import pytest

from src.models.transaction import Transaction, TransactionBatch, TransactionType
from src.tools import transaction_store as store_module

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "sample_transactions.csv"


@pytest.fixture
def sample_csv_path() -> Path:
    return SAMPLE_CSV_PATH


@pytest.fixture
def sample_transactions() -> list[Transaction]:
    return [
        Transaction(
            date="2026-07-01", description="STARBUCKS COFFEE", amount=5.75,
            type="debit", category=TransactionType.FOOD, confidence_score=0.95,
        ),
        Transaction(
            date="2026-07-02", description="WHOLE FOODS MARKET", amount=87.32,
            type="debit", category=TransactionType.FOOD, confidence_score=0.9,
        ),
        Transaction(
            date="2026-07-03", description="UBER TRIP", amount=18.40,
            type="debit", category=TransactionType.TRANSPORT, confidence_score=0.98,
        ),
        Transaction(
            date="2026-07-04", description="NETFLIX.COM", amount=15.99,
            type="debit", category=TransactionType.SUBSCRIPTION, confidence_score=0.99,
        ),
        Transaction(
            date="2026-07-05", description="SALARY DEPOSIT", amount=3200.00,
            type="credit", category=TransactionType.INCOME, confidence_score=0.99,
        ),
    ]


@pytest.fixture
def sample_batch(sample_transactions: list[Transaction]) -> TransactionBatch:
    debits = sum(t.amount for t in sample_transactions if t.type == "debit")
    credits = sum(t.amount for t in sample_transactions if t.type == "credit")
    dates = sorted(t.date for t in sample_transactions)
    return TransactionBatch(
        transactions=sample_transactions,
        total_count=len(sample_transactions),
        total_debits=debits,
        total_credits=credits,
        date_range=f"{dates[0]} to {dates[-1]}",
    )


@pytest.fixture
def chroma_test_dir(tmp_path, monkeypatch):
    # Points the store at a throwaway directory per test so tests never
    # read/write the real data/chroma_db used by the CLI.
    test_path = tmp_path / "chroma_db"
    monkeypatch.setattr(store_module, "CHROMA_DB_PATH", test_path)
    yield test_path
