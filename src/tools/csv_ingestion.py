import csv
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from src.models.transaction import Transaction, TransactionBatch

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "description", "amount", "type"}


def load_transactions_csv(path: str | Path) -> TransactionBatch:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}. Check the path and try again."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        transactions: list[Transaction] = []
        skipped_count = 0
        for line_num, row in enumerate(reader, start=2):
            try:
                transactions.append(
                    Transaction(
                        date=row["date"].strip(),
                        description=row["description"].strip(),
                        amount=float(row["amount"]),
                        type=row["type"].strip().lower(),
                    )
                )
            except (ValidationError, ValueError, KeyError, AttributeError) as e:
                skipped_count += 1
                logger.warning("Skipping malformed row %d: %s (%s)", line_num, row, e)

    if skipped_count:
        logger.warning(
            "Skipped %d malformed row(s) out of %d total rows in %s",
            skipped_count, skipped_count + len(transactions), path,
        )

    total_debits = sum(t.amount for t in transactions if t.type == "debit")
    total_credits = sum(t.amount for t in transactions if t.type == "credit")
    dates = sorted(t.date for t in transactions)
    date_range = f"{dates[0]} to {dates[-1]}" if dates else ""

    return TransactionBatch(
        transactions=transactions,
        total_count=len(transactions),
        total_debits=total_debits,
        total_credits=total_credits,
        date_range=date_range,
    )


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_transactions.csv"
    batch = load_transactions_csv(csv_path)

    print(f"Loaded {batch.total_count} transactions")
    print(f"Date range: {batch.date_range}")
    print(f"Total debits: {batch.total_debits:.2f}")
    print(f"Total credits: {batch.total_credits:.2f}")
    print()
    for t in batch.transactions:
        print(t.model_dump())
