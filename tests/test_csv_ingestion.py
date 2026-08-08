import logging

import pytest

from src.models.transaction import TransactionBatch
from src.tools.csv_ingestion import load_transactions_csv


def test_load_sample_csv_returns_transaction_batch(sample_csv_path):
    batch = load_transactions_csv(sample_csv_path)
    assert isinstance(batch, TransactionBatch)


def test_total_count_is_20(sample_csv_path):
    batch = load_transactions_csv(sample_csv_path)
    assert batch.total_count == 20


def test_total_debits_and_credits_are_correct(sample_csv_path):
    batch = load_transactions_csv(sample_csv_path)
    assert batch.total_debits == pytest.approx(685.70)
    assert batch.total_credits == pytest.approx(3884.50)


def test_malformed_rows_are_skipped(tmp_path):
    csv_content = (
        "date,description,amount,type\n"
        "2026-01-01,Valid Coffee,5.00,debit\n"
        "2026-01-02,Bad Amount,not_a_number,debit\n"
        "2026-01-03,Bad Type,10.00,not_debit_or_credit\n"
        "2026-01-04,Valid Salary,100.00,credit\n"
    )
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text(csv_content)

    batch = load_transactions_csv(csv_path)

    assert batch.total_count == 2
    assert {t.description for t in batch.transactions} == {"Valid Coffee", "Valid Salary"}


def test_missing_required_columns_raises(tmp_path):
    csv_content = "date,description,type\n2026-01-01,No Amount Column,debit\n"
    csv_path = tmp_path / "missing_columns.csv"
    csv_path.write_text(csv_content)

    with pytest.raises(ValueError):
        load_transactions_csv(csv_path)


def test_missing_file_raises_file_not_found_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_transactions_csv(missing_path)


def test_empty_file_raises_value_error(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")

    with pytest.raises(ValueError):
        load_transactions_csv(empty_path)


def test_skipped_row_count_is_logged(tmp_path, caplog):
    csv_content = (
        "date,description,amount,type\n"
        "2026-01-01,Valid Coffee,5.00,debit\n"
        "2026-01-02,Bad Amount,not_a_number,debit\n"
    )
    csv_path = tmp_path / "one_bad_row.csv"
    csv_path.write_text(csv_content)

    with caplog.at_level(logging.WARNING):
        load_transactions_csv(csv_path)

    assert any("Skipped 1 malformed row" in record.message for record in caplog.records)
