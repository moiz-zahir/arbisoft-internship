import logging
import re

from src.models.transaction import TransactionBatch, TransactionType

logger = logging.getLogger(__name__)

DATE_FORMAT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LOW_CONFIDENCE_THRESHOLD = 0.7
MAX_OTHER_RATIO = 0.2


def validate_no_negative_amounts(batch: TransactionBatch) -> None:
    """Raises if any debit has a negative amount - a debit is always a positive charge against the account."""
    offenders = [t for t in batch.transactions if t.type == "debit" and t.amount < 0]
    if offenders:
        raise ValueError(
            f"{len(offenders)} debit transaction(s) have a negative amount: "
            f"{[t.description for t in offenders]}"
        )


def validate_date_format(batch: TransactionBatch) -> None:
    """Raises if any transaction's date isn't in YYYY-MM-DD format."""
    offenders = [t for t in batch.transactions if not DATE_FORMAT_RE.match(t.date)]
    if offenders:
        raise ValueError(
            f"{len(offenders)} transaction(s) have a malformed date (expected YYYY-MM-DD): "
            f"{[(t.description, t.date) for t in offenders]}"
        )


def validate_confidence_scores(batch: TransactionBatch) -> None:
    """Logs a warning for every transaction whose categorization confidence is below the review threshold."""
    for t in batch.transactions:
        if t.confidence_score is not None and t.confidence_score < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low confidence (%.2f) categorization for %r: %s",
                t.confidence_score, t.description, t.category,
            )


def validate_category_coverage(batch: TransactionBatch) -> None:
    """Raises if more than MAX_OTHER_RATIO of transactions fell back to OTHER - a sign categorization is failing broadly rather than on isolated rows."""
    if not batch.transactions:
        return
    other_count = sum(1 for t in batch.transactions if t.category == TransactionType.OTHER)
    ratio = other_count / len(batch.transactions)
    if ratio > MAX_OTHER_RATIO:
        raise ValueError(
            f"{other_count}/{len(batch.transactions)} transactions ({ratio:.0%}) categorized "
            f"as OTHER, exceeding the {MAX_OTHER_RATIO:.0%} threshold - categorization may be failing."
        )


def run_guards(batch: TransactionBatch) -> list[str]:
    """
    Runs every validation guard against a categorized batch and returns the
    warnings collected. A guard that raises is caught here and turned into
    a warning message rather than propagated, so a data-quality issue can be
    surfaced to the user without crashing the pipeline over it - the same
    "surface, don't crash" approach already used elsewhere in this project.
    """
    warnings: list[str] = []
    for guard in (
        validate_no_negative_amounts,
        validate_date_format,
        validate_confidence_scores,
        validate_category_coverage,
    ):
        try:
            guard(batch)
        except ValueError as e:
            warnings.append(str(e))
    return warnings
