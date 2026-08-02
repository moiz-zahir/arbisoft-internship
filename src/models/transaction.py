from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    FOOD = "FOOD"
    TRANSPORT = "TRANSPORT"
    SHOPPING = "SHOPPING"
    SUBSCRIPTION = "SUBSCRIPTION"
    ENTERTAINMENT = "ENTERTAINMENT"
    INCOME = "INCOME"
    OTHER = "OTHER"


# We validate every transaction with Pydantic because the data enters the
# pipeline from two untrusted sources: raw bank CSVs (inconsistent formats,
# missing columns, stray whitespace) and LLM outputs (categorizer/pattern
# agents that can hallucinate a category or return an out-of-range score).
# Pydantic catches both failure modes at the boundary, before bad data can
# reach the vector store or downstream calculations like spending totals.
class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    # Constrained to the two values a bank statement actually uses, so a typo
    # or unexpected value from the CSV fails fast instead of silently
    # corrupting debit/credit totals later.
    type: Literal["debit", "credit"]

    # Populated by the categorizer agent after ingestion, not at parse time.
    category: Optional[TransactionType] = None
    # An LLM-reported confidence for its own categorization; bounded to
    # [0, 1] so a malformed model response can't skew downstream trust
    # scoring or filtering.
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_anomaly: bool = False
    notes: Optional[str] = None


class TransactionBatch(BaseModel):
    transactions: list[Transaction]
    total_count: int
    total_debits: float
    total_credits: float
    date_range: str
