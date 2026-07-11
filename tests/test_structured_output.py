"""Tests for src/structured_output.py.

These cover two things: that the Pydantic schema actually rejects the
malformed shapes it's supposed to reject (tests 1-4), and that the
end-to-end script produces a real, valid output.json when run against
the live model (test 5). Test 5 makes a real OpenRouter API call, so
it's slower and depends on network/API availability - the other four
are pure, offline schema checks.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from structured_output import OUTPUT_PATH, ResearchSummary

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_SUMMARY = {
    "title": "Attention Mechanism in Transformers",
    "key_points": [
        "Attention lets a model weigh how much each token matters to another.",
        "Queries, keys, and values are learned linear projections of the input.",
        "Multi-head attention runs several attention operations in parallel.",
    ],
    "confidence_score": 0.9,
    "limitations": "Does not cover computational complexity in depth.",
}


def test_valid_research_summary_passes_validation():
    summary = ResearchSummary(**VALID_SUMMARY)
    assert summary.title == VALID_SUMMARY["title"]
    assert len(summary.key_points) == 3
    assert summary.confidence_score == pytest.approx(0.9)


def test_validation_fails_with_fewer_than_three_key_points():
    data = {**VALID_SUMMARY, "key_points": ["Only one point.", "Only two points."]}
    with pytest.raises(ValidationError):
        ResearchSummary(**data)


def test_validation_fails_when_confidence_score_above_one():
    data = {**VALID_SUMMARY, "confidence_score": 1.5}
    with pytest.raises(ValidationError):
        ResearchSummary(**data)


def test_validation_fails_when_confidence_score_below_zero():
    data = {**VALID_SUMMARY, "confidence_score": -0.1}
    with pytest.raises(ValidationError):
        ResearchSummary(**data)


def test_output_json_exists_and_is_valid_after_running_script():
    """Runs structured_output.py end-to-end (real API call) and checks
    that it wrote a valid output.json matching the ResearchSummary schema.
    """
    result = subprocess.run(
        [sys.executable, "src/structured_output.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stdout}\n{result.stderr}"

    output_path = PROJECT_ROOT / OUTPUT_PATH
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    # Confirms the file isn't just valid JSON, but valid *ResearchSummary* JSON.
    ResearchSummary.model_validate(data)
