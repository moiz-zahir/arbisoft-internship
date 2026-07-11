"""
Structured output from an LLM, validated with Pydantic.

LLMs generate text, not data. Even when you ask for JSON, the model can
return a field with the wrong type, omit a required field, invent extra
fields, return a list that's too short, or wrap the JSON in markdown
fences or extra prose. If that raw output is used directly - e.g.
`data["confidence_score"]` fed straight into a calculation, or
`data["key_points"]` iterated over assuming it has at least 3 items -
the program can crash, silently store garbage, or make decisions on a
number the model made up out of thin air with no guarantee it's even
in a sane range.

Validating with Pydantic turns "hope the model behaved" into "prove the
model behaved": we define the exact shape we require, and anything that
doesn't match is rejected before it touches the rest of the program. On
top of that, we don't just fail on bad output - we feed the validation
error back to the model and ask it to correct itself, which recovers
from most formatting mistakes without any human involvement.
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = "anthropic/claude-haiku-4.5"

MAX_RETRIES = 3
OUTPUT_PATH = "output.json"

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


class ResearchSummary(BaseModel):
    """The exact schema we require from the model. Pydantic enforces
    types and constraints at construction time, so a ResearchSummary
    instance existing at all is proof the data is well-formed - no
    separate "is this valid" checks needed anywhere else in the code.
    """

    title: str
    key_points: list[str] = Field(min_length=3)
    confidence_score: float = Field(ge=0.0, le=1.0)
    limitations: str


SYSTEM_PROMPT = (
    "You are a research assistant. Respond with ONLY a single JSON object "
    "and no other text, markdown fences, or commentary. The JSON object "
    "must have exactly these fields:\n"
    '- "title": string\n'
    '- "key_points": array of at least 3 strings\n'
    '- "confidence_score": number between 0 and 1\n'
    '- "limitations": string describing what the summary leaves out or is unsure about\n'
)

USER_PROMPT = (
    "Summarize the attention mechanism used in transformer neural networks, "
    "as a JSON object matching the schema described above."
)


def extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json ... ``` fences even when told
    not to. Strip that wrapping so json.loads doesn't choke on it.
    """
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    return match.group(1) if match else text.strip()


def request_research_summary() -> ResearchSummary:
    """Ask the model for a ResearchSummary and validate the response.

    If the response is malformed JSON or fails Pydantic validation, we
    don't just retry blindly - we tell the model exactly what it got
    wrong (the parser/validation error) and ask it to fix that specific
    problem, which is far more likely to succeed than a bare retry.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # 1 initial attempt + up to MAX_RETRIES retries
        print(f"Attempt {attempt}: requesting structured summary from {CHAT_MODEL}...")
        response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        raw_content = response.choices[0].message.content

        try:
            data = json.loads(extract_json(raw_content))
            summary = ResearchSummary.model_validate(data)
            print(f"Attempt {attempt}: validation succeeded.")
            return summary
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = error
            print(f"Attempt {attempt}: validation failed - {error}")
            if attempt <= MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response was invalid and failed schema validation with this "
                            f"error:\n{error}\n\n"
                            "Respond again with ONLY a corrected JSON object matching the schema "
                            "exactly - no markdown fences, no extra text."
                        ),
                    }
                )

    raise RuntimeError(f"Failed to get a valid ResearchSummary after {MAX_RETRIES + 1} attempts: {last_error}")


def print_summary(summary: ResearchSummary) -> None:
    print("\n" + "=" * 60)
    print("VALIDATED RESEARCH SUMMARY")
    print("=" * 60)
    print(f"Title: {summary.title}")
    print(f"Confidence score: {summary.confidence_score:.2f}")
    print("Key points:")
    for i, point in enumerate(summary.key_points, start=1):
        print(f"  {i}. {point}")
    print(f"Limitations: {summary.limitations}")


if __name__ == "__main__":
    summary = request_research_summary()
    print_summary(summary)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))
    print(f"\nSaved validated output to {OUTPUT_PATH}")
