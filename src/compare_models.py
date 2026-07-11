import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

PROMPT = "Explain what a neural network is in 3 sentences for a first-year CS student"

MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
]


def main():
    for model in MODELS:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
        )
        elapsed = time.perf_counter() - start

        reply = response.choices[0].message.content

        print(f"Model: {model}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Response: {reply}")
        print("-" * 60)


if __name__ == "__main__":
    main()
