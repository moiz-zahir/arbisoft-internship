import os

from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env into the process environment (e.g. OPENROUTER_API_KEY).
load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "anthropic/claude-haiku-4.5"


def main():
    # This list is the model's entire "memory" of the conversation.
    # The OpenAI-compatible chat API is stateless - the server does not
    # remember anything between requests. Every call only knows about the
    # messages we include in this list, so to make the model "remember"
    # earlier turns, we append every user message and every assistant
    # reply here and send the whole list back on each new request.
    history = []

    print("Chat with Claude 3.5 Haiku via OpenRouter. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            # Resend the full history (not just the latest message) because
            # the API has no session/memory of its own - each request is
            # independent. Without resending prior turns, the model would
            # have no idea what was said earlier in the conversation.
            messages=history,
        )

        reply = response.choices[0].message.content
        print(f"Assistant: {reply}")

        # Store the assistant's reply too, so it becomes part of the context
        # for the next turn (letting the model reference its own past answers).
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
