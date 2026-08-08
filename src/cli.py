import sys

from src.agents import model_router
from src.agents.categorizer import categorize_batch
from src.agents.pattern_detector import build_report, print_report
from src.agents.query_agent import ask_question, print_answer
from src.agents.summarizer import generate_summary, print_summary
from src.tools.csv_ingestion import load_transactions_csv
from src.tools.transaction_store import clear_db, get_all, mark_anomalies, store_batch
from src.tools.validation_guards import run_guards


def choose_model_backend() -> bool:
    """
    Asks the user whether to categorize with a local (Ollama) or cloud
    (OpenRouter) model. Returns True only if the user asked for local AND
    Ollama is actually reachable right now - a preference alone isn't
    enough, since Ollama might not be running.
    """
    choice = input(
        "Use local model (Ollama) or cloud model (OpenRouter) for categorization? (local/cloud): "
    ).strip().lower()
    if choice != "local":
        return False
    if model_router.get_available_backend() != "local":
        print("Ollama not available, falling back to OpenRouter")
        return False
    return True


def run_pipeline(csv_path: str, use_local: bool) -> None:
    print(f"Loading transactions from {csv_path}...")
    batch = load_transactions_csv(csv_path)
    print(f"Loaded {batch.total_count} transactions ({batch.date_range})")

    print(f"Categorizing transactions ({'local' if use_local else 'cloud'} model)...")
    categorize_batch(batch, use_local=use_local)

    warnings = run_guards(batch)
    if warnings:
        print("\nValidation warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nAll validation checks passed.")

    print("Clearing existing vector store...")
    clear_db()

    print("Storing transactions in ChromaDB...")
    store_batch(batch)

    print("Analyzing spending patterns...\n")
    report = build_report(get_all())
    mark_anomalies([t.id for t in report.large_transactions])
    print_report(report)


def load_csv_with_retry(csv_path: str, use_local: bool) -> None:
    """Runs the pipeline against csv_path, reprompting for a new path if it can't be loaded."""
    while True:
        try:
            run_pipeline(csv_path, use_local)
            return
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            csv_path = input("Enter a valid CSV path (or 'exit' to quit): ").strip()
            if csv_path.lower() in ("exit", "quit"):
                print("Goodbye!")
                sys.exit(0)


def ask_a_question() -> None:
    question = input("Ask a question: ").strip()
    if not question:
        # Empty input - just return to the menu instead of treating it as a question.
        print("No question entered.")
        return
    print_answer(ask_question(question))


def get_spending_summary() -> None:
    period = input("What time period? (e.g. 'last week', 'this month', 'July') ").strip()
    if not period:
        print("No time period entered.")
        return
    try:
        print_summary(generate_summary(period))
    except ValueError as e:
        print(f"\nError: {e}")


def menu_loop() -> None:
    while True:
        print("\nOptions: (1) Ask a question  (2) Get spending summary  (3) Exit")
        try:
            choice = input("> ").strip()
        except EOFError:
            break

        if choice == "1":
            ask_a_question()
        elif choice == "2":
            get_spending_summary()
        elif choice in ("3", "exit", "quit"):
            print("Goodbye!")
            break
        else:
            print("Please choose 1, 2, or 3.")


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_transactions.csv"
    try:
        use_local = choose_model_backend()
        load_csv_with_retry(csv_path, use_local)
        menu_loop()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
