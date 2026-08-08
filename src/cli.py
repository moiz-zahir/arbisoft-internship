import sys

from src.agents.categorizer import categorize_batch
from src.agents.pattern_detector import build_report, print_report
from src.agents.query_agent import ask_question, print_answer
from src.tools.csv_ingestion import load_transactions_csv
from src.tools.transaction_store import clear_db, get_all, mark_anomalies, store_batch


def run_pipeline(csv_path: str) -> None:
    print(f"Loading transactions from {csv_path}...")
    batch = load_transactions_csv(csv_path)
    print(f"Loaded {batch.total_count} transactions ({batch.date_range})")

    print("Categorizing transactions...")
    categorize_batch(batch)

    print("Clearing existing vector store...")
    clear_db()

    print("Storing transactions in ChromaDB...")
    store_batch(batch)

    print("Analyzing spending patterns...\n")
    report = build_report(get_all())
    mark_anomalies([t.id for t in report.large_transactions])
    print_report(report)


def load_csv_with_retry(csv_path: str) -> None:
    """Runs the pipeline against csv_path, reprompting for a new path if it can't be loaded."""
    while True:
        try:
            run_pipeline(csv_path)
            return
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            csv_path = input("Enter a valid CSV path (or 'exit' to quit): ").strip()
            if csv_path.lower() in ("exit", "quit"):
                print("Goodbye!")
                sys.exit(0)


def interactive_loop() -> None:
    print("\nAsk questions about your transactions (type 'exit' to quit).")
    print(
        "Examples: 'how much did I spend on food?', 'show me my biggest expenses', "
        "'any suspicious transactions?', 'how much did I earn this month?'\n"
    )
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            # Empty input - just prompt again instead of treating it as a question.
            continue
        if question.lower() in ("exit", "quit"):
            break
        print_answer(ask_question(question))
        print()


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_transactions.csv"
    try:
        load_csv_with_retry(csv_path)
        interactive_loop()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
