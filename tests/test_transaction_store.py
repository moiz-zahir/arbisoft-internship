from src.models.transaction import TransactionType
from src.tools.transaction_store import clear_db, get_all, get_by_category, query_similar, store_batch


def test_store_batch_saves_to_chromadb(chroma_test_dir, sample_batch):
    store_batch(sample_batch)

    records = get_all()
    assert len(records) == len(sample_batch.transactions)


def test_query_returns_relevant_results(chroma_test_dir, sample_batch):
    store_batch(sample_batch)

    results = query_similar("coffee and food purchases", n_results=3)

    assert len(results) > 0
    categories = {r["metadata"]["category"] for r in results}
    assert TransactionType.FOOD.value in categories


def test_get_by_category_returns_correct_transactions(chroma_test_dir, sample_batch):
    store_batch(sample_batch)

    food_records = get_by_category(TransactionType.FOOD)

    assert len(food_records) == 2
    assert all(r["metadata"]["category"] == TransactionType.FOOD.value for r in food_records)


def test_clear_db_empties_collection(chroma_test_dir, sample_batch):
    store_batch(sample_batch)
    assert len(get_all()) > 0

    clear_db()

    assert get_all() == []
