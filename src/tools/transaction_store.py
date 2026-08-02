import hashlib
from pathlib import Path

import chromadb

from src.models.transaction import Transaction, TransactionBatch, TransactionType

CHROMA_DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "transactions"


def _get_client() -> chromadb.ClientAPI:
    # Persisted to disk (rather than an in-memory client) so the embeddings
    # and metadata survive between runs instead of being rebuilt every time.
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def _get_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(COLLECTION_NAME)


def _transaction_id(t: Transaction) -> str:
    # Deterministic id (hash of the transaction's identifying fields) so
    # re-storing the same statement twice upserts instead of duplicating.
    key = f"{t.date}|{t.description}|{t.amount}|{t.type}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _transaction_document(t: Transaction) -> str:
    # This is the text that gets embedded. Including the category alongside
    # the raw description lets semantic queries like "subscriptions and
    # streaming" match transactions even when the merchant name alone
    # (e.g. "SPOTIFY PREMIUM") wouldn't obviously match "streaming".
    category = t.category.value if t.category else "UNCATEGORIZED"
    return f"{t.description} | category: {category} | type: {t.type}"


def _transaction_metadata(t: Transaction) -> dict:
    # Chroma metadata values must be str/int/float/bool, so enums and
    # possibly-missing fields are normalized here.
    return {
        "date": t.date,
        "amount": t.amount,
        "category": t.category.value if t.category else "UNCATEGORIZED",
        "type": t.type,
    }


def store_batch(batch: TransactionBatch) -> None:
    """Embeds and stores every transaction in the batch, keyed by content hash."""
    if not batch.transactions:
        return
    collection = _get_collection(_get_client())
    collection.upsert(
        ids=[_transaction_id(t) for t in batch.transactions],
        documents=[_transaction_document(t) for t in batch.transactions],
        metadatas=[_transaction_metadata(t) for t in batch.transactions],
    )


def query_similar(query: str, n_results: int = 5) -> list[dict]:
    """Returns the transactions whose embedded text is most similar to `query`."""
    collection = _get_collection(_get_client())
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"id": id_, "document": doc, "metadata": meta, "distance": distance}
        for id_, doc, meta, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def get_by_category(category: TransactionType | str) -> list[dict]:
    """Returns all stored transactions matching an exact category (metadata filter, not semantic)."""
    collection = _get_collection(_get_client())
    category_value = category.value if isinstance(category, TransactionType) else category
    results = collection.get(where={"category": category_value})
    return [
        {"document": doc, "metadata": meta}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


def get_all() -> list[dict]:
    """Returns every stored transaction as {id, document, metadata}."""
    collection = _get_collection(_get_client())
    results = collection.get()
    return [
        {"id": id_, "document": doc, "metadata": meta}
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    ]


def mark_anomalies(ids: list[str]) -> None:
    """Sets is_anomaly=True in metadata for the given transaction ids."""
    if not ids:
        return
    collection = _get_collection(_get_client())
    existing = collection.get(ids=ids)
    updated_metadatas = []
    for meta in existing["metadatas"]:
        meta = dict(meta)
        meta["is_anomaly"] = True
        updated_metadatas.append(meta)
    collection.update(ids=existing["ids"], metadatas=updated_metadatas)


def clear_db() -> None:
    """Deletes the collection entirely so the next store_batch starts fresh."""
    client = _get_client()
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
