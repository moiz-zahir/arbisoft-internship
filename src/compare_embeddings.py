"""
Compare two embedding models for RAG retrieval quality.

RAG's retrieval step is only as good as its embedding model: if the
model doesn't place semantically similar text near each other in
vector space, ChromaDB will hand the LLM irrelevant chunks and the
final answer will suffer no matter how good the chat model is. This
script embeds the same document chunks with two different models,
runs the same test questions against each, and compares what comes
back so we can see which embedding model retrieves better context for
this document set.

Model A: openai/text-embedding-3-small - the model used in rag_demo.py.
Model B: baai/bge-large-en-v1.5 - an open-source embedding model that
is tuned specifically for retrieval and ranks well on retrieval
benchmarks (MTEB), making it a meaningful point of comparison.

Because the two models produce vectors of different sizes and scales,
their raw ChromaDB distances aren't comparable to each other. To judge
relevance on a common scale, we also score each retrieved chunk with a
simple keyword-overlap heuristic against the question. It's crude
(it can't detect a paraphrase that reuses no question words), but it's
model-agnostic, so it's fair to use for comparing model A against B.
"""

import os
import re
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from rag_demo import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, chunk_text, load_pdfs

load_dotenv()

# PDF text can contain characters outside the Windows terminal's default
# codepage (e.g. accented author names); avoid crashing when printing them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_A = "openai/text-embedding-3-small"
MODEL_B = "baai/bge-large-en-v1.5"

TOP_K = 3
SUMMARY_PATH = "embedding_comparison.md"

TEST_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "How does multi-head attention work in the Transformer model?",
    "What are dilated convolutions and how do they help with dense prediction tasks like semantic segmentation?",
]

STOPWORDS = {
    "what", "is", "the", "a", "an", "how", "does", "do", "in", "of", "and",
    "to", "for", "are", "with", "like", "tasks", "task", "on", "it", "its",
}

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def embed(texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts with the given OpenRouter embedding model."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def keywords(text: str) -> set[str]:
    """Pull out meaningful (non-stopword) lowercase words from text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def relevance_score(question: str, chunk: str) -> float:
    """Fraction of the question's meaningful words that show up in the
    chunk. A crude but model-agnostic proxy for "is this chunk on topic",
    used because embedding distances from different models aren't
    directly comparable to each other.
    """
    q_words = keywords(question)
    if not q_words:
        return 0.0
    c_words = keywords(chunk)
    return len(q_words & c_words) / len(q_words)


def build_all_chunks() -> tuple[list[str], list[str], list[dict]]:
    """Load and chunk the PDFs once - both models embed the identical
    chunks, so any difference in results comes from the embedding model,
    not from different text being fed in.
    """
    documents = load_pdfs(DOCS_DIR)
    all_chunks, all_ids, all_metadatas = [], [], []
    for doc in documents:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['source']}-{i}")
            all_metadatas.append({"source": doc["source"], "chunk_index": i})
    return all_chunks, all_ids, all_metadatas


def build_collection(
    chroma_client: chromadb.Client,
    name: str,
    model: str,
    chunks: list[str],
    ids: list[str],
    metadatas: list[dict],
) -> chromadb.Collection:
    """Embed every chunk with `model` and store it in its own collection.
    Each model gets its own collection because the two models produce
    differently-sized vectors that can't share a collection.
    """
    collection = chroma_client.get_or_create_collection(name)
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        batch_embeddings = embed(batch_chunks, model)
        collection.add(
            ids=batch_ids,
            documents=batch_chunks,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )
    return collection


def retrieve(collection: chromadb.Collection, question: str, model: str, top_k: int) -> dict:
    query_embedding = embed([question], model)[0]
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)


def run_comparison() -> list[dict]:
    """Run every test question against both collections and gather the
    results (including relevance scores) needed for the console report
    and the markdown summary.
    """
    chunks, ids, metadatas = build_all_chunks()

    chroma_client = chromadb.Client()
    collection_a = build_collection(chroma_client, "compare_model_a", MODEL_A, chunks, ids, metadatas)
    collection_b = build_collection(chroma_client, "compare_model_b", MODEL_B, chunks, ids, metadatas)

    results = []
    for question in TEST_QUESTIONS:
        result_a = retrieve(collection_a, question, MODEL_A, TOP_K)
        result_b = retrieve(collection_b, question, MODEL_B, TOP_K)

        def summarize(result):
            hits = []
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            ):
                hits.append(
                    {
                        "id": f"{meta['source']}-{meta['chunk_index']}",
                        "source": meta["source"],
                        "chunk_index": meta["chunk_index"],
                        "distance": dist,
                        "relevance": relevance_score(question, doc),
                        "preview": doc.replace("\n", " ").strip()[:150],
                    }
                )
            return hits

        hits_a = summarize(result_a)
        hits_b = summarize(result_b)
        overlap = len({h["id"] for h in hits_a} & {h["id"] for h in hits_b})

        results.append({"question": question, "a": hits_a, "b": hits_b, "overlap": overlap})

    return results


def print_report(results: list[dict]) -> None:
    for entry in results:
        print("=" * 80)
        print("Question:", entry["question"])
        for label, model, hits in (("Model A", MODEL_A, entry["a"]), ("Model B", MODEL_B, entry["b"])):
            print(f"\n{label} ({model}):")
            for rank, hit in enumerate(hits, start=1):
                print(
                    f"  {rank}. [distance={hit['distance']:.4f} relevance={hit['relevance']:.2f}] "
                    f"{hit['source']} (chunk {hit['chunk_index']}): {hit['preview']}..."
                )
        print(f"\nChunks both models agreed on: {entry['overlap']}/{TOP_K}")
        print()


def average_relevance(results: list[dict], key: str) -> float:
    scores = [hit["relevance"] for entry in results for hit in entry[key]]
    return sum(scores) / len(scores) if scores else 0.0


def write_summary(results: list[dict]) -> None:
    avg_a = average_relevance(results, "a")
    avg_b = average_relevance(results, "b")
    winner = MODEL_A if avg_a > avg_b else MODEL_B if avg_b > avg_a else "Tie"

    lines = []
    lines.append("# Embedding Model Comparison\n")
    lines.append(
        f"Comparing **{MODEL_A}** (Model A) vs **{MODEL_B}** (Model B) for retrieval "
        f"quality on the PDFs in `docs/`, using {len(TEST_QUESTIONS)} test questions "
        f"and top-{TOP_K} retrieval from ChromaDB.\n"
    )
    lines.append(
        "Relevance is scored as the fraction of meaningful question words that "
        "appear in the retrieved chunk (a model-agnostic proxy, since raw "
        "ChromaDB distances from different embedding models are not on the same "
        "scale and can't be compared directly).\n"
    )

    lines.append("## Per-question results\n")
    for entry in results:
        lines.append(f"### {entry['question']}\n")
        lines.append(f"Chunks both models agreed on: {entry['overlap']}/{TOP_K}\n")
        for label, hits in (("Model A", entry["a"]), ("Model B", entry["b"])):
            lines.append(f"**{label}:**\n")
            for rank, hit in enumerate(hits, start=1):
                lines.append(
                    f"{rank}. `{hit['source']}` chunk {hit['chunk_index']} "
                    f"(distance={hit['distance']:.4f}, relevance={hit['relevance']:.2f}): "
                    f"{hit['preview']}...\n"
                )
        lines.append("")

    lines.append("## Summary\n")
    lines.append(f"- Average relevance, Model A (`{MODEL_A}`): **{avg_a:.2f}**")
    lines.append(f"- Average relevance, Model B (`{MODEL_B}`): **{avg_b:.2f}**")
    lines.append(f"- Better performer on this test set: **{winner}**\n")
    lines.append(
        f"{winner} retrieved chunks whose text overlapped more with the "
        "question's own vocabulary across the test questions, suggesting its "
        "embeddings placed the genuinely relevant passages closer to the "
        "question than the other model did. `openai/text-embedding-3-small` "
        "is a general-purpose commercial embedding model, while "
        "`baai/bge-large-en-v1.5` is an open-source model tuned specifically "
        "for retrieval tasks, so a close or reversed result here is plausible "
        "depending on the document domain.\n"
    )
    lines.append(
        "**Caveats:** this is a tiny test set (2 papers, 3 questions) and the "
        "relevance score is lexical, not semantic - a chunk that answers the "
        "question in different words would score low even if a human would "
        "call it relevant. Treat this as a directional signal, not a "
        "rigorous benchmark.\n"
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    results = run_comparison()
    print_report(results)
    write_summary(results)
