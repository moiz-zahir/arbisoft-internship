"""
RAG (Retrieval-Augmented Generation) demo.

RAG is a technique for answering questions using an LLM that wasn't
trained on your specific documents. Instead of relying on the model's
built-in knowledge (which can be outdated, generic, or just wrong for
your domain), we:

  1. Break our own documents into small chunks of text.
  2. Convert each chunk into an "embedding" - a vector of numbers that
     captures its meaning - and store those vectors in a database.
  3. When a user asks a question, embed the question the same way and
     search the database for the chunks whose meaning is closest to it
     ("retrieval").
  4. Paste those chunks into the prompt we send the LLM, and ask it to
     answer using only that context ("augmented generation").

This grounds the model's answer in our actual source material instead
of whatever it happened to memorize during training, and lets us cite
exactly which chunks were used.
"""

import glob
import os
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

# PDF text can contain characters outside the Windows terminal's default
# codepage (e.g. accented author names); avoid crashing when printing them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "anthropic/claude-haiku-4.5"

DOCS_DIR = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def load_pdfs(docs_dir: str) -> list[dict]:
    """Step 1: Load every PDF in docs_dir and extract its raw text.

    We need plain text before we can chunk and embed it - embeddings
    models don't understand PDF layout/formatting, only text.
    """
    documents = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.pdf"))):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        documents.append({"source": os.path.basename(path), "text": text})
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Step 2: Split text into overlapping fixed-size chunks.

    Whole documents are too big to embed/search effectively and won't
    fit in a prompt. Small chunks let retrieval pinpoint just the
    relevant passage. The overlap keeps sentences/ideas that straddle
    a chunk boundary from being split apart and losing context.
    """
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Call OpenAI's text-embedding-3-small (via OpenRouter) to turn
    text into vectors that can be compared for semantic similarity.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_collection() -> chromadb.Collection:
    """Step 3: Chunk every PDF and store the chunks + embeddings in ChromaDB.

    ChromaDB is a vector database: it stores embeddings alongside the
    original text/metadata and can quickly find the nearest vectors to
    a query embedding, which is what makes retrieval fast at scale.
    """
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("rag_demo")

    documents = load_pdfs(DOCS_DIR)

    all_chunks = []
    all_ids = []
    all_metadatas = []
    for doc in documents:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['source']}-{i}")
            all_metadatas.append({"source": doc["source"], "chunk_index": i})

    # Embed in batches so we don't send an oversized request to the API.
    batch_size = 100
    for start in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[start : start + batch_size]
        batch_ids = all_ids[start : start + batch_size]
        batch_metadatas = all_metadatas[start : start + batch_size]
        batch_embeddings = embed(batch_chunks)
        collection.add(
            ids=batch_ids,
            documents=batch_chunks,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )

    return collection


def retrieve(collection: chromadb.Collection, question: str, top_k: int) -> dict:
    """Step 4: Embed the question and find the most relevant chunks.

    We embed the question with the same model used for the chunks so
    they live in the same vector space, then ask ChromaDB for the
    nearest neighbors - the chunks most likely to contain the answer.
    """
    query_embedding = embed([question])[0]
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)


def generate_answer(question: str, chunks: list[str]) -> str:
    """Step 5: Ask claude-haiku-4.5 (via OpenRouter) to answer the
    question using only the retrieved chunks as context. This is the
    "augmented generation" half of RAG - the model reasons over real
    source text instead of just its training data.
    """
    context = "\n\n---\n\n".join(chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def answer_question(collection: chromadb.Collection, question: str, top_k: int = TOP_K) -> None:
    """Step 6: Run retrieval + generation and print the answer plus
    the chunks that were used, so the result is traceable back to a
    source document instead of being an opaque black box.
    """
    results = retrieve(collection, question, top_k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    answer = generate_answer(question, chunks)

    print("Question:", question)
    print("\nAnswer:", answer)
    print("\nChunks used:")
    for metadata, chunk in zip(metadatas, chunks):
        preview = chunk.replace("\n", " ").strip()[:150]
        print(f"  - {metadata['source']} (chunk {metadata['chunk_index']}): {preview}...")


if __name__ == "__main__":
    collection = build_collection()
    answer_question(collection, "What is attention mechanism in transformers?")
