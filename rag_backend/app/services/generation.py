import logging

import ollama

from app.core.config import settings
from app.services.retrieval import retrieve_documents

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below, which was retrieved from a set of documents. "
    "If the context does not contain enough information to answer the "
    "question, say so honestly instead of making something up. "
    "Always ground your answer in the given context and mention which "
    "source(s) support your answer when relevant."
)


def build_prompt(question: str, retrieved_docs: list[dict]) -> str:
    """Combine the retrieved context with the user's question into one prompt."""
    if retrieved_docs:
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.get("source", "Unknown source")
            page = doc.get("page", "Unknown")
            text = doc.get("text", "")
            context_blocks.append(f"[{i}] Source: {source} (Page {page})\n{text}")
        context = "\n\n".join(context_blocks)
    else:
        context = "No relevant context was found in the document collection."

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(question: str, n_results: int | None = None) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks, build a grounded prompt, and call the LLM."""
    retrieved_docs = retrieve_documents(question, n_results)
    prompt = build_prompt(question=question, retrieved_docs=retrieved_docs)

    client = ollama.Client(host=settings.ollama_host)
    response = client.chat(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"]
    logger.info("Generated answer for question: %s", question)
    return answer, retrieved_docs
