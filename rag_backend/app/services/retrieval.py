import logging
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import settings

logger = logging.getLogger(__name__)

_collection = None


def _build_collection(client, embedding_function, pdf_dir: Path):
    """Build the Chroma collection from the PDFs in pdf_dir."""
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_dir.resolve()}"
        )

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {pdf_dir.resolve()}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    chunk_id = 0

    for pdf_path in pdf_files:
        logger.info("Reading %s", pdf_path.name)
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            for chunk in splitter.split_text(text):
                if not chunk.strip():
                    continue
                documents.append(chunk)
                metadatas.append({"source": pdf_path.name, "page": page_number})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

    if not documents:
        raise RuntimeError(
            f"No text chunks were created from PDFs in: {pdf_dir.resolve()}"
        )

    logger.info("Total chunks created: %d", len(documents))

    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.collection_name,
        embedding_function=embedding_function,
    )

    for start in range(0, len(documents), settings.ingest_batch_size):
        end = min(start + settings.ingest_batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        logger.info("Added chunks %d-%d", start, end)

    return collection


def init_vector_store() -> None:
    """Load an existing vector store or build it automatically from PDFs."""
    global _collection

    vector_store_dir = Path(settings.vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    embedding_function = (
        chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    )
    client = chromadb.PersistentClient(path=str(vector_store_dir))

    try:
        collection = client.get_collection(
            name=settings.collection_name,
            embedding_function=embedding_function,
        )
        count = collection.count()
        if count > 0:
            _collection = collection
            logger.info("Loaded existing vector store. Chunks stored: %d", count)
            return
        logger.info("Existing collection is empty; rebuilding from PDFs.")
    except Exception as exc:
        logger.info("Could not load existing collection (%s); building from PDFs.", exc)

    pdf_dir = Path(settings.pdf_dir)
    _collection = _build_collection(client, embedding_function, pdf_dir)
    logger.info("Vector store ready. Chunks stored: %d", _collection.count())


def get_chunk_count() -> int:
    if _collection is None:
        raise RuntimeError("Vector store not initialised")
    return _collection.count()


def retrieve_documents(question: str, n_results: int | None = None) -> list[dict]:
    if _collection is None:
        raise RuntimeError("Vector store not initialised")

    requested = n_results or settings.top_k
    available = _collection.count()
    if available == 0:
        return []
    requested = min(requested, available)

    results = _collection.query(query_texts=[question], n_results=requested)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved_docs = []
    for text, metadata in zip(documents, metadatas):
        retrieved_docs.append(
            {
                "text": text,
                "source": metadata.get("source", "Unknown source"),
                "page": metadata.get("page", "Unknown"),
            }
        )

    logger.info("Retrieved %d documents for question", len(retrieved_docs))
    return retrieved_docs
