# RAG Document Assistant Backend

FastAPI backend for a Retrieval-Augmented Generation (RAG) assistant over Deep Learning PDFs, with an optional image pipeline using YOLO document-layout detection and OCR.

## Architecture

```text
PDFs -> PDF text extraction -> chunking -> embeddings -> ChromaDB
                                                    |
Question ------------------------------------------+-> retrieval -> prompt -> Ollama -> answer

Image + Question -> YOLO layout detection -> OCR -> Question + OCR -> retrieval -> Ollama -> answer
```

## What this version fixes

- Loads an existing ChromaDB vector store when available.
- Automatically rebuilds the vector store from `data/pdfs/` when it is missing or empty.
- Uses configurable chunk size, overlap, Top-K, and ingestion batch size.
- Uses RapidOCR first, with pytesseract as a fallback.
- OCRs text-bearing DocLayNet regions, including tables and formulas.
- Falls back to OCR on the full image when layout detection/OCR produces no text.
- Does **not** invent or inject table values based on keywords detected by OCR.
- Loads the YOLO model once at startup and can download it from Hugging Face if the local file is missing.
- Keeps CORS flexible for local frontend development.
- Cleans temporary uploaded images after processing.
- Keeps generated vector-store files out of Git.

## Setup

### 1. Add the PDFs

Put the source PDFs in:

```text
data/pdfs/
```

The first startup will create:

```text
data/vector_store/
```

automatically. You do not need to commit the vector store to GitHub.

### 2. Environment

Copy `.env.example` to `.env` and change values if needed.

### 3. Python environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

### 4. Ollama

Install/run Ollama and pull the configured model:

```bash
ollama serve
ollama pull llama3.2:3b
```

### 5. YOLO model

A copy of `yolov8n-doclaynet.pt` is included in this package. If it is removed, the application can download the configured model from Hugging Face at startup.

## Run

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Endpoints

### `GET /health`

Returns server/vector-store status and chunk count.

### `POST /query`

JSON:

```json
{"question": "What is backpropagation?"}
```

Returns an answer and the retrieved PDF sources.

### `POST /query-image`

Multipart form with:

- `question`: user's question
- `image`: JPG/PNG/WEBP/BMP/TIFF image

The image is analyzed with YOLO + OCR. The extracted text is included directly in the LLM context and is also used for retrieval.

## Tests

```bash
pytest
```
