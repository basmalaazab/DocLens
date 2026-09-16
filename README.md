# RAG-Powered Document Assistant

## 1. Overview

RAG-Powered Document Assistant is an AI-powered question-answering system built using Retrieval-Augmented Generation (RAG).

The system processes a collection of Deep Learning documents, converts their content into semantic vector representations, retrieves relevant information for a user query, and generates grounded answers using a local Large Language Model.

The project also includes an Extended Computer Vision component that uses YOLO-based document layout detection and OCR to extract information from document images and integrate it into the RAG workflow.

The project consists of:

- A **RAG notebook** for document processing, embeddings, retrieval, evaluation, and the Computer Vision extension.
- A **FastAPI backend** that provides the RAG API.
- A **web frontend** built with HTML, CSS, and JavaScript.
- A **persistent ChromaDB vector store** containing the document embeddings.
- A **YOLO document-layout model** for the vision extension.

---

# 2. Project Objectives

The main objectives of the project are to:

- Process and inspect PDF documents.
- Extract text while preserving document and page metadata.
- Split documents into overlapping text chunks.
- Generate semantic embeddings.
- Store embeddings in a persistent vector database.
- Retrieve relevant document chunks for user queries.
- Generate answers grounded only in retrieved context.
- Provide source and page information with answers.
- Evaluate the system using multiple questions.
- Handle questions that are outside the document knowledge base.
- Extend the RAG pipeline with document-layout detection and OCR.

---

# 3. Domain and Dataset

## Domain

The project focuses on the **Deep Learning** domain.

The knowledge base contains educational material covering topics such as:

- Neural networks
- Backpropagation
- Gradient descent
- Activation functions
- Convolutional Neural Networks
- Recurrent Neural Networks
- Dropout
- Vanishing gradients
- Self-attention
- Transformers
- Overfitting

## Source Documents

The RAG knowledge base contains three PDF documents:

```text
rag_backend/data/pdfs/
├── book.pdf
├── dive_into_deep_learning.pdf
└── UnderstandingDeepLearning_02_09_26_C.pdf
```

The PDFs are processed page-by-page using `pypdf`.

Each extracted page is associated with metadata containing the source document and page number.

Empty pages are removed before the chunking stage.

---

# 4. System Architecture

The complete system follows this architecture:

```text
                         +----------------------+
                         |        User          |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |       Frontend       |
                         |    HTML/CSS/JS       |
                         +----------+-----------+
                                    |
                                    | HTTP
                                    v
                         +----------------------+
                         |       FastAPI        |
                         |       Backend        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |      Retrieval       |
                         |       ChromaDB        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Relevant Context   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Grounded Prompt    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |  Ollama / Llama 3.2  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Answer + Sources   |
                         +----------------------+
```

For the Extended Vision workflow:

```text
Document Image
      |
      v
YOLO Layout Detection
      |
      v
Content-Type Classification
      |
      +------> Text Regions ------> OCR
      |
      +------> Tables / Pictures
      |
      v
Image Content Information
      |
      v
User Question + Image Context
      |
      v
RAG Retrieval
      |
      v
Grounded LLM Response
```

---

# 5. RAG Pipeline

The RAG pipeline consists of the following stages:

```text
PDF Documents
      |
      v
Document Loading
      |
      v
Page-Level Parsing
      |
      v
Text Chunking
      |
      v
Embedding Generation
      |
      v
Persistent ChromaDB
      |
      v
Semantic Retrieval
      |
      v
Retrieved Context
      |
      v
Grounded Prompt
      |
      v
Llama 3.2
      |
      v
Answer + Sources
```

---

# 6. Document Processing

The documents are loaded using `pypdf`.

The notebook processes the PDFs page-by-page instead of treating each document as a single text block.

This allows the system to preserve page-level metadata, which is later used for source references.

The document processing stage includes:

- PDF discovery.
- PDF parsing.
- Page extraction.
- Empty-page filtering.
- Source metadata creation.
- Page metadata creation.

---

# 7. Chunking Strategy

The project uses LangChain's `RecursiveCharacterTextSplitter`.

The final configuration is:

```text
Chunk size:    1000 characters
Chunk overlap: 200 characters
```

## Why 1000 characters?

A chunk size of 1000 characters provides enough surrounding context for Deep Learning concepts while keeping individual retrieved passages focused.

## Why 200 characters overlap?

The overlap preserves contextual information between neighboring chunks and reduces the possibility of splitting an explanation in a way that removes important surrounding information.

The recursive splitter also attempts to preserve natural text boundaries when creating chunks.

---

# 8. Embeddings

The project uses Sentence Transformers for semantic embedding generation.

```text
Embedding model:
all-MiniLM-L6-v2
```

Each document chunk is converted into a numerical vector representing its semantic meaning.

These vectors are stored in ChromaDB and used during semantic similarity retrieval.

---

# 9. Vector Database

The project uses **ChromaDB** as the vector database.

The vector store is persisted locally at:

```text
rag_backend/data/vector_store/
```

The collection used by the application is:

```text
deep_learning_docs
```

The final vector database contains:

```text
5,666 chunks
```

The persisted vector store allows the backend to load the existing embeddings instead of rebuilding the complete database every time the application starts.

The current vector-store files include:

```text
rag_backend/data/vector_store/
├── chroma.sqlite3
└── af6bb3f5-0fec-45f0-b3ee-263bc4f9980d/
    ├── data_level0.bin
    ├── header.bin
    ├── index_metadata.pickle
    ├── length.bin
    └── link_lists.bin
```

---

# 10. Retrieval

For every user question, the backend performs semantic similarity search against the ChromaDB collection.

The default retrieval configuration returns the top five relevant chunks.

Each retrieved result contains:

- Text content
- Source document
- Page number

The retrieved information is then inserted into the grounded generation prompt.

This allows the LLM to answer based on the retrieved document context instead of relying only on its pretrained knowledge.

---

# 11. Grounded Generation

The project uses Ollama to run the local LLM.

```text
LLM:
llama3.2:3b
```

The generation prompt instructs the model to:

- Use only the provided context.
- Avoid unsupported information.
- State when the required information is not available in the documents.
- Provide a clear and concise answer.
- Mention relevant sources and page numbers.

This design is intended to reduce hallucinated information and keep generated responses connected to the document collection.

---

# 12. Evaluation

The RAG system was evaluated using ten questions related to Deep Learning.

The evaluation questions cover:

1. Backpropagation
2. Gradient descent
3. Activation functions
4. Convolutional Neural Networks
5. CNNs vs RNNs
6. Dropout
7. Vanishing gradients
8. Self-attention
9. Transformers
10. Overfitting

Each answer was manually reviewed for:

- Correctness
- Groundedness in the retrieved context

## Results

| Metric | Result |
|---|---:|
| Questions evaluated | 10 |
| Correct answers | 9 / 10 |
| Correctness | 90% |
| Grounded answers | 9 / 10 |
| Groundedness | 90% |

One of the evaluated questions, the comparison between CNNs and RNNs, was not sufficiently supported by the retrieved context.

This represents a retrieval limitation: relevant information may exist somewhere in the document collection while the retrieved chunks do not contain enough information to answer a particular question.

---

# 13. Failure Cases

Additional questions outside the Deep Learning knowledge base were used to test the grounding behavior.

Examples include:

```text
What is the capital of France?

What is the boiling point of water at sea level?

How do I repair a car engine?
```

These questions are not covered by the project's document collection.

The intended behavior is for the system to state that the information is not available in the provided documents rather than generating an unsupported answer.

## Mitigation

The grounded generation prompt explicitly instructs the model not to use information that is not supported by the retrieved context.

Possible future improvements include:

- Retrieval confidence thresholds.
- Query rewriting.
- Hybrid semantic and keyword retrieval.
- Reranking.
- Larger evaluation datasets.
- Automated groundedness evaluation.

---

# 14. Computer Vision Extension

The project includes a Computer Vision extension for document image analysis.

The YOLO model is stored at:

```text
rag_backend/models/yolov8n-doclaynet.pt
```

The model is used for document-layout detection.

It can detect document elements such as:

- Text
- Title
- Table
- Picture
- Caption
- List-item
- Section-header
- Footnote
- Formula

The detected regions are classified according to their content type and can be incorporated into the RAG workflow.

---

# 15. OCR

Text-like regions detected by the layout model can be processed using OCR.

The project uses:

```text
Tesseract OCR
```

OCR is applied to text-oriented regions such as:

- Text
- Title
- Caption
- List-item
- Section-header
- Footnote

Tables and pictures are handled as separate visual content types rather than being treated as ordinary text regions.

---

# 16. Image-Based RAG

The Computer Vision extension connects image understanding with the existing RAG pipeline.

The process is:

```text
Uploaded Image
      |
      v
YOLO Document Layout Detection
      |
      v
Detected Regions
      |
      +---- Text-like Regions ----> OCR
      |
      +---- Tables / Pictures
      |
      v
Extracted Image Information
      |
      v
Question + Image Information
      |
      v
Semantic Retrieval
      |
      v
Retrieved Document Context
      |
      v
Grounded LLM Generation
      |
      v
Answer + Sources
```

The same Deep Learning ChromaDB collection is used as the knowledge source.

---

# 17. Backend

The backend is implemented using FastAPI.

The backend source code is located in:

```text
rag_backend/
```

## Backend Structure

```text
rag_backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── query.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── query.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── generation.py
│   │   ├── retrieval.py
│   │   └── vision.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py
│
├── data/
│   ├── pdfs/
│   │   ├── book.pdf
│   │   ├── dive_into_deep_learning.pdf
│   │   └── UnderstandingDeepLearning_02_09_26_C.pdf
│   │
│   └── vector_store/
│       └── ...
│
├── models/
│   └── yolov8n-doclaynet.pt
│
├── tests/
│   ├── __init__.py
│   └── test_query.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Main Backend Components

### `app/main.py`

Initializes the FastAPI application and configures middleware and application startup.

### `app/api/routes/query.py`

Contains the API route responsible for processing user queries.

### `app/core/config.py`

Contains application configuration and environment-based settings.

### `app/schemas/query.py`

Defines request and response schemas used by the API.

### `app/services/retrieval.py`

Handles retrieval of relevant document chunks from ChromaDB.

### `app/services/generation.py`

Handles grounded answer generation through the configured LLM.

### `app/services/vision.py`

Contains the Computer Vision functionality used by the extended image-based workflow.

### `app/utils/logging_config.py`

Provides logging configuration for the backend.

---

# 18. API Reference

## Health Check

```http
GET /health
```

Used to verify that the backend is running.

Example:

```bash
curl http://localhost:8000/health
```

## Query

```http
POST /query
```

The query endpoint receives a user question and returns a response generated from retrieved document context.

Example:

```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is backpropagation?\"}"
```

FastAPI also provides interactive API documentation when the backend is running.

---

# 19. Backend Setup

From the project root:

```bash
cd rag_backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the required model is available:

```text
llama3.2:3b
```

Start the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

The API runs locally on port `8000`.

---

# 20. Frontend

The frontend is a lightweight web interface built using:

- HTML
- CSS
- JavaScript

The frontend files are located at:

```text
rag_frontend/rag_frontend/
```

## Frontend Structure

```text
rag_frontend/
└── rag_frontend/
    ├── .env
    ├── app.js
    ├── config.js
    ├── config.js.example
    ├── index.html
    ├── README.md
    └── style.css
```

### `index.html`

Contains the main user interface.

### `style.css`

Contains the visual styling of the application.

### `app.js`

Handles frontend interaction and communication with the backend.

### `config.js`

Contains the frontend configuration.

### `config.js.example`

Provides an example configuration without exposing local environment-specific values.

---

# 21. Frontend Setup

Navigate to the frontend directory:

```bash
cd rag_frontend/rag_frontend
```

Configure the backend URL using the provided configuration example.

The frontend should point to the running FastAPI backend.

The application can then be opened through the frontend's local serving method.

---

# 22. Environment Variables

Environment-specific configuration should not be committed to GitHub.

The project contains local `.env` files:

```text
rag_backend/.env
rag_frontend/rag_frontend/.env
```

These files should remain local.

Example configuration files should be used instead:

```text
rag_frontend/rag_frontend/config.js.example
```

The `.gitignore` files are configured to prevent environment files and other sensitive or generated content from being committed.

---

# 23. Tests

Backend tests are located at:

```text
rag_backend/tests/test_query.py
```

The project uses:

- Pytest
- FastAPI TestClient

Tests cover API behavior including valid requests and invalid input handling.

Run the tests from the backend directory:

```bash
pytest
```

---

# 24. Notebook

The complete experimental RAG pipeline is provided in:

```text
notebooks/rag_pipeline.ipynb
```

The notebook documents the development process from raw documents to the evaluated RAG system.

It includes:

- Document loading.
- Document inspection.
- PDF parsing.
- Chunking.
- Embedding generation.
- ChromaDB creation.
- Retrieval experiments.
- Grounded prompting.
- LLM generation.
- Ten-question evaluation.
- Failure-case testing.
- Vector-store configuration.
- YOLO document-layout detection.
- OCR.
- Image-based RAG.

The notebook represents the experimentation and evaluation stage, while the FastAPI backend provides the application API.

---

# 25. Exported Configuration

The final RAG configuration is:

```json
{
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2",
    "vector_database": "ChromaDB",
    "collection_name": "deep_learning_docs",
    "llm_model": "llama3.2:3b"
}
```

This configuration describes the components used to create and serve the final RAG pipeline.

---

# 26. Complete Project Structure

```text
rag_powered_assistant/
│
├── .gitignore
├── run.bat
│
├── rag_backend/
│   ├── .env
│   ├── .gitignore
│   ├── README.md
│   ├── requirements.txt
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       └── query.py
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── query.py
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── generation.py
│   │   │   ├── retrieval.py
│   │   │   └── vision.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logging_config.py
│   │
│   ├── data/
│   │   ├── pdfs/
│   │   │   ├── book.pdf
│   │   │   ├── dive_into_deep_learning.pdf
│   │   │   └── UnderstandingDeepLearning_02_09_26_C.pdf
│   │   │
│   │   └── vector_store/
│   │       └── ...
│   │
│   ├── models/
│   │   └── yolov8n-doclaynet.pt
│   │
│   └── tests/
│       ├── __init__.py
│       └── test_query.py
│
├── rag_frontend/
│   └── rag_frontend/
│       ├── .env
│       ├── app.js
│       ├── config.js
│       ├── config.js.example
│       ├── index.html
│       ├── README.md
│       └── style.css
│
└── notebooks/
    └── rag_pipeline.ipynb
```

---

# 27. Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Backend Framework | FastAPI |
| ASGI Server | Uvicorn |
| RAG | LangChain |
| PDF Processing | pypdf |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| LLM Runtime | Ollama |
| LLM | Llama 3.2 3B |
| Computer Vision | Ultralytics YOLO |
| Document Layout Model | YOLOv8 DocLayNet |
| OCR | Tesseract |
| Frontend | HTML / CSS / JavaScript |
| Testing | Pytest / FastAPI TestClient |

---

# 28. Reproducibility

To reproduce the RAG pipeline:

1. Install the Python dependencies.
2. Place the source PDFs in the configured data directory.
3. Run the notebook from top to bottom.
4. Perform document parsing and chunking.
5. Generate embeddings using `all-MiniLM-L6-v2`.
6. Build the persistent ChromaDB collection.
7. Run the retrieval experiments.
8. Evaluate the ten test questions.
9. Export the resulting vector store and configuration.
10. Start the FastAPI backend.
11. Start the frontend and connect it to the backend.

The notebook provides the experimental implementation and evaluation, while the backend consumes the resulting persisted vector store.

---

# 29. Screenshots

Screenshots can be added to this section to demonstrate the working application.

Recommended screenshots:

### Frontend

```text
[Add frontend screenshot here]
```

### RAG Question and Answer

```text
[Add example question and grounded answer screenshot here]
```

### Retrieved Sources

```text
[Add screenshot showing source/page information here]
```

### Computer Vision Extension

```text
[Add YOLO/OCR result screenshot here]
```

---

# 30. Assignment Requirements Coverage

| Requirement | Project Implementation |
|---|---|
| Load and inspect documents | Implemented in notebook |
| PDF parsing | `pypdf` |
| Chunking strategy | RecursiveCharacterTextSplitter |
| Chunk size | 1000 |
| Chunk overlap | 200 |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| Persistent vector store | `rag_backend/data/vector_store/` |
| Retrieval | Semantic similarity search |
| Grounded generation | Context-restricted prompt |
| Local LLM | Ollama / Llama 3.2 3B |
| Retrieval evaluation | 10 questions |
| Correctness evaluation | 90% |
| Groundedness evaluation | 90% |
| Failure cases | Included |
| Backend | FastAPI |
| Health endpoint | `/health` |
| Query endpoint | `/query` |
| Backend tests | Pytest / TestClient |
| Frontend | HTML / CSS / JavaScript |
| Notebook | `notebooks/rag_pipeline.ipynb` |
| Vision extension | YOLO document-layout detection |
| OCR | Tesseract |
| Image-based RAG | Implemented as an extension |

---

# 31. Limitations and Future Improvements

The current implementation can be improved through:

- Better retrieval for comparison questions.
- Hybrid keyword and semantic retrieval.
- Retrieval reranking.
- Query rewriting.
- Retrieval confidence thresholds.
- More extensive evaluation.
- Automated groundedness metrics.
- Improved table and diagram understanding.
- More robust OCR processing.
- Improved image-based retrieval.
- Production deployment configuration.

---

# 32. Conclusion

The project implements a complete RAG-based Document Assistant for Deep Learning content.

It covers the complete pipeline from PDF processing and chunking to semantic retrieval and grounded LLM generation.

The system uses a persistent ChromaDB vector store containing 5,666 document chunks and a local Llama 3.2 model through Ollama.

The evaluation achieved 90% correctness and 90% groundedness across ten manually reviewed questions.

The project also includes an Extended Computer Vision pipeline using YOLO document-layout detection and Tesseract OCR, allowing information extracted from document images to be incorporated into the RAG workflow.

The final application combines the experimental notebook, FastAPI backend, persistent vector database, Computer Vision components, and browser-based frontend into one end-to-end system.
