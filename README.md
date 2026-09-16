# RAG-Powered Document Assistant

## 1. Overview

RAG-Powered Document Assistant is an AI-powered question-answering system that uses Retrieval-Augmented Generation (RAG) to answer questions from a collection of Deep Learning documents.

The system combines document processing, semantic search, vector databases, and a local Large Language Model (LLM) to provide answers grounded in the provided documents.

An extended Computer Vision component is also included to analyze uploaded document images using YOLO-based document layout detection and OCR, and incorporate the detected content into the RAG pipeline.

The project is divided into three main parts:

- **Notebook:** Data processing, chunking, embeddings, vector store creation, retrieval experiments, evaluation, and the Computer Vision extension.
- **Backend:** FastAPI service that loads the persisted vector store and serves grounded queries.
- **Frontend:** Web interface for interacting with the RAG assistant.

---

## 2. Project Objectives

The project aims to build a complete RAG-based AI application capable of:

- Processing multiple PDF documents.
- Splitting documents into meaningful text chunks.
- Generating semantic embeddings.
- Storing embeddings in a persistent vector database.
- Retrieving relevant document passages for a user query.
- Generating grounded answers using a local LLM.
- Providing source and page information for retrieved content.
- Evaluating answer correctness and groundedness.
- Handling questions that are outside the provided document collection.
- Extending the system with document image analysis using Computer Vision and OCR.

---

## 3. Domain and Dataset

### Domain

The selected domain is **Deep Learning**.

The knowledge base consists of educational Deep Learning materials covering topics such as:

- Neural networks
- Backpropagation
- Gradient descent
- Activation functions
- Convolutional Neural Networks
- Recurrent Neural Networks
- Dropout
- Vanishing gradients
- Attention mechanisms
- Transformers
- Overfitting

### Source Documents

The RAG pipeline was built using three PDF documents:

1. `book.pdf`
2. `UnderstandingDeepLearning_02_09_26_C.pdf`
3. `dive_into_deep_learning.pdf`

The documents were parsed page-by-page, and each extracted page was stored with metadata identifying its source document and page number.

Empty pages were removed before chunking.

---

## 4. RAG Pipeline

The complete RAG pipeline follows this workflow:

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
Sentence-Transformer Embeddings
      |
      v
ChromaDB Vector Store
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
Llama 3.2 LLM
      |
      v
Answer + Sources
```

---

## 5. Document Processing

The documents were loaded using `pypdf`.

Each PDF was processed page-by-page so that page information could be preserved as metadata.

Each document chunk contains metadata including:

- Source document
- Page number

This metadata is later used to provide source references with generated answers.

### Document Inspection

The notebook performs document inspection before building the vector database, including:

- Detecting available PDF files.
- Counting documents.
- Parsing pages.
- Identifying empty pages.
- Removing unusable page content.
- Preserving document and page metadata.

---

## 6. Chunking Strategy

The project uses LangChain's `RecursiveCharacterTextSplitter`.

```text
Chunk size: 1000 characters
Chunk overlap: 200 characters
```

### Why this strategy?

A chunk size of 1000 characters provides enough surrounding context for most Deep Learning explanations while keeping retrieved passages reasonably focused.

A 200-character overlap helps preserve context between adjacent chunks and reduces the chance of losing information that crosses chunk boundaries.

The recursive splitting strategy also attempts to split text at natural boundaries before falling back to smaller separators.

---

## 7. Embeddings

The project uses:

```text
SentenceTransformer
Model: all-MiniLM-L6-v2
```

The embedding model converts each text chunk into a numerical vector representing its semantic meaning.

This allows the system to retrieve passages based on semantic similarity rather than exact keyword matching.

---

## 8. Vector Database

The project uses **ChromaDB** as the vector database.

The collection is:

```text
Collection: deep_learning_docs
```

The vector store is persisted to disk so that it can be reused by the backend without rebuilding the embeddings every time the application starts.

The final vector collection contains:

```text
5,666 chunks
```

The notebook also exports the main RAG configuration:

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

---

## 9. Retrieval

For each user question, the system performs semantic similarity search against the ChromaDB collection.

The default retrieval configuration returns the top:

```text
5 relevant chunks
```

Each retrieved result contains:

- Text content
- Source document
- Page number

The retrieved passages are then passed to the generation component as context.

---

## 10. Grounded Generation

The project uses a local Ollama model:

```text
llama3.2:3b
```

The LLM receives the user's question together with the retrieved document context.

The generation prompt instructs the model to:

- Answer using only the provided context.
- Avoid unsupported information.
- State when the required information is not available.
- Provide a concise answer.
- Mention relevant source documents and page numbers.

This grounding strategy is designed to reduce unsupported answers and keep responses connected to the document collection.

---

# 11. Evaluation

The RAG system was evaluated using 10 Deep Learning questions.

The evaluation covers topics including:

- Backpropagation
- Gradient descent
- Activation functions
- CNNs
- CNN vs RNN
- Dropout
- Vanishing gradients
- Self-attention
- Transformers
- Overfitting

Each question was manually reviewed for:

- Answer correctness
- Groundedness in the retrieved context

## Evaluation Results

| Metric | Result |
|---|---:|
| Questions evaluated | 10 |
| Correct answers | 9/10 |
| Correctness | 90% |
| Grounded answers | 9/10 |
| Groundedness | 90% |

The main failure case was the question comparing CNNs and RNNs. The retrieved context did not explicitly contain enough information to support a direct comparison, so the generated answer was considered insufficiently grounded.

This demonstrates the importance of retrieval quality in a RAG system: even when relevant concepts exist in the overall document collection, the answer can fail if the retrieved context does not contain the required information.

---

## 12. Failure Cases and Mitigation

Additional out-of-domain questions were tested, including questions about:

- The capital of France.
- The boiling point of water at sea level.
- Repairing a car engine.

These questions are outside the Deep Learning document collection.

The intended behavior is for the assistant to state that the requested information is not available in the provided documents instead of relying on unsupported external knowledge.

### Mitigation

The generation prompt explicitly instructs the LLM not to use information that is not supported by the retrieved context.

Future improvements could include:

- Retrieval confidence thresholds.
- Better query rewriting.
- Hybrid keyword + semantic retrieval.
- Reranking retrieved chunks.
- More comprehensive evaluation datasets.
- Automated groundedness evaluation.

---

# 13. Computer Vision Extension

The project includes an Extended Track component for analyzing document images.

The Computer Vision pipeline uses a YOLO-based document layout detection model.

The model is:

```text
yolov8n-doclaynet.pt
```

It is based on the DocLayNet document-layout dataset.

The detector can identify document regions such as:

- Text
- Title
- Table
- Picture
- Caption
- List-item
- Section-header
- Footnote
- Formula

---

## 14. OCR

OCR is applied to text-like detected regions.

The system uses:

```text
Tesseract OCR
```

The OCR process extracts text from detected regions such as:

- Text
- Title
- Caption
- List-item
- Section-header
- Footnote

Non-text regions such as tables and pictures are classified separately instead of being directly passed through OCR.

---

## 15. Image-Based RAG

The image pipeline combines Computer Vision and RAG.

The workflow is:

```text
Uploaded Document Image
          |
          v
YOLO Layout Detection
          |
          v
Content-Type Classification
          |
          +------> Text Regions
          |             |
          |             v
          |         OCR Extraction
          |
          +------> Tables / Pictures
          |
          v
Image Content Description
          |
          v
User Question + Image Context
          |
          v
Semantic Retrieval
          |
          v
Grounded LLM Generation
          |
          v
Answer + Sources
```

The detected image information is combined with the user's question and used to retrieve relevant information from the same Deep Learning vector store.

This allows visual document information to participate in the RAG process.

---

# 16. Project Architecture

```text
                         +----------------------+
                         |     User / Browser   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |       Frontend       |
                         |   HTML/CSS/JavaScript |
                         +----------+-----------+
                                    |
                                    | HTTP
                                    v
                         +----------------------+
                         |       FastAPI        |
                         |       Backend        |
                         +----------+-----------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
          +-------------------+           +-------------------+
          |     Retrieval     |           | Vision Component  |
          |     ChromaDB      |           | YOLO + OCR        |
          +---------+---------+           +---------+---------+
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
                         +----------------------+
                         |   Retrieved Context  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Ollama / Llama 3.2 |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Grounded AI Response  |
                         +----------------------+
```

---

# 17. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Server | Uvicorn |
| RAG Framework | LangChain |
| PDF Processing | pypdf |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| LLM Runtime | Ollama |
| LLM | Llama 3.2 3B |
| Computer Vision | YOLO / Ultralytics |
| OCR | Tesseract |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest + FastAPI TestClient |

---

# 18. Project Structure

```text
rag-powered-assistant/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── utils/
│   │   └── main.py
│   │
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   └── ...
│
├── notebooks/
│   └── rag_pipeline.ipynb
│
├── README.md
├── .gitignore
└── run.bat
```

---

# 19. Notebook

The complete RAG development and evaluation process is documented in:

```text
notebooks/rag_pipeline.ipynb
```

The notebook contains:

1. Document loading and inspection.
2. PDF page parsing.
3. Text cleaning.
4. Chunking.
5. Embedding generation.
6. ChromaDB vector store creation.
7. Semantic retrieval.
8. Grounded prompt construction.
9. Ollama LLM generation.
10. Evaluation using 10 questions.
11. Failure-case testing.
12. Vector store configuration export.
13. YOLO document-layout detection.
14. OCR processing.
15. Image-based RAG integration.

The notebook serves as the experimental and evaluation part of the project, while the backend provides the application-facing API.

---

# 20. Backend

The backend is implemented using FastAPI.

It loads the required RAG components during application startup so that the vector store and generation components are initialized once rather than being recreated for every request.

## Main Endpoints

### Health Check

```http
GET /health
```

Used to verify that the backend is running correctly.

Example:

```bash
curl http://localhost:8000/health
```

### Query

```http
POST /query
```

The endpoint accepts a user question, retrieves relevant document chunks, generates a grounded response, and returns the answer together with source information.

Example:

```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is backpropagation?\"}"
```

---

# 21. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available locally at:

```text
http://localhost:8000
```

Interactive API documentation is available through FastAPI's generated documentation.

---

# 22. Environment Variables

Configuration values should be stored in environment variables rather than hard-coded in the application.

Example:

```text
BACKEND_URL=http://localhost:8000
OLLAMA_MODEL=llama3.2:3b
```

An `.env.example` file is provided as a template for required environment variables.

Sensitive credentials and local environment files should not be committed to GitHub.

---

# 23. Frontend

The frontend provides a browser-based interface for interacting with the RAG assistant.

It is implemented using:

- HTML
- CSS
- JavaScript

The frontend communicates with the FastAPI backend through HTTP requests.

The backend URL is configurable so that the frontend does not need to depend on a hard-coded production server address.

### Frontend Features

- User question input.
- Chat-style interaction.
- Loading state while waiting for the backend.
- Display of generated answers.
- Display of retrieved source information.
- Friendly error handling.

---

# 24. End-to-End Workflow

The complete application follows this flow:

```text
User
 |
 v
Frontend
 |
 v
POST /query
 |
 v
FastAPI Backend
 |
 v
Semantic Retrieval
 |
 v
ChromaDB
 |
 v
Top Relevant Chunks
 |
 v
Grounded Prompt
 |
 v
Llama 3.2 via Ollama
 |
 v
Answer + Sources
 |
 v
Frontend
```

For the Extended Vision workflow, an uploaded document image is additionally processed by the YOLO and OCR components before the resulting information is incorporated into the retrieval and generation process.

---

# 25. Reproducibility

The project separates the experimental pipeline from the application backend.

The notebook documents the process used to:

- Parse the source documents.
- Create chunks.
- Generate embeddings.
- Build the vector database.
- Evaluate retrieval and generation.
- Export the configuration required by the backend.

The backend uses the resulting persisted vector store instead of rebuilding the complete pipeline for every application startup.

The exact configuration used for the final vector store is documented in the notebook and exported configuration.

---

# 26. Screenshots

Screenshots of the working application can be added here.

Suggested screenshots:

### Application Interface

```text
[Add frontend screenshot here]
```

### API / Backend

```text
[Add FastAPI or application screenshot here]
```

### RAG Response with Sources

```text
[Add example grounded response screenshot here]
```

### Vision Extension

```text
[Add YOLO/OCR image-processing screenshot here]
```

---

# 27. Assignment Requirements Coverage

| Requirement | Implementation |
|---|---|
| Document loading | pypdf |
| Document inspection | Notebook |
| Chunking strategy | RecursiveCharacterTextSplitter |
| Chunk size / overlap | 1000 / 200 |
| Embeddings | all-MiniLM-L6-v2 |
| Persistent vector store | ChromaDB |
| Retrieval | Semantic similarity search |
| Grounded generation | Context-restricted LLM prompt |
| LLM | Llama 3.2 3B through Ollama |
| Evaluation | 10-question evaluation |
| Correctness result | 90% |
| Groundedness result | 90% |
| Failure cases | Included |
| Vision extension | YOLO document layout detection |
| OCR | Tesseract |
| Backend | FastAPI |
| Health endpoint | `/health` |
| Query endpoint | `/query` |
| Testing | Pytest / TestClient |
| Frontend | HTML/CSS/JavaScript |
| Notebook | `notebooks/rag_pipeline.ipynb` |
| Configuration export | JSON configuration |

---

# 28. Limitations and Future Improvements

The current system has several areas that can be improved:

- Improve retrieval for comparison-style questions.
- Add reranking after initial semantic retrieval.
- Introduce hybrid retrieval using semantic and keyword search.
- Add retrieval confidence thresholds.
- Expand the evaluation dataset.
- Automate correctness and groundedness evaluation.
- Improve citation formatting.
- Add better handling of tables and diagrams.
- Improve OCR quality for complex document layouts.
- Add more comprehensive frontend error handling.
- Add deployment configuration for production environments.

---

# 29. Conclusion

This project demonstrates a complete Retrieval-Augmented Generation pipeline for Deep Learning documents.

The system processes raw PDF documents, creates semantic embeddings, stores them in a persistent ChromaDB vector database, retrieves relevant context, and generates grounded answers using a local Llama model.

The project also extends the traditional text-based RAG workflow with document-layout detection and OCR, allowing information extracted from document images to participate in the retrieval and generation process.

The notebook provides the experimental, evaluation, and Computer Vision pipeline, while the FastAPI backend and web frontend provide the application layer for end-to-end interaction.
