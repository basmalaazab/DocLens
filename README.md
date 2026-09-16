# RAG-Powered Document Assistant

A Retrieval-Augmented Generation (RAG) system for answering questions from a collection of Deep Learning documents. The system combines document processing, semantic retrieval, vector search, and a local Large Language Model (LLM) to generate grounded answers with source and page references.

The project also includes an Extended Track computer vision component that allows users to upload or capture images of document pages, diagrams, and tables. YOLO-based document layout detection and OCR are used to extract useful information from the image and combine it with the user's question before retrieval.

---

## 1. Project Overview

The system follows a complete RAG pipeline:

```text
PDF Documents
     |
     v
PDF Parsing with PyPDF
     |
     v
Page-level Documents
     |
     v
Recursive Character Chunking
     |
     v
Sentence Transformer Embeddings
     |
     v
ChromaDB Vector Store
     |
     v
Semantic Retrieval
     |
     v
Context + User Question
     |
     v
Grounded Prompt
     |
     v
Llama 3.2 3B via Ollama
     |
     v
Answer + Sources
```

For image-based queries, the extended pipeline is:

```text
Image Upload / Camera
        |
        v
YOLO DocLayNet
        |
        +------------------+
        |                  |
        v                  v
Document Layout       Text Regions
Detection                  |
        |                  v
        |                 OCR
        |                  |
        +--------+---------+
                 |
                 v
       Image Description
                 |
                 v
     Combined User Query
                 |
                 v
       ChromaDB Retrieval
                 |
                 v
      Grounded LLM Answer
```

---

## 2. Domain and Dataset

### Domain

The selected domain is **Deep Learning**.

The document collection contains educational material covering topics such as:

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

### Source Documents

The RAG pipeline was built using three PDF documents:

- `book.pdf`
- `UnderstandingDeepLearning_02_09_26_C.pdf`
- `dive_into_deep_learning.pdf`

The PDFs are parsed page by page using `pypdf`.

For each page, the system stores:

```text
source
page
text
```

Empty pages are ignored before chunking.

---

## 3. Document Processing

PDF documents are loaded using `pypdf`.

Each page is converted into a LangChain `Document` with metadata containing the original filename and page number.

Example metadata:

```python
{
    "source": "book.pdf",
    "page": 98
}
```

This metadata is preserved throughout the pipeline so that retrieved chunks can be traced back to their original document and page.

---

## 4. Chunking Strategy

The project uses LangChain's `RecursiveCharacterTextSplitter`.

Configuration:

| Parameter | Value |
|---|---:|
| Chunk size | 1000 characters |
| Chunk overlap | 200 characters |

The recursive splitter was selected because it attempts to preserve meaningful text boundaries while dividing long document pages into smaller retrieval units.

The overlap helps preserve context between neighboring chunks and reduces the chance of losing information at chunk boundaries.

---

## 5. Embeddings

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The model converts document chunks and user questions into vector representations.

Semantic similarity between the question and document chunks is then used to retrieve the most relevant context.

---

## 6. Vector Database

The vector store is implemented using **ChromaDB**.

Configuration:

```text
Database: ChromaDB
Collection: deep_learning_docs
Persistence: /content/vector_store
```

The vector store is persisted to disk so that the backend can load the existing embeddings instead of rebuilding the entire index for every request.

The final collection contains:

```text
5666 chunks
```

---

## 7. Retrieval

For each user question, the system performs semantic retrieval against the ChromaDB collection.

The retrieval function returns:

```python
{
    "text": "...",
    "source": "document.pdf",
    "page": 123
}
```

The top relevant chunks are then passed to the generation stage.

Example:

```python
retrieved = retrieve_documents(
    "What is backpropagation?",
    n_results=5
)
```

The retrieved source and page information are preserved and returned with the generated answer.

---

## 8. Grounded Generation

The generation component uses a local LLM:

```text
Llama 3.2 3B
```

The model is accessed through **Ollama**.

The prompt explicitly instructs the model to:

1. Use only the retrieved context.
2. Avoid unsupported information.
3. State when the required information is not available.
4. Provide a clear answer.
5. Include relevant source and page references.

This reduces the risk of generating answers that are unrelated to the indexed documents.

The generation flow is:

```text
User Question
      |
      v
Retrieve Relevant Chunks
      |
      v
Build Grounded Prompt
      |
      v
Llama 3.2 3B
      |
      v
Answer + Source References
```

---

# 9. Evaluation

The RAG pipeline was evaluated using **10 predefined questions** covering major Deep Learning topics.

### Evaluation Questions

1. What is backpropagation?
2. What is gradient descent?
3. What is the purpose of activation functions in neural networks?
4. How does a convolutional neural network work?
5. What is the difference between CNNs and RNNs?
6. What is dropout and why is it used?
7. What is the vanishing gradient problem?
8. What is self-attention?
9. What is a Transformer?
10. What is overfitting in deep learning?

### Evaluation Criteria

Each generated answer was manually reviewed according to:

- **Correctness** — whether the answer accurately addresses the question according to the retrieved document context.
- **Groundedness** — whether the answer is supported by the retrieved context without introducing unsupported claims.

### Results

| Metric | Result |
|---|---:|
| Questions evaluated | 10 |
| Reviewed questions | 10/10 |
| Correctness | 90% |
| Groundedness | 90% |

The results indicate that 9 out of the 10 evaluated responses were marked both correct and grounded.

One evaluation case, the CNN vs. RNN comparison, was marked incorrect and not grounded because the retrieved context did not explicitly provide enough information to answer the comparison. This represents a retrieval/context limitation rather than treating unsupported model output as correct.

### Example Evaluation

| Question | Result | Grounded |
|---|---|---|
| What is backpropagation? | Correct | Grounded |
| What is gradient descent? | Correct | Grounded |
| What is the purpose of activation functions? | Correct | Grounded |
| How does a CNN work? | Correct | Grounded |
| What is the difference between CNNs and RNNs? | Incorrect | Not grounded |
| What is dropout and why is it used? | Correct | Grounded |
| What is the vanishing gradient problem? | Correct | Grounded |
| What is self-attention? | Correct | Grounded |
| What is a Transformer? | Correct | Grounded |
| What is overfitting in deep learning? | Correct | Grounded |

---

# 10. Failure Case Analysis

Additional questions outside the document domain were used to examine how the system behaves when the required information is not available.

Example questions include:

```text
What is the capital of France?
What is the boiling point of water at sea level?
How do I repair a car engine?
```

These tests are intended to identify cases where retrieval provides weak or unrelated context.

The main mitigation is the grounding instruction in the generation prompt:

```text
Answer the user's question using ONLY the provided context.
```

The model is also instructed to explicitly state when the information is not available in the provided documents.

This makes unsupported generation a visible failure case that can be identified during evaluation.

---

# 11. Computer Vision Extension

The project implements the Extended Track using:

- YOLO
- DocLayNet
- OCR
- OpenCV
- Tesseract
- Gradio

The selected YOLO model is trained for document layout analysis:

```text
yolo-doclaynet
```

It can identify document elements such as:

- Text
- Title
- Caption
- List-item
- Section-header
- Footnote
- Table
- Picture
- Formula

### Image Processing Pipeline

When an image is provided:

```text
Image
 |
 v
YOLO Document Layout Detection
 |
 +------------------+
 |                  |
 v                  v
Layout Elements    Text Regions
                       |
                       v
                      OCR
                       |
                       v
              Extracted Text
                       |
        +--------------+
        |
        v
Image Description
        |
        v
Question + Image Information
        |
        v
Semantic Retrieval
        |
        v
Grounded Answer
```

---

## 12. OCR

OCR is applied only to regions that YOLO identifies as text-like.

The text labels used for OCR are:

```python
TEXT_LABELS = {
    "Text",
    "Title",
    "Caption",
    "List-item",
    "Section-header",
    "Footnote"
}
```

Table and picture regions are not directly passed through the text OCR pipeline.

This allows the system to focus OCR processing on regions that are expected to contain readable text.

---

## 13. Image Content Classification

The image content type is determined from the detected document layout elements.

Possible classifications include:

```text
table
diagram
text
unknown
```

For example:

- `Table` detected → `table`
- `Picture` detected → `diagram`
- Text-related labels detected → `text`

The classification result is included in the image description used during retrieval.

---

## 14. Image-Based RAG

The image information is combined with the user's question before retrieval.

Example:

```python
combined_query = f"""
{question}

{image_description}
"""
```

The combined query is then passed to the same ChromaDB retrieval pipeline used for normal text questions.

This allows the system to connect visual information with knowledge contained in the indexed Deep Learning documents.

Example use cases:

- Asking what a photographed diagram represents.
- Explaining a Deep Learning architecture.
- Asking about a table shown in a document.
- Asking a question about a photographed page.

---

# 15. Frontend

The project includes a web-based frontend for interacting with the assistant.

The frontend provides:

- Question input
- Answer display
- Source references
- Backend API communication
- Image upload/capture support for the computer vision extension
- Loading and error handling

The image-based demonstration also uses **Gradio** to provide an upload/camera interface.

---

# 16. Backend

The backend is implemented using **FastAPI**.

Main responsibilities:

- Load the persisted vector store.
- Receive user questions.
- Retrieve relevant document chunks.
- Generate grounded answers using Ollama.
- Return answers and source references.
- Support the extended image-based pipeline.

### Main API Endpoints

#### `GET /health`

Checks whether the backend is running.

Example:

```bash
curl http://localhost:8000/health
```

#### `POST /query`

Receives a user question and returns a generated answer with source information.

Example:

```bash
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is backpropagation?\"}"
```

Example request:

```json
{
    "question": "What is backpropagation?"
}
```

The response contains the generated answer and the retrieved sources.

---

# 17. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| RAG Framework | LangChain |
| PDF Processing | PyPDF |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| LLM | Llama 3.2 3B |
| Local LLM Runtime | Ollama |
| Computer Vision | YOLO / DocLayNet |
| OCR | Tesseract / pytesseract |
| Image Processing | OpenCV |
| Image Demo | Gradio |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |

---

# 18. Project Structure

```text
rag_powered_assistant/
│
├── rag_backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       └── query.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── schemas/
│   │   │   └── query.py
│   │   ├── services/
│   │   │   ├── retrieval.py
│   │   │   ├── generation.py
│   │   │   └── vision.py
│   │   ├── utils/
│   │   │   └── logging_config.py
│   │   └── main.py
│   │
│   ├── tests/
│   │   └── test_query.py
│   │
│   ├── requirements.txt
│   └── README.md
│
├── rag_frontend/
│   └── rag_frontend/
│       ├── index.html
│       ├── app.js
│       ├── style.css
│       ├── config.js
│       └── config.js.example
│
├── run.bat
├── .gitignore
└── README.md
```

The persisted vector store and raw PDF corpus are treated as generated/source artifacts and should not be committed when they exceed the repository requirements.

---

# 19. Environment Variables

Backend configuration is managed through environment variables.

Example:

```env
OLLAMA_MODEL=llama3.2:3b
VECTOR_STORE_PATH=./vector_store
COLLECTION_NAME=deep_learning_docs
```

Frontend configuration:

```env
API_BASE_URL=http://localhost:8000
```

Actual variable names should match the configuration files included in the project.

---

# 20. Backend Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd rag_powered_assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install backend dependencies

```bash
cd rag_backend
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama and make sure the Ollama server is running.

Pull the required model:

```bash
ollama pull llama3.2:3b
```

### 5. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation is available through:

```text
http://localhost:8000/docs
```

---

# 21. Frontend Setup

Open the frontend directory:

```bash
cd rag_frontend/rag_frontend
```

Configure the backend URL in the frontend configuration:

```env
API_BASE_URL=http://localhost:8000
```

Then serve the frontend using a local web server.

For example:

```bash
python -m http.server 5500
```

The frontend can then communicate with the FastAPI backend.

---

# 22. Running the Complete System

The complete application requires:

```text
Frontend
   |
   v
FastAPI Backend
   |
   +----> ChromaDB
   |
   +----> Ollama
            |
            v
       Llama 3.2 3B
```

For image queries:

```text
Frontend
   |
   v
Image
   |
   v
Vision Pipeline
   |
   +----> YOLO
   |
   +----> OCR
   |
   v
FastAPI / RAG Pipeline
   |
   v
ChromaDB
   |
   v
Ollama
```

---

# 23. Notebook

The RAG pipeline notebook documents the complete data and retrieval workflow.

It includes:

1. PDF loading and inspection
2. Page-level extraction
3. Chunking
4. Embedding generation
5. ChromaDB persistence
6. Retrieval experiments
7. Prompt construction
8. Ollama generation
9. Ten-question evaluation
10. Failure-case analysis
11. Vector store export
12. Computer vision extension
13. YOLO document layout detection
14. OCR
15. Image-based RAG

The notebook is designed to demonstrate the transition from raw documents to a persisted vector store that can be served by the backend.

---

# 24. Reproducibility

The main RAG configuration is:

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

The vector store is persisted so the backend can reuse the indexed documents without recomputing embeddings during every application startup.

---

# 25. Screenshots

Add screenshots of the running application here before final submission.

Recommended screenshots:

### Main Chat Interface

```text
[Add screenshot here]
```

### Question and Grounded Answer

```text
[Add screenshot here]
```

### Retrieved Sources

```text
[Add screenshot here]
```

### Image-Based Query

```text
[Add screenshot here]
```

### YOLO Detection

```text
[Add screenshot here]
```

---

# 26. Assignment Requirements Coverage

| Requirement | Implementation |
|---|---|
| Document loading | PyPDF |
| Document inspection | PDF/page statistics in notebook |
| Chunking | RecursiveCharacterTextSplitter |
| Chunk size and overlap | 1000 / 200 |
| Embeddings | all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Persistence | Persistent ChromaDB |
| Retrieval testing | 10 questions |
| Grounded prompting | Explicit context-only prompt |
| Evaluation | 10 reviewed questions |
| Correctness evaluation | 90% |
| Groundedness evaluation | 90% |
| Failure cases | Out-of-domain questions |
| Local LLM | Ollama + Llama 3.2 3B |
| Backend | FastAPI |
| Health endpoint | `/health` |
| Query endpoint | `/query` |
| Tests | Pytest |
| Frontend | HTML/CSS/JavaScript |
| Image extension | YOLO + OCR |
| Image upload/camera | Gradio |
| Document layout detection | YOLO DocLayNet |
| Image-to-RAG integration | Combined image description + question |
| Source citations | Document filename + page number |

---

# 27. Limitations and Future Improvements

Current limitations include:

- Retrieval quality depends on the semantic similarity between the question and indexed chunks.
- Some questions may require information that is not present in the retrieved context.
- The evaluation dataset contains 10 manually reviewed questions, so it is useful for validation but not a comprehensive benchmark.
- OCR accuracy can vary depending on image quality, lighting, orientation, and text size.
- YOLO document-layout detection should be tested on several real images before deployment.
- The local Llama model may produce unsupported information if the retrieved context is insufficient despite the grounding instructions.

Potential improvements include:

- Hybrid keyword + semantic retrieval.
- Reranking retrieved chunks.
- Better OCR preprocessing.
- Larger and more diverse evaluation datasets.
- Automated retrieval metrics such as Recall@K.
- More robust citation formatting.
- Additional document types.
- Production-level authentication and deployment configuration.

---

# 28. Conclusion

This project implements a complete RAG-powered Document Assistant for the Deep Learning domain.

The pipeline processes PDF documents, splits them into overlapping chunks, generates semantic embeddings, stores them in ChromaDB, retrieves relevant context, and generates grounded answers using a local Llama 3.2 3B model through Ollama.

The system was evaluated on 10 Deep Learning questions and achieved:

```text
Correctness: 90%
Groundedness: 90%
```

The Extended Track adds computer vision capabilities using YOLO DocLayNet and OCR, allowing image information to be incorporated into the same retrieval and generation pipeline.
