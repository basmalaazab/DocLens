# RAG Frontend

A clean HTML/CSS/JS frontend for the FastAPI RAG backend.

## Features

- **Text Question** → `POST /query` — retrieves relevant document chunks, generates a grounded answer
- **Image + Question** → `POST /query-image` — adds YOLO document-layout detection + OCR before answering
- Image drag-and-drop + preview
- Answer display with copy-to-clipboard
- Source citations
- Vision content-type + detected elements panel
- Live backend health indicator (polls `/health` every 3 s)

## Setup

### 1. Configure the backend URL

```bash
# Copy the example and edit the file to point to your backend
cp config.js.example config.js
```

Open `config.js` and set:
```js
window.API_BASE = "http://127.0.0.1:8000";  // change if backend runs elsewhere
```

`config.js` is listed in `.gitignore` and will never be committed.

### 2. Start the backend

See `../backend_share/README.md` for full instructions. Quick start:

```bash
cd ../backend_share
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Serve the frontend

Open a terminal in this folder and run a simple HTTP server:

```bash
python -m http.server 5500
```

Then open **http://localhost:5500** in your browser.

> **Note:** Do NOT open `index.html` directly with `file://` — browser security blocks `fetch()` calls from file:// origins.

## Architecture

```
Browser (localhost:5500)
  │
  ├─ POST /query          → retrieve → prompt → Ollama → grounded answer
  └─ POST /query-image    → YOLO → OCR → retrieve → prompt → Ollama → answer
                                        ↑
                               FastAPI (localhost:8000)
```
