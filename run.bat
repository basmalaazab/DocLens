@echo off
setlocal

REM ============================================================
REM  RAG Document Assistant — one-click launcher
REM
REM  Expected layout (place this file next to these two folders):
REM    project-root\
REM      backend\      (FastAPI app, with its own .venv)
REM      frontend\     (index.html, app.js, config.js, style.css)
REM      run.bat       <-- this file
REM
REM  What it does:
REM    1. Makes sure Ollama is running (starts it if not).
REM    2. Starts the FastAPI backend on http://127.0.0.1:8000
REM    3. Serves the frontend as static files on http://localhost:5500
REM    4. Opens the frontend in your default browser
REM ============================================================

set BACKEND_DIR=%~dp0backend
set FRONTEND_DIR=%~dp0frontend
set BACKEND_PORT=8000
set FRONTEND_PORT=5500

echo [1/4] Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if errorlevel 1 (
    echo       Ollama not running — starting it...
    start "Ollama" cmd /k "ollama serve"
    timeout /t 3 /nobreak >NUL
) else (
    echo       Ollama is already running.
)

echo [2/4] Starting backend (FastAPI) on port %BACKEND_PORT%...
if exist "%BACKEND_DIR%\.venv\Scripts\activate.bat" (
    start "RAG Backend" cmd /k "cd /d "%BACKEND_DIR%" && call .venv\Scripts\activate.bat && uvicorn app.main:app --reload --port %BACKEND_PORT%"
) else (
    echo       WARNING: no .venv found in "%BACKEND_DIR%" — using system Python.
    start "RAG Backend" cmd /k "cd /d "%BACKEND_DIR%" && uvicorn app.main:app --reload --port %BACKEND_PORT%"
)

echo [3/4] Waiting for backend to warm up...
timeout /t 5 /nobreak >NUL

echo [4/4] Starting frontend static server on port %FRONTEND_PORT%...
start "RAG Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && python -m http.server %FRONTEND_PORT%"

timeout /t 2 /nobreak >NUL
start "" "http://localhost:%FRONTEND_PORT%/index.html"

echo.
echo All set. Two windows are running (backend + frontend). Close them to stop the app.
endlocal
