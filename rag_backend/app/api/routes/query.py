import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.query import HealthResponse, QueryImageResponse, QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.retrieval import get_chunk_count
from app.services.vision import query_with_image

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", collection_count=get_chunk_count())


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        answer, retrieved_docs = generate_answer(request.question)
    except Exception:
        logger.exception("Failed to answer question: %s", request.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer")

    sources = [f"{doc['source']} - Page {doc['page']}" for doc in retrieved_docs]
    return QueryResponse(answer=answer, sources=sources)


@router.post("/query-image", response_model=QueryImageResponse)
async def query_image(
    question: str = Form(...),
    image: UploadFile = File(...),
) -> QueryImageResponse:
    tmp_path: str | None = None
    suffix = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name
        result = query_with_image(question, tmp_path)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to answer image question: %s", question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer for this image")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    sources = [f"{doc['source']} - Page {doc['page']}" for doc in result["sources"]]
    return QueryImageResponse(
        answer=result["answer"],
        sources=sources,
        content_type=result["content_type"],
        detected_elements=result["detected_elements"],
    )
