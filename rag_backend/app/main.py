from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import init_vector_store
from app.services.vision import init_vision_model
from app.utils.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_vector_store()
    init_vision_model()
    yield


app = FastAPI(
    title="RAG Document Assistant API",
    description=(
        "FastAPI backend for a Retrieval-Augmented Generation pipeline "
        "over PDF documents, with optional image/OCR support."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(query_router)