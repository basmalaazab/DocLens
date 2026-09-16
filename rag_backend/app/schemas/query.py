from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    question: str = Field(..., min_length=1, description="The user's question.")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    collection_count: int


class QueryImageResponse(BaseModel):
    """Response for POST /query-image (Extended Track — vision + RAG)."""

    answer: str
    sources: list[str]
    content_type: str
    detected_elements: str
