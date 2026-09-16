from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from .env or environment variables."""

    vector_store_dir: str = "./data/vector_store"
    pdf_dir: str = "./data/pdfs"
    collection_name: str = "deep_learning_docs"

    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"

    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    ingest_batch_size: int = 500

    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8501,http://127.0.0.1:8501,http://localhost:7860,http://127.0.0.1:7860"

    yolo_model_path: str = "./models/yolov8n-doclaynet.pt"
    yolo_model_filename: str = "yolov8n-doclaynet.pt"
    yolo_model_repo: str = "hantian/yolo-doclaynet"
    yolo_confidence: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


settings = Settings()
