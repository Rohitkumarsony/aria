from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    serper_api_key: str = ""

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "aria_docs"
    sessions_dir: str = "./sessions"

    chunk_size: int = 600
    chunk_overlap: int = 100
    retrieval_top_k: int = 6
    bm25_top_k: int = 6
    hybrid_final_k: int = 4


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()