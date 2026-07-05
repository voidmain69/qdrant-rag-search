from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "qdrant-product-search"
    debug: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "products"
    meta_collection_name: str = "service_meta"

    # comma-separated; empty string disables auth (dev only)
    api_keys: str = ""

    dense_model: str = "intfloat/multilingual-e5-large"
    dense_dim: int = 1024
    sparse_model: str = "Qdrant/bm25"
    sparse_language: str = "russian"
    embed_batch_size: int = 64
    upsert_batch_size: int = 256

    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 50

    prefetch_limit: int = 50
    schema_version: int = 1

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
