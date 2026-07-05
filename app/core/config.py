from enum import StrEnum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    app_name: str = "qdrant-product-search"
    debug: bool = False
    environment: Environment = Environment.DEVELOPMENT
    # default: human-readable text in development, JSON everywhere else (see _apply_env_defaults)
    log_format: LogFormat | None = None

    # --- observability ---
    metrics_enabled: bool = True
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)

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

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @model_validator(mode="after")
    def _apply_env_defaults(self) -> "Settings":
        if self.log_format is None:
            self.log_format = (
                LogFormat.TEXT if self.environment == Environment.DEVELOPMENT else LogFormat.JSON
            )
        return self

    @model_validator(mode="after")
    def _guard_production(self) -> "Settings":
        if self.is_production and not self.api_key_list:
            raise ValueError(
                "API_KEYS must not be empty when ENVIRONMENT=production — "
                "running an unauthenticated ingestion/search API in production is unsafe."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
