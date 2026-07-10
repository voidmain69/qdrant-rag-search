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


class LLMProvider(StrEnum):
    OLLAMA = "ollama"  # local, via OLLAMA_URL
    OPENAI = "openai"  # OpenAI or any OpenAI-compatible endpoint (OPENAI_BASE_URL)


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
    # re-upsert of a product whose searchable content is unchanged skips the (expensive)
    # embedding + enrichment and only refreshes the payload; set true to always re-embed
    # (e.g. after changing the enrichment prompt/model)
    reembed_unchanged: bool = False

    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 50

    # --- LLM backend (shared by query understanding and ingest enrichment) ---
    # ollama: local model at OLLAMA_URL; openai: OpenAI / any OpenAI-compatible API.
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    ollama_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- LLM query understanding (strict mode) ---
    query_llm_enabled: bool = False
    # ollama default gemma4:e4b; for openai set e.g. gpt-4o-mini
    query_llm_model: str = "gemma4:e4b"
    # measured 6-23s per call on gemma4:e4b / RTX 3080 Ti under WSL2 (huge variance from
    # GPU paravirtualization); results are LRU-cached and search degrades to token
    # fallback past the budget — a cloud provider is far faster, tune down accordingly
    query_llm_timeout_s: float = Field(default=30.0, gt=0)

    # --- LLM product enrichment at ingest (pre-embedding) ---
    ingest_llm_enabled: bool = False
    ingest_llm_model: str = "gemma4:e4b"
    ingest_llm_timeout_s: float = Field(default=60.0, gt=0)
    ingest_llm_concurrency: int = Field(default=2, ge=1, le=8)
    # enrich products that ALREADY have supplier attributes too — their gain is the
    # uk/ru/en synonyms (the biggest cross-lingual recall lever; measured to beat BGE-M3),
    # while supplier attributes still win on merge. False = only attribute-less products
    # (cheaper: one fewer LLM call per attribute-rich product).
    ingest_enrich_with_attributes: bool = True

    prefetch_limit: int = 50
    schema_version: int = 1

    # Path to the SQLite file that persists background job (import / async batch) state so
    # jobs survive a restart and a job left running by a died process is reaped on boot.
    # Empty = ephemeral in-memory only (jobs vanish on restart — the pre-durability default).
    # In docker this points at a mounted volume (see docker-compose.yml).
    jobs_db_path: str = ""

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def llm_enabled(self) -> bool:
        return self.query_llm_enabled or self.ingest_llm_enabled

    @model_validator(mode="after")
    def _guard_openai(self) -> "Settings":
        if self.llm_enabled and self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_PROVIDER=openai and an LLM feature "
                "(QUERY_LLM_ENABLED / INGEST_LLM_ENABLED) is enabled."
            )
        return self

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
