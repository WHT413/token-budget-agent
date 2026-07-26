"""Application configuration loaded from environment and validated by Pydantic."""

from functools import lru_cache

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings.

    Environment variables are the single source of truth. This class only
    declares required fields, types, and validation constraints.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    llm_base_url: HttpUrl = Field(alias="LLM_BASE_URL")
    llm_api_key: SecretStr = Field(alias="LLM_API_KEY")
    default_model: str = Field(alias="DEFAULT_MODEL", min_length=1)
    standard_model: str = Field(alias="STANDARD_MODEL", min_length=1)
    fallback_model: str = Field(alias="FALLBACK_MODEL", min_length=1)
    token_budget_limit: int = Field(alias="TOKEN_BUDGET_LIMIT", gt=0)
    compression_threshold: float = Field(alias="COMPRESSION_THRESHOLD", ge=0.0, le=1.0)
    mock_mode: bool = Field(alias="MOCK_MODE")
    provider_timeout_seconds: float = Field(alias="PROVIDER_TIMEOUT_SECONDS", gt=0)
    max_prompt_length: int = Field(alias="MAX_PROMPT_LENGTH", ge=100)
    routing_medium_length: int = Field(alias="ROUTING_MEDIUM_LENGTH", ge=50)
    routing_complex_length: int = Field(alias="ROUTING_COMPLEX_LENGTH", ge=100)
    qdrant_url: str = Field(alias="QDRANT_URL", min_length=1)
    qdrant_collection: str = Field(alias="QDRANT_COLLECTION", min_length=1)
    embedding_model: str = Field(alias="EMBEDDING_MODEL", min_length=1)
    similarity_threshold: float = Field(alias="SIMILARITY_THRESHOLD", ge=0.0, le=1.0)
    min_samples_to_trust: int = Field(alias="MIN_SAMPLES_TO_TRUST", ge=0)

    @property
    def model_map(self) -> dict[str, str]:
        """Map logical model sizes to configured model names."""
        return {"small": self.default_model, "large": self.fallback_model}

    @property
    def routing_models(self) -> dict[str, dict[str, str]]:
        """Return the centrally configured model profiles used by routing."""
        return {
            "simple": {
                "profile": "economy",
                "display_name": "Economy Model",
                "provider_model_id": self.default_model,
            },
            "medium": {
                "profile": "standard",
                "display_name": "Standard Model",
                "provider_model_id": self.standard_model,
            },
            "complex": {
                "profile": "advanced",
                "display_name": "Advanced Model",
                "provider_model_id": self.fallback_model,
            },
        }


@lru_cache
def get_settings() -> Settings:
    """Return cached validated application settings."""
    return Settings()


settings = get_settings()

LLM_BASE_URL = str(settings.llm_base_url).rstrip("/")
LLM_API_KEY = settings.llm_api_key.get_secret_value()
DEFAULT_MODEL = settings.default_model
FALLBACK_MODEL = settings.fallback_model
TOKEN_BUDGET_LIMIT = settings.token_budget_limit
COMPRESSION_THRESHOLD = settings.compression_threshold
MODEL_MAP = settings.model_map
ROUTING_MODELS = settings.routing_models
QDRANT_URL = settings.qdrant_url
QDRANT_COLLECTION = settings.qdrant_collection
EMBEDDING_MODEL = settings.embedding_model
SIMILARITY_THRESHOLD = settings.similarity_threshold
MIN_SAMPLES_TO_TRUST = settings.min_samples_to_trust
