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
    fallback_model: str = Field(alias="FALLBACK_MODEL", min_length=1)
    token_budget_limit: int = Field(alias="TOKEN_BUDGET_LIMIT", gt=0)
    compression_threshold: float = Field(alias="COMPRESSION_THRESHOLD", ge=0.0, le=1.0)

    @property
    def model_map(self) -> dict[str, str]:
        """Map logical model sizes to configured model names."""
        return {"small": self.default_model, "large": self.fallback_model}


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
