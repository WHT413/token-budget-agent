"""Pydantic API contracts for AI request submission."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import settings


class AIRequest(BaseModel):
    prompt: str = Field(max_length=settings.max_prompt_length)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be empty or whitespace-only.")
        return value.strip()


class RoutingResponse(BaseModel):
    complexity: Literal["simple", "medium", "complex"]
    model_profile: Literal["economy", "standard", "advanced"]
    display_name: str
    provider_model_id: str
    reason: str
    matched_rules: list[str]
    prompt_length: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class AIResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["success", "error"]
    prompt: str
    response: str | None
    routing: RoutingResponse
    usage: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None
    error: ErrorDetail | None = None
