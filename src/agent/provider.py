"""Provider abstraction for real and deterministic mock LLM calls."""

from typing import Protocol

from openai import APITimeoutError, OpenAI

from ..config import Settings, settings


class ProviderError(Exception):
    """A controlled provider failure safe to expose through the API."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LLMProvider(Protocol):
    """Minimal provider contract used by request processing."""

    def generate(self, prompt: str, model_id: str) -> str: ...


class MockProvider:
    """Deterministic provider used for local development and demonstrations."""

    def generate(self, prompt: str, model_id: str) -> str:
        return f"Mock response from {model_id}: I received your request: {prompt}"


class OpenAICompatibleProvider:
    """Adapter for the configured OpenAI-compatible endpoint."""

    def __init__(self, config: Settings = settings) -> None:
        api_key = config.llm_api_key.get_secret_value()
        if not api_key or api_key.casefold() in {"dummy", "changeme", "your-api-key"}:
            raise ProviderError("missing_api_key", "LLM_API_KEY is required when MOCK_MODE is false.")
        self.client = OpenAI(
            base_url=str(config.llm_base_url).rstrip("/"),
            api_key=api_key,
            timeout=config.provider_timeout_seconds,
        )

    def generate(self, prompt: str, model_id: str) -> str:
        try:
            result = self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            content = result.choices[0].message.content
        except APITimeoutError as exc:
            raise ProviderError("provider_timeout", "The AI provider timed out.") from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("provider_failure", "The AI provider request failed.") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("invalid_provider_response", "The AI provider returned no usable text.")
        return content


def build_provider(config: Settings = settings) -> LLMProvider:
    """Select mock or real provider from validated configuration."""
    return MockProvider() if config.mock_mode else OpenAICompatibleProvider(config)
