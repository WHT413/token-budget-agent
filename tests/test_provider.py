"""Provider and request-service behavior tests."""

import pytest

from src.agent.provider import MockProvider, OpenAICompatibleProvider, ProviderError
from src.agent.request_service import process_request
from src.config import Settings


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "LLM_BASE_URL": "http://localhost:9999/v1",
        "LLM_API_KEY": "real-test-key",
        "DEFAULT_MODEL": "economy",
        "STANDARD_MODEL": "standard",
        "FALLBACK_MODEL": "advanced",
        "TOKEN_BUDGET_LIMIT": 8000,
        "COMPRESSION_THRESHOLD": 0.75,
        "MOCK_MODE": False,
        "PROVIDER_TIMEOUT_SECONDS": 1,
        "MAX_PROMPT_LENGTH": 10000,
        "ROUTING_MEDIUM_LENGTH": 500,
        "ROUTING_COMPLEX_LENGTH": 1500,
    }
    values.update(overrides)
    return Settings(**values)


def test_mock_provider_returns_deterministic_response() -> None:
    provider = MockProvider()
    assert provider.generate("Hello", "economy") == provider.generate("Hello", "economy")


class FailingProvider:
    def generate(self, prompt: str, model_id: str) -> str:
        raise ProviderError("provider_failure", "Provider unavailable.")


def test_provider_failure_is_structured() -> None:
    result = process_request("Hello", FailingProvider())
    assert result["status"] == "error"
    assert result["error"]["code"] == "provider_failure"


def test_missing_api_key_is_handled() -> None:
    with pytest.raises(ProviderError, match="LLM_API_KEY"):
        OpenAICompatibleProvider(make_settings(LLM_API_KEY="dummy"))


def test_invalid_provider_output_is_controlled() -> None:
    provider = OpenAICompatibleProvider(make_settings())

    class EmptyCompletions:
        def create(self, **kwargs: object) -> object:
            message = type("Message", (), {"content": None})()
            choice = type("Choice", (), {"message": message})()
            return type("Result", (), {"choices": [choice]})()

    provider.client.chat.completions = EmptyCompletions()  # type: ignore[assignment]
    with pytest.raises(ProviderError) as exc:
        provider.generate("Hello", "economy")
    assert exc.value.code == "invalid_provider_response"
