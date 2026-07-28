"""API validation, routing, error, and end-to-end tests."""

from fastapi.testclient import TestClient

from src.agent import request_service
from src.api import routes

client = TestClient(routes.app)


class FakeProvider:
    """Deterministic stand-in for a real LLM, used to control the classifier's answer."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, model_id: str) -> str:
        return self.response


def _force_llm_classifier(monkeypatch, level: str) -> None:
    """Bypass Qdrant recall and real providers so the classifier deterministically picks `level`."""
    monkeypatch.setattr(request_service, "build_provider", lambda: FakeProvider(level))
    monkeypatch.setattr("src.services.vector_store.get_collection_count", lambda: 0)


def test_valid_request_runs_end_to_end(monkeypatch) -> None:
    _force_llm_classifier(monkeypatch, "simple")
    response = client.post("/api/v1/ai/requests", json={"prompt": "Translate hello."})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["response"]
    assert body["routing"]["model_profile"] == "economy"
    assert body["usage"] is None
    assert body["cost"] is None


def test_medium_request_uses_standard_profile(monkeypatch) -> None:
    _force_llm_classifier(monkeypatch, "medium")
    body = client.post(
        "/api/v1/ai/requests", json={"prompt": "Compare REST and GraphQL."}
    ).json()
    assert body["routing"]["model_profile"] == "standard"


def test_complex_request_uses_advanced_profile(monkeypatch) -> None:
    _force_llm_classifier(monkeypatch, "complex")
    body = client.post(
        "/api/v1/ai/requests",
        json={"prompt": "Design a secure and scalable RAG architecture with RBAC."},
    ).json()
    assert body["routing"]["model_profile"] == "advanced"


def test_provider_error_returns_controlled_response(monkeypatch) -> None:
    def fail(prompt: str) -> dict[str, object]:
        return {
            "status": "error",
            "prompt": prompt,
            "response": None,
            "routing": {
                "complexity": "simple",
                "model_profile": "economy",
                "display_name": "Economy Model",
                "provider_model_id": "test-economy",
                "reason": "Safe default.",
                "matched_rules": ["safe_default"],
                "prompt_length": len(prompt),
            },
            "usage": None,
            "cost": None,
            "error": {"code": "provider_failure", "message": "Provider unavailable."},
        }

    monkeypatch.setattr(routes, "process_request", fail)
    response = client.post("/api/v1/ai/requests", json={"prompt": "Hello"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_failure"


def test_empty_prompt_is_rejected() -> None:
    assert client.post("/api/v1/ai/requests", json={"prompt": ""}).status_code == 422


def test_whitespace_prompt_is_rejected() -> None:
    assert client.post("/api/v1/ai/requests", json={"prompt": "   "}).status_code == 422


def test_prompt_over_maximum_is_rejected() -> None:
    response = client.post("/api/v1/ai/requests", json={"prompt": "x" * 10001})
    assert response.status_code == 422


def test_frontend_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Routing" in response.text
