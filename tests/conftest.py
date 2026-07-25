"""Test environment configuration loaded before application imports."""

import os

TEST_ENV = {
    "LLM_BASE_URL": "http://localhost:9999/v1",
    "LLM_API_KEY": "dummy",
    "DEFAULT_MODEL": "test-economy",
    "STANDARD_MODEL": "test-standard",
    "FALLBACK_MODEL": "test-advanced",
    "TOKEN_BUDGET_LIMIT": "8000",
    "COMPRESSION_THRESHOLD": "0.75",
    "MOCK_MODE": "true",
    "PROVIDER_TIMEOUT_SECONDS": "1",
    "MAX_PROMPT_LENGTH": "10000",
    "ROUTING_MEDIUM_LENGTH": "500",
    "ROUTING_COMPLEX_LENGTH": "1500",
}

for key, value in TEST_ENV.items():
    os.environ.setdefault(key, value)
