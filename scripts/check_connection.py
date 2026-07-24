"""Check connectivity to configured OpenAI-compatible LLM endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from src.config import settings


def build_client() -> OpenAI:
    """Build an OpenAI-compatible client from validated settings."""
    return OpenAI(
        base_url=str(settings.llm_base_url).rstrip("/"),
        api_key=settings.llm_api_key.get_secret_value(),
    )


def check_model(client: OpenAI, model: str) -> bool:
    """Send a minimal chat completion request and print the result."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        content = response.choices[0].message.content
        print(f"PASS {model}: {content}")
        return True
    except Exception as exc:  # noqa: BLE001 - report exact connection/API failure
        print(f"FAIL {model}: {exc}")
        return False


def main() -> int:
    """Run connection checks for all required models."""
    client = build_client()

    results = [check_model(client, model) for model in settings.model_map.values()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())