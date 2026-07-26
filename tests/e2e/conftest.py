"""End-to-end test fixtures for Docker-backed Qdrant."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import QDRANT_COLLECTION, QDRANT_URL


def _qdrant_reachable() -> bool:
    """Return whether Qdrant can answer a lightweight collections request."""
    try:
        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception:
        return False
    return True


def _wait_for_qdrant(timeout_seconds: float = 10.0) -> None:
    """Wait for Qdrant to become reachable, failing the test session on timeout."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            QdrantClient(url=QDRANT_URL).get_collections()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)

    pytest.fail(f"Qdrant was not reachable within {timeout_seconds}s: {last_error}")


@pytest.fixture(scope="session", autouse=True)
def qdrant_docker() -> None:
    """Start docker compose if Qdrant is unavailable and leave it running."""
    if not _qdrant_reachable():
        subprocess.run(["docker", "compose", "up", "-d", "qdrant"], check=True)

    _wait_for_qdrant(timeout_seconds=10.0)


@pytest.fixture(scope="session", autouse=True)
def initialized_qdrant(qdrant_docker: None) -> None:
    """Run the project initializer once so the collection exists."""
    subprocess.run([sys.executable, "scripts/init_qdrant.py"], check=True)


@pytest.fixture()
def qdrant_client(initialized_qdrant: None) -> QdrantClient:
    """Return a real Qdrant client for e2e tests."""
    return QdrantClient(url=QDRANT_URL)


def recreate_collection(client: QdrantClient) -> None:
    """Delete and recreate the configured collection with the embedder vector size."""
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


@pytest.fixture()
def empty_qdrant(qdrant_client: QdrantClient) -> None:
    """Provide an empty collection for cold-start and isolated pipeline tests."""
    recreate_collection(qdrant_client)