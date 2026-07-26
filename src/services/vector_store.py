"""Qdrant vector-store helpers for prompt classifications."""

from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct

from src.config import QDRANT_COLLECTION, QDRANT_URL

client = QdrantClient(url=QDRANT_URL)


def _collection_exists() -> bool:
    try:
        return client.collection_exists(QDRANT_COLLECTION)
    except (UnexpectedResponse, ConnectionError):
        return False


def search_similar(embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search similar prompt classifications, returning normalized payload data."""
    if not _collection_exists():
        return []

    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedding,
        limit=top_k,
        with_payload=True,
    )

    normalized = []
    for result in results:
        payload = result.payload or {}
        normalized.append(
            {
                "level": payload.get("level"),
                "score": result.score,
                "model_used": payload.get("model_used"),
                "prompt": payload.get("prompt"),
            }
        )
    return normalized


def upsert_classification(
    prompt: str,
    embedding: list[float],
    level: str,
    model_used: str,
    token_count: int,
) -> None:
    """Persist one prompt classification in Qdrant."""
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "prompt": prompt,
                    "level": level,
                    "model_used": model_used,
                    "token_count": token_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )


def get_collection_count() -> int:
    """Return point count for the classification collection, or 0 if absent."""
    if not _collection_exists():
        return 0
    return client.count(collection_name=QDRANT_COLLECTION, exact=True).count