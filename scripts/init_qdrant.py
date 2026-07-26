"""Initialize the Qdrant collection used by semantic routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich import print

from src.config import QDRANT_COLLECTION, QDRANT_URL


def main() -> int:
    """Create the configured Qdrant collection if it does not exist."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        if client.collection_exists(QDRANT_COLLECTION):
            print(f"[Qdrant] ALREADY EXISTS: {QDRANT_COLLECTION}")
            return 0

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"[Qdrant] CREATED: {QDRANT_COLLECTION}")
        return 0
    except Exception as exc:
        print(f"[Qdrant] Connection failure: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())