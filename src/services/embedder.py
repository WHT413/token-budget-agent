"""SentenceTransformer embedding service."""

from rich import print
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)
print(f"[Embedder] Model loaded: {EMBEDDING_MODEL}")


def embed(text: str) -> list[float]:
    """Return a dense embedding for text."""
    return model.encode(text).tolist()