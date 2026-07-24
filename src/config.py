"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL_MAP = {
    "default": os.getenv("DEFAULT_MODEL", "claude-haiku-4-5-20251001"),
    "fallback": os.getenv("FALLBACK_MODEL", "claude-sonnet-4-6"),
}

BUDGET_LIMIT = int(os.getenv("TOKEN_BUDGET_LIMIT", "8000"))
COMPRESSION_THRESHOLD = float(os.getenv("COMPRESSION_THRESHOLD", "0.75"))
