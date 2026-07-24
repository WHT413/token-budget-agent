"""FastAPI application stub."""

from fastapi import FastAPI

app = FastAPI(title="token-budget-agent")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok"}
