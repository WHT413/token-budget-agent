"""FastAPI app exposing the agent over HTTP."""

from fastapi import FastAPI

app = FastAPI(title="token-budget-agent")


@app.get("/health")
def health():
    """Placeholder health check endpoint."""
    return {"status": "ok"}
