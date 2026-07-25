"""FastAPI application and AI request routes."""

from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import FileResponse, JSONResponse

from ..agent.request_service import process_request

from .schemas import AIRequest, AIResponse

app = FastAPI(title="Agent Cost Minimum", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the lightweight prompt submission UI."""
    return FileResponse(Path(__file__).parents[1] / "fe" / "index.html")


@app.post(
    "/api/v1/ai/requests",
    response_model=AIResponse,
    responses={502: {"model": AIResponse}},
)
def submit_ai_request(request: AIRequest) -> AIResponse | JSONResponse:
    """Validate, route, and execute an AI request."""
    result = process_request(request.prompt)
    if result["status"] == "error":
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=result,
        )
    return AIResponse.model_validate(result)
