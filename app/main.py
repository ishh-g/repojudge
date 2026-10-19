"""RepoJudge FastAPI app: routes and static frontend.

Endpoints
    GET  /health                 -> {"status": "ok"}
    POST /analyze                -> full report JSON
    GET  /report/{id}/markdown   -> the same report as Markdown
    GET  /                       -> the single-page frontend
"""
from __future__ import annotations

import logging
from typing import List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, cache, config, report
from .errors import RepoJudgeError

logger = logging.getLogger("repojudge")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RepoJudge",
    version=__version__,
    description="Judge any public GitHub repository fairly, with proof.",
)


class AnalyzeRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    requirements: Union[List[str], str, None] = Field(
        default=None,
        description="Optional extra things to check (one per line or a list).",
    )
    nocache: bool = Field(
        default=False,
        description="Ignore every cached result and run the analysis again.",
    )


def _normalise_requirements(value: Union[List[str], str, None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


@app.exception_handler(RepoJudgeError)
def _repojudge_error_handler(request: Request, exc: RepoJudgeError) -> JSONResponse:
    """Return clean JSON for every expected failure."""
    logger.info("Handled error %s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "model": config.GEMMA_MODEL,
        "gemma_configured": bool(config.GEMINI_API_KEY),
    }


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> JSONResponse:
    """Judge a public GitHub repository and return the full report."""
    result = report.analyze_repo(
        payload.repo_url,
        _normalise_requirements(payload.requirements),
        force_refresh=bool(payload.nocache),
    )
    return JSONResponse(status_code=200, content=result)


@app.get("/report/{report_id}/markdown", response_class=PlainTextResponse)
def report_markdown(report_id: str):
    """Return a previously generated report as Markdown."""
    cached = cache.get(report_id)
    if not cached or "score" not in cached:
        raise RepoJudgeError(
            "That report is not in the cache any more. Run the analysis again.",
            code="report_not_found",
            status_code=404,
        )
    markdown = report.to_markdown(cached)
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="repojudge-{report_id}.md"'},
    )


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(config.STATIC_DIR / "index.html")


# Static assets (style.css, app.js).
if config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
