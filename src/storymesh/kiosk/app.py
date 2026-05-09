"""FastAPI app for the StoryMesh kiosk frontend.

Run locally with::

    uvicorn storymesh.kiosk.app:app --reload --port 8000

In production the built React bundle is served from ``frontend/dist`` at the
site root; in development the Vite dev server proxies ``/api/*`` here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from storymesh.config import get_kiosk_config
from storymesh.kiosk.jobs import JobManager
from storymesh.kiosk.models import (
    GalleryItem,
    JobStatus,
    PromptStyleOption,
    RunSynopsis,
    SubmitRequest,
    SubmitResponse,
)
from storymesh.kiosk.prompt_styles import load_prompt_style_options, valid_prompt_style_ids

logger = logging.getLogger(__name__)


def _frontend_dist() -> Path | None:
    """Return ``frontend/dist`` path when a built bundle exists, else None."""
    here = Path(__file__).resolve()
    # The kiosk package lives at <repo>/src/storymesh/kiosk; the frontend at <repo>/frontend.
    candidate = here.parent.parent.parent.parent / "frontend" / "dist"
    return candidate if candidate.exists() else None


def create_app(*, manager: JobManager | None = None) -> FastAPI:
    """Construct the FastAPI app.

    Accepts an optional pre-built :class:`JobManager` to make integration
    testing trivial; tests can inject a manager backed by a temp artifact
    store and a stubbed subprocess.
    """
    cfg = get_kiosk_config()
    max_concurrent = int(cfg.get("max_concurrent_runs", 5))
    seed_from_recent = int(cfg.get("seed_from_recent_runs", 4))

    job_manager = manager or JobManager(
        max_concurrent=max_concurrent,
        seed_from_recent=seed_from_recent,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await job_manager.start()
        try:
            yield
        finally:
            await job_manager.stop()

    app = FastAPI(
        title="StoryMesh Kiosk",
        version="0.1.0",
        lifespan=lifespan,
        default_response_class=JSONResponse,
    )

    # Permit the Vite dev server to talk to us during development. In
    # production the bundle is served from this same origin so CORS is moot.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/prompt-styles", response_model=list[PromptStyleOption])
    async def prompt_styles() -> list[PromptStyleOption]:
        return load_prompt_style_options()

    @app.post("/api/submit", response_model=SubmitResponse)
    async def submit(req: SubmitRequest) -> SubmitResponse:
        if req.prompt_style not in valid_prompt_style_ids():
            raise HTTPException(status_code=400, detail="unknown prompt_style")
        run_id, queue_position = await job_manager.submit(
            prompt=req.prompt,
            email=req.email,
            prompt_style=req.prompt_style,
        )
        return SubmitResponse(run_id=run_id, queue_position=queue_position)

    @app.get("/api/jobs", response_model=list[JobStatus])
    async def jobs() -> list[JobStatus]:
        return job_manager.list_jobs()

    @app.get("/api/gallery", response_model=list[GalleryItem])
    async def gallery() -> list[GalleryItem]:
        return job_manager.gallery()

    @app.get("/api/cover/{run_id}")
    async def cover(run_id: str) -> Response:
        path = job_manager.cover_path(run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="cover not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/run/{run_id}/synopsis", response_model=RunSynopsis)
    async def run_synopsis(run_id: str) -> RunSynopsis:
        result = job_manager.run_synopsis(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="synopsis not available")
        title, synopsis = result
        return RunSynopsis(run_id=run_id, title=title, synopsis=synopsis)

    @app.get("/api/events")
    async def events(request: Request) -> EventSourceResponse:
        bus = job_manager.event_bus

        async def stream() -> AsyncIterator[dict[str, Any]]:
            async for event in bus.subscribe():
                if await request.is_disconnected():
                    break
                yield {"event": event.get("type", "message"), "data": orjson.dumps(event).decode()}

        return EventSourceResponse(stream())

    # Mount the built frontend last, so /api/* routes win.
    dist = _frontend_dist()
    if dist is not None:
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        logger.info("frontend/dist not present; run `cd frontend && npm run build` to serve UI in-process")

    return app


# Module-level instance for ``uvicorn storymesh.kiosk.app:app``.
app = create_app()
