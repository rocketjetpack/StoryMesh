"""Pydantic request/response models for the kiosk HTTP API.

CRITICAL invariant: no model in this file contains a field named ``email`` or
that otherwise carries a recipient address. The user's email address is held
exclusively inside :class:`storymesh.kiosk.jobs.JobRecord` (server-side memory)
and is forwarded to the pipeline subprocess via env var. The contract test in
``tests/kiosk/test_api_contract.py`` enforces this.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# RFC 5322-lite. Sufficient for kiosk format-only validation; the SMTP server
# is the real authority on deliverability.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


JobStatusName = Literal["queued", "running", "completed", "failed"]


class SubmitRequest(BaseModel):
    """Body of ``POST /api/submit``."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=4, max_length=2000)
    email: str = Field(min_length=3, max_length=320)
    prompt_style: str = Field(default="default", max_length=64)

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        if not _EMAIL_RE.match(value):
            raise ValueError("email does not look like a valid address")
        return value


class SubmitResponse(BaseModel):
    """Reply to ``POST /api/submit``. Deliberately omits any echo of the email."""

    run_id: str
    queue_position: int = Field(
        description="0 means running immediately. >0 means queued behind that many others.",
    )


class PromptStyleOption(BaseModel):
    """One entry in the prompt style picker."""

    id: str
    name: str
    description: str
    is_recommended: bool = False


class JobStatus(BaseModel):
    """Public, email-free description of one pipeline run."""

    run_id: str
    status: JobStatusName
    title: str | None = None
    stage: str | None = None
    stage_index: int = 0
    total_stages: int
    started_at: float | None = None  # epoch seconds; None while queued
    queue_position: int | None = None  # set only while queued
    prompt_style: str


class GalleryItem(BaseModel):
    """One completed run shown in the cover-art gallery."""

    run_id: str
    title: str
    cover_url: str  # relative URL the frontend can <img src=...> directly
    completed_at: float  # epoch seconds


class RunSynopsis(BaseModel):
    """Lazy detail payload for a single completed run."""

    run_id: str
    title: str
    synopsis: str
