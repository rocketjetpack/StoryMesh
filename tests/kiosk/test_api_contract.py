"""HTTP contract tests for the kiosk API.

The single load-bearing invariant: **no API response ever contains the user's
email address**, in any field, in any nesting level. The kiosk is shown to
conference attendees on a public screen — leaking somebody's email through
``/api/jobs`` would be a privacy incident, not a bug.

These tests use ``fastapi.testclient.TestClient`` against ``create_app()``
with a ``JobManager`` whose ``_start_subprocess_locked`` is stubbed so we
never actually spawn pipeline subprocesses.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from storymesh.core.artifacts import ArtifactStore
from storymesh.kiosk.app import create_app
from storymesh.kiosk.jobs import JobManager, JobRecord


SECRET_EMAIL = "very-private-attendee@example-corp.com"


class _StubProcess:
    """Stand-in subprocess that never finishes during the test."""

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:  # noqa: D401
        return


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    store = ArtifactStore(root=tmp_path)
    manager = JobManager(max_concurrent=2, artifact_store=store, poll_interval=0.05)

    def fake_start(self: JobManager, record: JobRecord) -> None:
        record.process = _StubProcess()  # type: ignore[assignment]
        record.status = "running"
        import time
        record.started_at = time.time()

    monkeypatch.setattr(JobManager, "_start_subprocess_locked", fake_start)

    app = create_app(manager=manager)
    with TestClient(app) as test_client:
        yield test_client


def _walk_for_email(blob: Any, needle: str) -> list[str]:
    """Return JSON-paths where ``needle`` appears anywhere inside ``blob``."""
    hits: list[str] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, str):
            if needle in node:
                hits.append(path)
        elif isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and "email" in key.lower():
                    hits.append(f"{path}.{key} (email-shaped key)")
                visit(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                visit(value, f"{path}[{i}]")

    visit(blob, "$")
    return hits


def test_submit_response_does_not_leak_email(client: TestClient) -> None:
    resp = client.post(
        "/api/submit",
        json={"prompt": "A literary story about lighthouses", "email": SECRET_EMAIL, "prompt_style": "default"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "email" not in body
    assert SECRET_EMAIL not in resp.text


def test_jobs_endpoint_does_not_leak_email(client: TestClient) -> None:
    submit = client.post(
        "/api/submit",
        json={"prompt": "A noir thriller in old San Francisco", "email": SECRET_EMAIL, "prompt_style": "default"},
    )
    assert submit.status_code == 200, submit.text

    listing = client.get("/api/jobs")
    assert listing.status_code == 200
    hits = _walk_for_email(listing.json(), SECRET_EMAIL)
    assert hits == [], f"email leaked at: {hits}"
    # Belt-and-braces: also assert it's nowhere in the raw response text, not even
    # in a key name like "email" → "".
    assert SECRET_EMAIL not in listing.text


def test_invalid_email_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/submit",
        json={"prompt": "A nice prompt about anything", "email": "not-a-real-email", "prompt_style": "default"},
    )
    assert resp.status_code == 422  # pydantic validation error


def test_unknown_prompt_style_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/api/submit",
        json={
            "prompt": "A story about something good",
            "email": SECRET_EMAIL,
            "prompt_style": "made_up_style",
        },
    )
    assert resp.status_code == 400


def test_prompt_styles_endpoint_returns_picker_data(client: TestClient) -> None:
    resp = client.get("/api/prompt-styles")
    assert resp.status_code == 200
    styles = resp.json()
    assert len(styles) >= 2
    assert any(s["is_recommended"] for s in styles), "at least one style should be recommended"
    for s in styles:
        assert {"id", "name", "description", "is_recommended"} <= set(s.keys())


def test_gallery_starts_empty(client: TestClient) -> None:
    resp = client.get("/api/gallery")
    assert resp.status_code == 200
    assert resp.json() == []


def test_synopsis_404_for_unknown_run(client: TestClient) -> None:
    resp = client.get("/api/run/does-not-exist/synopsis")
    assert resp.status_code == 404


def test_jobs_payload_field_set(client: TestClient) -> None:
    """Lock the JobStatus contract — no surprise fields, ever."""
    client.post(
        "/api/submit",
        json={"prompt": "A small ordinary story", "email": SECRET_EMAIL, "prompt_style": "default"},
    )
    resp = client.get("/api/jobs")
    body = resp.json()
    assert isinstance(body, list) and body
    allowed = {
        "run_id", "status", "title", "stage", "stage_index",
        "total_stages", "started_at", "queue_position", "prompt_style",
    }
    for entry in body:
        assert set(entry.keys()) == allowed, f"unexpected fields: {set(entry.keys()) - allowed}"


def test_response_text_never_mentions_recipient_anywhere(client: TestClient) -> None:
    """Defense in depth: scrape every API endpoint for the email substring."""
    client.post(
        "/api/submit",
        json={"prompt": "A long enough prompt to validate", "email": SECRET_EMAIL, "prompt_style": "default"},
    )
    for path in ("/api/prompt-styles", "/api/jobs", "/api/gallery"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert SECRET_EMAIL not in resp.text
        # Also check that no response includes the literal word "email" as a key.
        try:
            body = resp.json()
        except json.JSONDecodeError:
            continue
        hits = _walk_for_email(body, SECRET_EMAIL)
        assert hits == [], f"{path} leaked email at: {hits}"
