"""Behavioural tests for :class:`storymesh.kiosk.jobs.JobManager`.

We avoid spawning real subprocesses by replacing
``JobManager._start_subprocess_locked`` with a stub that records the call and
attaches a fake "process" object whose ``poll()`` returns whatever the test
chooses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from storymesh.core.artifacts import ArtifactStore
from storymesh.kiosk.jobs import JobManager, JobRecord


class FakeProcess:
    """Stand-in for ``subprocess.Popen`` with a controllable exit state."""

    def __init__(self) -> None:
        self._exit_code: int | None = None

    def poll(self) -> int | None:
        return self._exit_code

    def terminate(self) -> None:
        if self._exit_code is None:
            self._exit_code = -15

    def finish(self, exit_code: int = 0) -> None:
        self._exit_code = exit_code


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(root=tmp_path)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch, store: ArtifactStore) -> Iterator[JobManager]:
    mgr = JobManager(max_concurrent=2, artifact_store=store, poll_interval=0.05)
    started: list[str] = []

    def fake_start(self: JobManager, record: JobRecord) -> None:
        record.process = FakeProcess()  # type: ignore[assignment]
        record.status = "running"
        import time
        record.started_at = time.time()
        started.append(record.run_id)

    monkeypatch.setattr(JobManager, "_start_subprocess_locked", fake_start)
    mgr._started_log = started  # type: ignore[attr-defined]  # test introspection
    yield mgr


@pytest.mark.anyio
async def test_concurrency_cap_holds(manager: JobManager) -> None:
    """A 3rd submission while two are running must be queued, not spawned."""
    rid1, pos1 = await manager.submit(prompt="A first prompt", email="a@b.co", prompt_style="default")
    rid2, pos2 = await manager.submit(prompt="A second prompt", email="c@d.co", prompt_style="default")
    rid3, pos3 = await manager.submit(prompt="A third prompt", email="e@f.co", prompt_style="default")

    assert pos1 == 0
    assert pos2 == 0
    assert pos3 == 1, "third submission should be queued at position 1"

    statuses = {s.run_id: s for s in manager.list_jobs()}
    assert statuses[rid1].status == "running"
    assert statuses[rid2].status == "running"
    assert statuses[rid3].status == "queued"
    assert statuses[rid3].queue_position == 1


@pytest.mark.anyio
async def test_queue_drains_when_active_finishes(manager: JobManager, store: ArtifactStore) -> None:
    """Finishing one running job should promote the head of the queue."""
    rid1, _ = await manager.submit(prompt="First long-form prompt", email="a@b.co", prompt_style="default")
    _rid2, _ = await manager.submit(prompt="Second long-form prompt", email="c@d.co", prompt_style="default")
    rid3, _ = await manager.submit(prompt="Third long-form prompt", email="e@f.co", prompt_style="default")

    # Simulate rid1 finishing successfully: write the assembler output and exit 0.
    run_dir = store.runs_dir / rid1
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "book_assembler_output.json").write_bytes(b'{"title":"A Test Book"}')
    rec = manager._records[rid1]
    assert isinstance(rec.process, FakeProcess)
    rec.process.finish(0)

    # One poll cycle should finalise rid1 and promote rid3 to running.
    await manager._poll_once()

    statuses = {s.run_id: s for s in manager.list_jobs()}
    assert statuses[rid1].status == "completed"
    assert statuses[rid3].status == "running"

    gallery = manager.gallery()
    assert len(gallery) == 1
    assert gallery[0].run_id == rid1


@pytest.mark.anyio
async def test_failed_run_does_not_enter_gallery(manager: JobManager, store: ArtifactStore) -> None:
    """Exit code != 0 (or missing assembler output) should mark failed and skip the gallery."""
    rid1, _ = await manager.submit(prompt="A prompt that crashes", email="a@b.co", prompt_style="default")
    rec = manager._records[rid1]
    assert isinstance(rec.process, FakeProcess)
    rec.process.finish(1)
    await manager._poll_once()

    statuses = {s.run_id: s for s in manager.list_jobs()}
    assert statuses[rid1].status == "failed"
    assert manager.gallery() == []


@pytest.mark.anyio
async def test_email_cleared_after_completion(manager: JobManager, store: ArtifactStore) -> None:
    """The email is wiped from the in-memory record once delivery is the subprocess's job."""
    rid, _ = await manager.submit(
        prompt="A nice prompt for fiction",
        email="secret@user.example",
        prompt_style="default",
    )
    rec = manager._records[rid]
    assert rec.email == "secret@user.example"

    run_dir = store.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "book_assembler_output.json").write_bytes(b'{"title":"Done"}')
    assert isinstance(rec.process, FakeProcess)
    rec.process.finish(0)
    await manager._poll_once()

    assert rec.email == "", "email must be wiped from memory once the subprocess no longer needs it"


@pytest.mark.anyio
async def test_title_extracted_from_proposal_draft(manager: JobManager, store: ArtifactStore) -> None:
    """The poller should pluck the title out of proposal_draft_output.json mid-run."""
    rid, _ = await manager.submit(prompt="A literary prompt about gardens", email="g@h.co", prompt_style="default")
    run_dir = store.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "proposal_draft_output.json").write_bytes(
        b'{"proposal":{"title":"The Glass Conservatory"}}'
    )
    await manager._poll_once()
    statuses = {s.run_id: s for s in manager.list_jobs()}
    assert statuses[rid].title == "The Glass Conservatory"


def test_prompt_style_passed_to_subprocess(monkeypatch: pytest.MonkeyPatch, store: ArtifactStore) -> None:
    """When a non-default prompt style is given, --prompt-style must appear in the subprocess argv."""
    captured: dict[str, Any] = {}

    def fake_popen(cmd: list[str], **kwargs: Any) -> FakeProcess:  # noqa: ANN401
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return FakeProcess()

    import storymesh.kiosk.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod.subprocess, "Popen", fake_popen)

    mgr = JobManager(max_concurrent=1, artifact_store=store, poll_interval=0.05)
    asyncio.run(mgr.submit(prompt="A surprising premise", email="x@y.co", prompt_style="verbalized_sampling"))

    assert "--prompt-style" in captured["cmd"]
    assert "verbalized_sampling" in captured["cmd"]
    # Email is in env, not argv.
    assert not any("x@y.co" in arg for arg in captured["cmd"]), "email must not appear in argv"
    assert captured["env"]["STORYMESH_EMAIL_RECIPIENT"] == "x@y.co"


@pytest.mark.anyio
async def test_gallery_seeded_from_disk_on_start(tmp_path: Path) -> None:
    """start() should pre-populate the gallery with the most recent successful runs."""
    store = ArtifactStore(root=tmp_path)
    # Seed three "completed" runs on disk with different mtimes.
    for idx, run_id in enumerate(("oldest", "middle", "newest")):
        run_dir = store.runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "cover_art.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (run_dir / "book_assembler_output.json").write_bytes(
            f'{{"title":"Book {run_id}"}}'.encode()
        )
        # Touch with monotonically-increasing mtime so order is deterministic.
        import os
        ts = 1_700_000_000 + idx * 100
        os.utime(run_dir / "cover_art.png", (ts, ts))

    # A run missing the cover should be skipped.
    half = store.runs_dir / "incomplete"
    half.mkdir(parents=True)
    (half / "book_assembler_output.json").write_bytes(b'{"title":"No Cover"}')

    mgr = JobManager(max_concurrent=1, artifact_store=store, seed_from_recent=2)
    await mgr.start()
    try:
        gallery = mgr.gallery()
        # Two newest only; sorted newest-first.
        assert [item.title for item in gallery] == ["Book newest", "Book middle"]
        # Cover URLs use the standard endpoint pattern.
        assert gallery[0].cover_url == "/api/cover/newest"
    finally:
        await mgr.stop()


def test_run_synopsis_reads_story_writer_output(store: ArtifactStore) -> None:
    """Synopsis lookup should pull back_cover_summary off disk for completed runs."""
    from storymesh.kiosk.jobs import JobManager
    mgr = JobManager(max_concurrent=1, artifact_store=store)
    run_dir = store.runs_dir / "abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "story_writer_output.json").write_bytes(
        b'{"back_cover_summary": "A short story about a long winter."}'
    )
    (run_dir / "book_assembler_output.json").write_bytes(b'{"title":"The Long Winter"}')
    result = mgr.run_synopsis("abc123")
    assert result == ("The Long Winter", "A short story about a long winter.")


def test_run_synopsis_returns_none_when_missing(store: ArtifactStore) -> None:
    from storymesh.kiosk.jobs import JobManager
    mgr = JobManager(max_concurrent=1, artifact_store=store)
    assert mgr.run_synopsis("nonexistent") is None


def test_default_prompt_style_omits_flag(monkeypatch: pytest.MonkeyPatch, store: ArtifactStore) -> None:
    """The default style needs no flag — relies on the configured pipeline default."""
    captured: dict[str, Any] = {}

    def fake_popen(cmd: list[str], **kwargs: Any) -> FakeProcess:  # noqa: ANN401
        captured["cmd"] = cmd
        return FakeProcess()

    import storymesh.kiosk.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod.subprocess, "Popen", fake_popen)

    mgr = JobManager(max_concurrent=1, artifact_store=store, poll_interval=0.05)
    asyncio.run(mgr.submit(prompt="A small premise about anything", email="x@y.co", prompt_style="default"))

    assert "--prompt-style" not in captured["cmd"]
