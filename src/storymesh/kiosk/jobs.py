"""Background pipeline orchestrator for the kiosk.

The JobManager is the single source of truth for pipeline runs in the kiosk:

- Holds at most ``max_concurrent`` subprocess pipelines at any time; further
  submissions queue FIFO.
- Spawns each pipeline as a fresh ``python -m storymesh.cli generate ...``
  subprocess. Crashes (segfaults in WeasyPrint/Pillow, OOM, etc.) cannot
  take down the kiosk server.
- Polls each active run's artifact directory once per second to advance
  ``current_stage`` and pluck the title out of ``proposal_draft_output.json``
  as soon as it's written.
- Publishes state-change events via :class:`EventBus` for the SSE endpoint.

CRITICAL: the email recipient is held only inside :class:`JobRecord` and
forwarded to the subprocess via the ``STORYMESH_EMAIL_RECIPIENT`` env var.
It is never returned by any HTTP endpoint and never written to argv (where
``ps`` would expose it).
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import orjson

from storymesh.core.artifacts import ArtifactStore
from storymesh.core.stage_progress import STAGE_NAMES, infer_stage_statuses
from storymesh.kiosk.events import EventBus
from storymesh.kiosk.models import GalleryItem, JobStatus, JobStatusName

logger = logging.getLogger(__name__)


JobLifecycle = Literal["queued", "running", "completed", "failed"]


@dataclass
class JobRecord:
    """In-memory record of one pipeline run."""

    run_id: str
    prompt: str
    email: str
    prompt_style: str
    status: JobLifecycle = "queued"
    title: str | None = None
    current_stage: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)


class JobManager:
    """Concurrency-capped pipeline orchestrator with artifact-dir polling."""

    def __init__(
        self,
        *,
        max_concurrent: int = 5,
        artifact_store: ArtifactStore | None = None,
        event_bus: EventBus | None = None,
        poll_interval: float = 1.0,
        python_executable: str | None = None,
        seed_from_recent: int = 0,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.python_executable = python_executable or sys.executable
        self.seed_from_recent = seed_from_recent
        self._store = artifact_store or ArtifactStore()
        self._bus = event_bus or EventBus()

        self._records: dict[str, JobRecord] = {}
        # Submission order is preserved for the queued portion.
        self._queue: deque[str] = deque()
        # Completed runs (gallery). Populated from disk at start() and from
        # finishing runs during the session.
        self._gallery: list[GalleryItem] = []

        self._lock = asyncio.Lock()
        self._poller_task: asyncio.Task[None] | None = None

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    async def start(self) -> None:
        """Begin the background poller. Idempotent.

        Also seeds the gallery with the most-recent successful runs on disk
        when ``seed_from_recent`` is positive.
        """
        if self.seed_from_recent > 0 and not self._gallery:
            self._seed_gallery_from_disk(self.seed_from_recent)
        if self._poller_task is None or self._poller_task.done():
            self._poller_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel the poller and terminate any active subprocesses."""
        if self._poller_task is not None:
            self._poller_task.cancel()
            try:
                await self._poller_task
            except asyncio.CancelledError:
                pass
            self._poller_task = None
        async with self._lock:
            for record in self._records.values():
                if record.process is not None and record.process.poll() is None:
                    record.process.terminate()

    async def submit(self, *, prompt: str, email: str, prompt_style: str) -> tuple[str, int]:
        """Register a new run. Returns ``(run_id, queue_position)``.

        ``queue_position`` is 0 when the subprocess starts immediately, or the
        1-indexed position behind currently-running and previously-queued runs.
        """
        run_id = uuid.uuid4().hex
        record = JobRecord(
            run_id=run_id,
            prompt=prompt,
            email=email,
            prompt_style=prompt_style,
        )
        async with self._lock:
            self._records[run_id] = record
            active_count = self._count_active_locked()
            if active_count < self.max_concurrent:
                self._start_subprocess_locked(record)
                queue_position = 0
            else:
                self._queue.append(run_id)
                queue_position = self._queue.index(run_id) + 1
        self._publish_status_event(record, queue_position=queue_position)
        return run_id, queue_position

    def list_jobs(self) -> list[JobStatus]:
        """Snapshot of every non-completed and recently-completed run."""
        statuses: list[JobStatus] = []
        for record in self._records.values():
            statuses.append(self._record_to_status(record))
        statuses.sort(
            key=lambda s: (
                {"running": 0, "queued": 1, "completed": 2, "failed": 3}.get(s.status, 4),
                s.started_at or 0.0,
            )
        )
        return statuses

    def gallery(self) -> list[GalleryItem]:
        """Newest cover-art entries first (by completion time)."""
        return sorted(self._gallery, key=lambda item: item.completed_at, reverse=True)

    def cover_path(self, run_id: str) -> Path | None:
        """Return the on-disk path to a completed run's cover art, or None."""
        candidate = self._store.runs_dir / run_id / "cover_art.png"
        return candidate if candidate.exists() else None

    def run_synopsis(self, run_id: str) -> tuple[str, str] | None:
        """Return ``(title, back_cover_summary)`` for a completed run, or None.

        Reads ``story_writer_output.json`` for the synopsis and falls back to
        ``proposal_draft_output.json`` for the title when the assembler step
        hasn't written its own.
        """
        run_dir = self._store.runs_dir / run_id
        story_path = run_dir / "story_writer_output.json"
        if not story_path.exists():
            return None
        try:
            story = orjson.loads(story_path.read_bytes())
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(story, dict):
            return None
        synopsis = story.get("back_cover_summary")
        if not isinstance(synopsis, str) or not synopsis.strip():
            return None
        title = _try_read_assembler_title(run_dir) or _try_read_title(run_dir) or "Untitled"
        return title, synopsis.strip()

    # ── Internals ─────────────────────────────────────────────────────────

    def _count_active_locked(self) -> int:
        return sum(1 for r in self._records.values() if r.status == "running")

    def _seed_gallery_from_disk(self, count: int) -> None:
        """Pre-populate the gallery from the most recent successful runs on disk.

        A run qualifies when it has both ``cover_art.png`` and
        ``book_assembler_output.json``. Ordered by cover-file mtime so the
        newest successful book appears first.
        """
        runs_dir = self._store.runs_dir
        if not runs_dir.exists():
            return
        candidates: list[tuple[float, Path]] = []
        for entry in runs_dir.iterdir():
            if not entry.is_dir():
                continue
            cover = entry / "cover_art.png"
            assembler = entry / "book_assembler_output.json"
            if not (cover.exists() and assembler.exists()):
                continue
            try:
                mtime = cover.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, entry))
        candidates.sort(key=lambda c: c[0], reverse=True)
        for mtime, entry in candidates[:count]:
            title = _try_read_assembler_title(entry) or _try_read_title(entry) or "Untitled"
            self._gallery.append(
                GalleryItem(
                    run_id=entry.name,
                    title=title,
                    cover_url=f"/api/cover/{entry.name}",
                    completed_at=mtime,
                )
            )

    def _start_subprocess_locked(self, record: JobRecord) -> None:
        """Spawn the pipeline subprocess for ``record``. Caller holds the lock."""
        env = os.environ.copy()
        # Email travels via env var so it never appears in argv (visible to `ps`).
        # The CLI's `generate` command falls back to STORYMESH_EMAIL_RECIPIENT when
        # --email is omitted (see cli.py).
        env["STORYMESH_EMAIL_RECIPIENT"] = record.email
        env["STORYMESH_KIOSK_RUN"] = "1"

        cmd = [
            self.python_executable,
            "-m",
            "storymesh.cli",
            "generate",
            record.prompt,
            "--quality",
            "standard",
            "--run-id",
            record.run_id,
        ]
        if record.prompt_style and record.prompt_style != "default":
            cmd.extend(["--prompt-style", record.prompt_style])

        record.process = subprocess.Popen(  # noqa: S603
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        record.status = "running"
        record.started_at = time.time()

    def _record_to_status(self, record: JobRecord) -> JobStatus:
        queue_position: int | None = None
        if record.status == "queued":
            try:
                queue_position = self._queue.index(record.run_id) + 1
            except ValueError:
                queue_position = None
        stage = record.current_stage
        try:
            stage_index = STAGE_NAMES.index(stage) + 1 if stage else 0
        except ValueError:
            stage_index = 0
        return JobStatus(
            run_id=record.run_id,
            status=_status_to_public(record.status),
            title=record.title,
            stage=stage,
            stage_index=stage_index,
            total_stages=len(STAGE_NAMES),
            started_at=record.started_at,
            queue_position=queue_position,
            prompt_style=record.prompt_style,
        )

    def _publish_status_event(self, record: JobRecord, *, queue_position: int | None = None) -> None:
        status = self._record_to_status(record)
        if queue_position is not None and status.status == "queued":
            status = status.model_copy(update={"queue_position": queue_position})
        self._bus.publish({"type": "job_status", "job": status.model_dump()})

    async def _poll_loop(self) -> None:
        """Periodically inspect each run dir to advance status."""
        while True:
            try:
                await self._poll_once()
            except Exception:
                logger.exception("kiosk poller iteration failed; continuing")
            await asyncio.sleep(self.poll_interval)

    async def _poll_once(self) -> None:
        async with self._lock:
            running_ids = [rid for rid, r in self._records.items() if r.status == "running"]
        for run_id in running_ids:
            await self._refresh_running(run_id)
        await self._drain_queue_if_capacity()

    async def _refresh_running(self, run_id: str) -> None:
        record = self._records.get(run_id)
        if record is None or record.status != "running":
            return
        run_dir = self._store.runs_dir / run_id

        # Title becomes available as soon as proposal_draft_output.json lands.
        if record.title is None:
            title = _try_read_title(run_dir)
            if title:
                record.title = title
                self._publish_status_event(record)

        # Stage progress.
        _statuses, active_stage = infer_stage_statuses(run_dir if run_dir.exists() else None)
        if active_stage and active_stage != record.current_stage:
            record.current_stage = active_stage
            self._publish_status_event(record)

        # Subprocess outcome.
        proc = record.process
        if proc is not None and proc.poll() is not None:
            await self._finalize(record)

    async def _finalize(self, record: JobRecord) -> None:
        run_dir = self._store.runs_dir / record.run_id
        assembler_done = (run_dir / "book_assembler_output.json").exists()
        proc = record.process
        exit_code = proc.poll() if proc is not None else None
        success = exit_code == 0 and assembler_done

        async with self._lock:
            record.completed_at = time.time()
            record.status = "completed" if success else "failed"
            # Last-resort title: read from book_assembler_output.json if proposal-time read missed it.
            if record.title is None:
                record.title = _try_read_title(run_dir) or _try_read_assembler_title(run_dir) or "Untitled"
            # Email cleared from memory the moment we no longer need it.
            record.email = ""
            if success:
                self._gallery.append(
                    GalleryItem(
                        run_id=record.run_id,
                        title=record.title or "Untitled",
                        cover_url=f"/api/cover/{record.run_id}",
                        completed_at=record.completed_at,
                    )
                )
        self._publish_status_event(record)
        self._bus.publish(
            {
                "type": "job_completed" if success else "job_failed",
                "run_id": record.run_id,
                "title": record.title,
            }
        )

    async def _drain_queue_if_capacity(self) -> None:
        async with self._lock:
            while self._queue and self._count_active_locked() < self.max_concurrent:
                next_id = self._queue.popleft()
                record = self._records.get(next_id)
                if record is None or record.status != "queued":
                    continue
                self._start_subprocess_locked(record)
                self._publish_status_event(record, queue_position=0)


def _status_to_public(status: JobLifecycle) -> JobStatusName:
    return status  # types align by construction; explicit for the contract.


def _try_read_title(run_dir: Path) -> str | None:
    """Pull ``proposal.title`` out of proposal_draft_output.json if present."""
    path = run_dir / "proposal_draft_output.json"
    if not path.exists():
        return None
    try:
        raw = orjson.loads(path.read_bytes())
    except Exception:  # noqa: BLE001
        return None
    proposal = raw.get("proposal") if isinstance(raw, dict) else None
    if isinstance(proposal, dict):
        title = proposal.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None


def _try_read_assembler_title(run_dir: Path) -> str | None:
    """Fallback: pull title out of book_assembler_output.json."""
    path = run_dir / "book_assembler_output.json"
    if not path.exists():
        return None
    try:
        raw = orjson.loads(path.read_bytes())
    except Exception:  # noqa: BLE001
        return None
    if isinstance(raw, dict):
        title = raw.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None
