"""Server-Sent Events fan-out for the kiosk frontend.

A tiny in-process pub/sub: each connected client gets its own asyncio queue;
publishers call :meth:`EventBus.publish` to broadcast a dict to every
subscriber. The frontend uses ``EventSource`` to subscribe to
``/api/events`` and update its UI without polling.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any


class EventBus:
    """In-memory async fan-out for SSE."""

    def __init__(self, queue_size: int = 64) -> None:
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._queue_size = queue_size

    def publish(self, event: dict[str, Any]) -> None:
        """Send ``event`` to every connected subscriber.

        Slow consumers whose queues are full silently drop events rather than
        backing up the publisher — this is a best-effort UI hint, not a
        guaranteed transport.
        """
        for queue in list(self._queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """Yield events forever for one subscriber."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_size)
        self._queues.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._queues.discard(queue)
