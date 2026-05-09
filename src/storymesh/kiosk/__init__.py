"""Kiosk frontend backend for conference-booth StoryMesh demos.

A lightweight FastAPI server that orchestrates pipeline runs as subprocesses,
exposes a JSON/SSE API for a React frontend, and *never* leaks the user's
email address through any response. See ``app.py`` for the entrypoint.
"""

from storymesh.kiosk.app import app, create_app

__all__ = ["app", "create_app"]
