"""``storymesh kiosk`` Typer subcommand: start / stop / status the kiosk server.

The kiosk is a uvicorn-hosted FastAPI process. ``start`` launches uvicorn
(detached by default), records its PID to ``~/.storymesh/kiosk.pid``, and
prints the URL. ``stop`` reads the pidfile and sends SIGTERM. ``status``
reports whether the process is alive.

Foreground mode (``--foreground``) does not detach — useful for systemd
services or when you want logs streaming to your terminal.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, NamedTuple

import httpx
import typer
from rich.console import Console

from storymesh.config import get_kiosk_config

kiosk_app = typer.Typer(help="Run the conference-booth kiosk web frontend.")
_console = Console()


def _pid_file() -> Path:
    """Default pidfile path; STORYMESH_KIOSK_PID overrides for tests/CI."""
    override = os.environ.get("STORYMESH_KIOSK_PID")
    if override:
        return Path(override)
    return Path.home() / ".storymesh" / "kiosk.pid"


def _frontend_dist_dir() -> Path:
    """Resolve <repo>/frontend/dist relative to this source file."""
    return Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"


class _RunInfo(NamedTuple):
    pid: int
    host: str
    port: int


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pidfile() -> _RunInfo | None:
    """Return ``(pid, host, port)`` if the kiosk is running, else None.

    The pidfile is JSON ``{"pid": int, "host": str, "port": int}``. Falls back
    to legacy bare-int format for compatibility with older runs.
    """
    path = _pid_file()
    if not path.exists():
        return None
    raw = path.read_text().strip()
    try:
        data = json.loads(raw)
        info = _RunInfo(int(data["pid"]), str(data.get("host", "127.0.0.1")), int(data.get("port", 8000)))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Legacy: pidfile contained just the integer PID.
        try:
            info = _RunInfo(int(raw), "127.0.0.1", 8000)
        except ValueError:
            return None
    return info if _is_running(info.pid) else None


def _write_pidfile(pid: int, host: str, port: int) -> None:
    path = _pid_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "host": host, "port": port}))


@kiosk_app.command("start")
def kiosk_start(
    host: Annotated[str | None, typer.Option("--host", help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port.")] = None,
    foreground: Annotated[
        bool,
        typer.Option("--foreground", "-f", help="Run uvicorn in the foreground (don't detach)."),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable uvicorn auto-reload (development only)."),
    ] = False,
) -> None:
    """Start the kiosk web server."""
    cfg = get_kiosk_config()
    bind_host = host or str(cfg.get("bind_host", "127.0.0.1"))
    bind_port = port or int(cfg.get("bind_port", 8000))

    existing = _read_pidfile()
    if existing is not None:
        _console.print(f"[yellow]Kiosk is already running[/yellow] (PID {existing.pid}).")
        _console.print(f"URL: [bold]http://{existing.host}:{existing.port}[/bold]")
        raise typer.Exit(code=0)

    dist = _frontend_dist_dir()
    if not dist.exists():
        _console.print(
            "[yellow]Warning:[/yellow] frontend bundle not found at "
            f"[dim]{dist}[/dim]. The API will work but the UI will be unavailable."
        )
        _console.print("Build it with: [bold]cd frontend && npm install && npm run build[/bold]")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "storymesh.kiosk.app:app",
        "--host",
        bind_host,
        "--port",
        str(bind_port),
        "--log-level",
        "info",
    ]
    if reload:
        cmd.append("--reload")

    if foreground:
        _console.print(f"[bold]Starting kiosk[/bold] at http://{bind_host}:{bind_port} (foreground)")
        os.execvp(cmd[0], cmd)  # noqa: S606  # never returns
        return

    log_path = Path.home() / ".storymesh" / "kiosk.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=log_handle,
        stderr=log_handle,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    _write_pidfile(proc.pid, bind_host, bind_port)

    # Brief wait + healthz probe so we fail loudly when uvicorn can't bind.
    healthy = _wait_for_health(bind_host, bind_port, timeout=4.0)
    if not healthy:
        _console.print(
            f"[red]Kiosk did not become healthy within 4s.[/red] "
            f"Check log: [dim]{log_path}[/dim]"
        )
        raise typer.Exit(code=1)

    _console.print(f"[green]Kiosk started[/green] (PID {proc.pid}) at [bold]http://{bind_host}:{bind_port}[/bold]")
    _console.print(f"Log: [dim]{log_path}[/dim]")
    _console.print("Stop with: [bold]storymesh kiosk stop[/bold]")


@kiosk_app.command("stop")
def kiosk_stop() -> None:
    """Stop the running kiosk web server."""
    pidfile = _pid_file()
    info = _read_pidfile()
    if info is None:
        if pidfile.exists():
            pidfile.unlink(missing_ok=True)
            _console.print("[yellow]Stale pidfile cleaned up; no running kiosk found.[/yellow]")
        else:
            _console.print("[yellow]No kiosk running.[/yellow]")
        return

    try:
        os.kill(info.pid, signal.SIGTERM)
    except OSError as exc:
        _console.print(f"[red]Failed to signal PID {info.pid}:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Wait up to 5s for it to exit; escalate to SIGKILL if needed.
    for _ in range(50):
        if not _is_running(info.pid):
            break
        time.sleep(0.1)
    else:
        _console.print(f"[yellow]Kiosk did not exit; sending SIGKILL to PID {info.pid}[/yellow]")
        try:
            os.kill(info.pid, signal.SIGKILL)
        except OSError:
            pass

    pidfile.unlink(missing_ok=True)
    _console.print(f"[green]Kiosk stopped[/green] (PID {info.pid}).")


@kiosk_app.command("status")
def kiosk_status() -> None:
    """Report whether the kiosk is running."""
    info = _read_pidfile()
    if info is None:
        _console.print("[dim]Kiosk is not running.[/dim]")
        raise typer.Exit(code=1)

    _console.print(f"[green]Kiosk running[/green] (PID {info.pid}) at [bold]http://{info.host}:{info.port}[/bold]")
    if _wait_for_health(info.host, info.port, timeout=1.0):
        _console.print("Health: [green]ok[/green]")
    else:
        _console.print("Health: [yellow]unresponsive[/yellow] (process is alive but /healthz did not respond)")


def _wait_for_health(host: str, port: int, *, timeout: float) -> bool:
    """Poll /healthz until it responds 200 or the timeout elapses."""
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=0.5)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.15)
    return False
