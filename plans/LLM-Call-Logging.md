# StoryMesh Implementation Plan — Structured LLM Call Logging

**Date:** 2026-04-02
**Version:** current → next minor
**Scope:** Structured on-disk LLM call logging with per-run JSONL artifacts and LangSmith tracing consistency

This document is the authoritative implementation plan for replacing ad-hoc Python `logger` debug output with a structured, two-layer LLM observability system. It is intended to be consumed by Claude Code or any developer working on the repository. Each work item includes the rationale, the files affected, the exact changes required, testing expectations, and ordering constraints.

---

## Table of Contents

1. [Design Overview](#1-design-overview)
2. [Work Item Ordering and Dependencies](#2-work-item-ordering-and-dependencies)
3. [WI-1: `LLMCallLogger` and `ArtifactStore.log_llm_call()`](#3-wi-1-llmcalllogger-and-artifactstorelog_llm_call)
4. [WI-2: `current_run_id` ContextVar and `agent_name` on `LLMClient`](#4-wi-2-current_run_id-contextvar-and-agent_name-on-llmclient)
5. [WI-3: Instrument `complete_json()` to Write Call Records](#5-wi-3-instrument-complete_json-to-write-call-records)
6. [WI-4: Thread `agent_name` and `on_call` Through `graph.py`](#6-wi-4-thread-agent_name-and-on_call-through-graphpy)
7. [WI-5: Set `current_run_id` in Node Wrappers](#7-wi-5-set-current_run_id-in-node-wrappers)
8. [WI-6: Retire Pipeline-Hardening-Plan WI-1](#8-wi-6-retire-pipeline-hardening-plan-wi-1)
9. [Design Decision Record](#9-design-decision-record)
10. [Files Affected Summary](#10-files-affected-summary)
11. [Validation Checklist](#11-validation-checklist)

---

## 1. Design Overview

### The Problem

`LLMClient.complete_json()` currently emits no structured record of LLM interactions. When a parse failure or Pydantic validation error occurs, there is no way to see what prompt was sent, what raw response was returned, which agent triggered the call, or which pipeline run it belonged to — without inserting print statements and re-running. For a multi-agent pipeline that makes several LLM calls per run, this is unacceptable.

The existing `Pipeline-Hardening-Plan.md` (WI-1) proposes patching this with `logger.debug(...)` calls. That approach is insufficient: Python log output is ephemeral, unstructured, mixed with all other log output, and requires a developer to configure `logging.DEBUG` before the fact. It cannot be used for post-mortem inspection of a past run.

### The Solution: Two Complementary Layers

**Layer 1 — On-disk JSONL (durable, offline, structured).**
Every LLM call appends one JSON record to `~/.storymesh/runs/<run_id>/llm_calls.jsonl`. This file is written incrementally so that a mid-run crash still leaves a readable partial record. Each record captures the agent name, model, system prompt, user prompt, raw response, latency, parse outcome, and run ID. This is the primary debugging artifact.

**Layer 2 — LangSmith tracing (interactive, UI-backed).**
LangSmith is already partially wired in via `@_traceable` in `llm/base.py`. This plan ensures `agent_name` is surfaced as span metadata so that individual LLM call spans in the LangSmith dashboard are labeled by agent, not just by provider method name. This layer is optional (gated on `LANGCHAIN_TRACING_V2=true`) and is not the fallback for offline debugging.

### JSONL Record Schema

Each line in `llm_calls.jsonl` is a JSON object with the following fields:

```json
{
  "ts": "2026-04-02T14:32:01.123Z",
  "run_id": "abc123def456",
  "agent": "theme_extractor",
  "model": "claude-sonnet-4-6",
  "temperature": 0.6,
  "attempt": 1,
  "system_prompt": "...",
  "user_prompt": "...",
  "raw_response": "...",
  "parse_success": true,
  "latency_ms": 842
}
```

`parse_success` is `false` if `orjson.JSONDecodeError` or Pydantic validation raises during `complete_json()`. `latency_ms` is wall-clock time from just before `self.complete()` is called to just after it returns (i.e., network + model time only, not retry overhead).

---

## 2. Work Item Ordering and Dependencies

```text
WI-1 (ArtifactStore.log_llm_call + LLMCallLogger)
  │
  └─ WI-2 (ContextVar + agent_name on LLMClient.__init__)
       │
       └─ WI-3 (instrument complete_json())
            │
            └─ WI-4 (thread agent_name + on_call through graph.py)
                 │
                 └─ WI-5 (set current_run_id in node wrappers)
                      │
                      └─ WI-6 (retire Pipeline-Hardening-Plan WI-1)
```

**Recommended execution order:** WI-1 → WI-2 → WI-3 → WI-4 → WI-5 → WI-6

Each step is independently testable before the next begins. WI-3 can be tested with `on_call=None` (no-op). WI-4 wires in the real `ArtifactStore`-backed writer. WI-5 ensures `run_id` is available at call time.

---

## 3. WI-1: `LLMCallLogger` and `ArtifactStore.log_llm_call()`

### Rationale

Before instrumenting `complete_json()`, the write target must exist. `ArtifactStore` already manages the per-run directory structure and is the right owner for `llm_calls.jsonl`.

### Design

Add a single `log_llm_call()` method to `ArtifactStore` that opens the JSONL file in append mode and writes one line. JSONL (newline-delimited JSON) is chosen over a single JSON array because partial runs produce readable files even if the pipeline crashes before closing.

Introduce a `LLMCallLogger` protocol (a simple callable type alias) so that `LLMClient` can accept the writer without importing `ArtifactStore` directly, avoiding a circular dependency between `storymesh.llm` and `storymesh.core`.

### Files Affected

| File | Change |
|---|---|
| `src/storymesh/core/artifacts.py` | Add `log_llm_call()` method |
| `src/storymesh/llm/base.py` | Add `LLMCallLogger` type alias |

### Changes to `src/storymesh/core/artifacts.py`

Add to the `ArtifactStore` class:

```python
def log_llm_call(self, run_id: str, record: dict[str, Any]) -> None:
    """Append one LLM call record to the run's llm_calls.jsonl file.

    Opens in append mode so partial runs produce readable files on crash.
    Creates the run directory if it does not already exist.

    Args:
        run_id: Unique run identifier (matches the run directory name).
        record: Dict conforming to the LLM call record schema.
    """
    run_dir = self.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    line = orjson.dumps(record) + b"\n"
    with open(run_dir / "llm_calls.jsonl", "ab") as f:
        f.write(line)
```

### Changes to `src/storymesh/llm/base.py`

Add near the top of the file, after imports:

```python
from collections.abc import Callable
from typing import Any

# A callable that accepts (run_id, record) and writes it somewhere.
# Using a protocol/alias avoids importing ArtifactStore here (circular dep risk).
LLMCallLogger = Callable[[str, dict[str, Any]], None]
```

Note: `Callable` is already imported in `base.py` for `_traceable`. Consolidate the import rather than duplicating it.

### Testing

Add to `tests/test_artifacts.py`:

- Test that `log_llm_call()` creates `llm_calls.jsonl` in the correct run directory.
- Test that calling `log_llm_call()` twice appends two lines, each valid JSON.
- Test that each line round-trips through `orjson.loads()` without error.
- Test that calling with a non-existent run directory creates it (no pre-existing `run_metadata.json` required).

---

## 4. WI-2: `current_run_id` ContextVar and `agent_name` on `LLMClient`

### Rationale

`complete_json()` needs two pieces of context to write a useful record: the agent name (stable, known at construction time) and the run ID (dynamic, generated per pipeline invocation). These arrive via different mechanisms because of their different lifetimes.

**`agent_name` on `__init__`:** Each agent constructs its own `LLMClient` instance in `graph.py`. Binding `agent_name` at construction is natural and makes the association explicit at the call site. There is no sharing concern with the current architecture.

**`run_id` via `ContextVar`:** Run ID is generated in `pipeline.py` just before graph invocation and does not exist when `LLMClient` instances are constructed (which happens once at graph build time, not per run). A `ContextVar` is the standard Python mechanism for propagating request-scoped values through a call stack without threading them through every function signature. Node wrappers set the var; `complete_json()` reads it.

### Files Affected

| File | Change |
|---|---|
| `src/storymesh/llm/base.py` | Add `agent_name` and `on_call` to `LLMClient.__init__`; add `current_run_id` module-level `ContextVar` |
| `src/storymesh/llm/fake.py` (or wherever `FakeLLMClient` lives) | Add `agent_name` and `on_call` to match new signature |

### Changes to `src/storymesh/llm/base.py`

Add module-level ContextVar:

```python
from contextvars import ContextVar

# Set by node wrappers before calling agent.run(). Read by complete_json()
# when writing LLM call records. Defaults to empty string so that calls made
# outside a pipeline run (e.g., in tests) do not raise.
current_run_id: ContextVar[str] = ContextVar("current_run_id", default="")
```

Update `LLMClient.__init__`:

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    model: str | None = None,
    agent_name: str = "unknown",
    on_call: LLMCallLogger | None = None,
) -> None:
    self.api_key = api_key
    self.model = model
    self.agent_name = agent_name
    self._on_call = on_call
```

### Changes to `FakeLLMClient`

Add `agent_name: str = "fake"` and `on_call: LLMCallLogger | None = None` to `FakeLLMClient.__init__` and pass them through to `super().__init__()`. All existing tests pass `agent_name` as a keyword argument with a default, so no test changes are required unless `FakeLLMClient` has a positional-only constructor.

### Testing

- Test that `current_run_id` defaults to `""` when not set.
- Test that setting it via `.set()` in one context does not affect a concurrent context (standard `contextvars` behavior — a brief note in tests is sufficient to document the intent).
- Test that `LLMClient` stores `agent_name` and `_on_call` as instance attributes.
- Test that `FakeLLMClient` accepts the new keyword arguments without error.

---

## 5. WI-3: Instrument `complete_json()` to Write Call Records

### Rationale

With the writer and context in place, `complete_json()` can now produce a structured record on every call. This is the core of the feature.

### Design

Wrap the `self.complete()` call with a `time.perf_counter()` measurement. After the call returns, build the record dict and invoke `self._on_call` if set. On parse failure, set `parse_success: false` in the record before invoking the callback, then re-raise. This ensures failures are recorded even when retries are exhausted.

### Files Affected

| File | Change |
|---|---|
| `src/storymesh/llm/base.py` | Instrument `complete_json()` |

### Changes to `complete_json()`

The full updated method structure (pseudocode — do not generate final code without approval):

```python
def complete_json(self, prompt, *, system_prompt=None, temperature, max_tokens, max_retries=1):
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        raw = self.complete(prompt, system_prompt=system_prompt,
                            temperature=temperature, max_tokens=max_tokens)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        parse_success = True
        try:
            cleaned = _strip_markdown_fences(raw.strip())
            parsed = orjson.loads(cleaned)
        except orjson.JSONDecodeError:
            parse_success = False
            # existing warning log retained from Pipeline-Hardening-Plan WI-1
            logger.warning("JSON parse failed (attempt %d/%d). Raw response:\n%s",
                           attempt + 1, max_retries + 1, raw)
        finally:
            self._write_call_record(
                system_prompt=system_prompt,
                user_prompt=prompt,
                raw_response=raw,
                temperature=temperature,
                attempt=attempt + 1,
                latency_ms=latency_ms,
                parse_success=parse_success,
            )

        if not parse_success:
            if attempt < max_retries:
                continue
            raise  # re-raise the JSONDecodeError

        # ... isinstance check and retry logic unchanged ...
        return parsed

def _write_call_record(self, *, system_prompt, user_prompt, raw_response,
                       temperature, attempt, latency_ms, parse_success):
    """Build and dispatch one LLM call record to self._on_call, if set."""
    if self._on_call is None:
        return
    run_id = current_run_id.get()
    record = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "agent": self.agent_name,
        "model": self.model,
        "temperature": temperature,
        "attempt": attempt,
        "system_prompt": system_prompt or "",
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "parse_success": parse_success,
        "latency_ms": latency_ms,
    }
    try:
        self._on_call(run_id, record)
    except Exception:
        # Never let logging errors crash the pipeline.
        logger.warning("Failed to write LLM call record", exc_info=True)
```

Note: `datetime` and `timezone` are already imported in `pipeline.py`; add them to `base.py` imports.

### Testing

- Test that `_write_call_record` is not called when `_on_call` is `None`.
- Test that on a successful parse, a record with `parse_success=True` is dispatched.
- Test that on a parse failure, a record with `parse_success=False` is dispatched before re-raising.
- Test that an exception raised inside `_on_call` does not propagate (the pipeline must not crash due to a logging error).
- Test that `latency_ms` is a non-negative integer.
- Use `FakeLLMClient` with a list-based `on_call` collector (a plain `list.append` lambda) for all of these.

---

## 6. WI-4: Thread `agent_name` and `on_call` Through `graph.py`

### Rationale

The `ArtifactStore` instance and the per-agent name must be passed into each `LLMClient` at construction time. `graph.py`'s `build_graph()` function is already the canonical place where clients are constructed and injected into agents.

### Files Affected

| File | Change |
|---|---|
| `src/storymesh/orchestration/graph.py` | Pass `agent_name` and `on_call` to each `LLMClient` constructor |

### Changes to `build_graph()`

For each agent that uses an `LLMClient`, update the client construction to include:

```python
llm_client = AnthropicClient(
    api_key=...,
    model=...,
    agent_name="theme_extractor",          # matches graph node name
    on_call=artifact_store.log_llm_call,   # ArtifactStore already injected here
)
```

`agent_name` values must match the corresponding LangGraph node names (e.g., `"genre_normalizer"`, `"book_ranker"`, `"theme_extractor"`, `"proposal_draft"`). This keeps the JSONL records and LangSmith spans consistently labeled.

### Testing

- No new unit tests required for `graph.py` itself — this is wiring.
- The existing integration test (if any) that runs the full pipeline should now produce a `llm_calls.jsonl` file in the run directory. Add an assertion for this.
- Update any existing `test_graph.py` tests that construct clients directly to pass `agent_name` as a keyword argument (use the default `"unknown"` where the specific name does not matter for the test).

---

## 7. WI-5: Set `current_run_id` in Node Wrappers

### Rationale

The `ContextVar` established in WI-2 has no effect until something sets it. Node wrappers are the right place: they receive `state` (which contains `run_id`) and call `agent.run()`. Setting the var in the wrapper means the run ID flows into `complete_json()` without any agent being aware of it.

### Files Affected

| File | Change |
|---|---|
| `src/storymesh/orchestration/nodes/genre_normalizer.py` | Set `current_run_id` before `agent.run()` |
| `src/storymesh/orchestration/nodes/book_fetcher.py` | Same |
| `src/storymesh/orchestration/nodes/book_ranker.py` | Same |
| `src/storymesh/orchestration/nodes/theme_extractor.py` | Same |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | Same |
| *(all future node wrappers)* | Same — establish as a convention |

### Pattern

In each node function (the inner function returned by the `make_*_node` factory):

```python
from storymesh.llm.base import current_run_id

def theme_extractor_node(state: StoryMeshState) -> dict[str, Any]:
    token = current_run_id.set(state["run_id"])
    try:
        output = agent.run(...)
    finally:
        current_run_id.reset(token)   # restore previous value (good practice)
    ...
```

Using `.set()` / `.reset()` with the token is the correct `contextvars` pattern. It ensures that if LangGraph ever runs nodes concurrently, each context is isolated.

### Testing

- Test that after a node wrapper runs, `current_run_id.get()` outside the wrapper returns its previous value (i.e., the reset works).
- Test that inside the node, `current_run_id.get()` returns `state["run_id"]`.

---

## 8. WI-6: Retire Pipeline-Hardening-Plan WI-1

### Rationale

`Pipeline-Hardening-Plan.md` WI-1 proposes adding `logger.debug(...)` calls to `complete_json()` for system prompt, user prompt, and raw response. This plan supersedes that work item: structured JSONL logging is strictly more useful. The `WARNING`-level parse failure log (also part of WI-1) is retained — it is kept in WI-3 above alongside the JSONL record.

### Action

Mark WI-1 in `Pipeline-Hardening-Plan.md` as **superseded by this plan**. Add a note at the top of that section:

```
> **Status: Superseded.** The DEBUG logging described here has been replaced
> by structured JSONL call records (see LLM-Call-Logging-Plan.md). The
> WARNING log on parse failure is retained and implemented in WI-3 of that plan.
```

No code changes are required; this is documentation hygiene only.

---

## 9. Design Decision Record

This section records the key architectural choices made during planning so that future reviewers understand the reasoning.

**Why JSONL and not a single JSON array?**
A JSONL file written in append mode produces a readable partial record if the pipeline crashes mid-run. A single JSON array requires the writer to hold all records in memory and flush at the end, which loses data on crash and requires more complex bookkeeping.

**Why `agent_name` on `__init__` and not as a `complete_json()` parameter?**
Each agent constructs exactly one `LLMClient`. Binding the name at construction makes the association explicit at the call site in `graph.py` and avoids adding a parameter to `complete_json()` that every caller must provide. If a future agent ever calls `complete_json()` with a different logical name, the parameter approach could be revisited.

**Why a `ContextVar` for `run_id` and not pass it through `complete_json()`?**
`run_id` does not exist when `LLMClient` is constructed (it is generated in `pipeline.py` per invocation). Passing it through `complete_json()` would require every agent's `run()` method to accept and thread a `run_id` argument — a significant and invasive change. The `ContextVar` pattern is the standard mechanism for this in Python, used by frameworks like Starlette and structlog for exactly this purpose.

**Why not use LangSmith as the sole logging layer?**
LangSmith requires a network connection, an API key, and `LANGCHAIN_TRACING_V2=true`. It is opt-in. The JSONL file is always written to disk regardless of configuration, making it available for offline debugging, post-mortem inspection, and automated testing without any external service dependency.

**Conflict with existing plans to note:**
`Pipeline-Hardening-Plan.md` WI-1 and this plan both touch `complete_json()`. They must not be implemented simultaneously. This plan takes priority and WI-1 is retired (see WI-6).

---

## 10. Files Affected Summary

| File | Nature of Change |
|---|---|
| `src/storymesh/core/artifacts.py` | Add `log_llm_call()` method |
| `src/storymesh/llm/base.py` | Add `LLMCallLogger` alias; `current_run_id` ContextVar; `agent_name` + `on_call` on `__init__`; instrument `complete_json()`; add `_write_call_record()` |
| `src/storymesh/llm/fake.py` | Add `agent_name` + `on_call` to `FakeLLMClient.__init__` |
| `src/storymesh/orchestration/graph.py` | Pass `agent_name` + `on_call` to each client constructor |
| `src/storymesh/orchestration/nodes/genre_normalizer.py` | Set/reset `current_run_id` ContextVar |
| `src/storymesh/orchestration/nodes/book_fetcher.py` | Same |
| `src/storymesh/orchestration/nodes/book_ranker.py` | Same |
| `src/storymesh/orchestration/nodes/theme_extractor.py` | Same |
| `src/storymesh/orchestration/nodes/proposal_draft.py` | Same |
| `plans/Pipeline-Hardening-Plan.md` | Mark WI-1 as superseded |
| `tests/test_artifacts.py` | Add `log_llm_call()` tests |
| `tests/test_llm_base.py` (new or existing) | Add ContextVar, `_write_call_record`, and failure-isolation tests |

---

## 11. Validation Checklist

After all work items are complete, verify the following manually:

- [ ] Running `storymesh generate "dark fantasy thriller"` produces a `llm_calls.jsonl` file in `~/.storymesh/runs/<run_id>/`.
- [ ] The file contains one line per LLM call, each valid JSON.
- [ ] Each record contains non-empty `agent`, `model`, `system_prompt`, `user_prompt`, `raw_response`, and `run_id` fields.
- [ ] `latency_ms` is a positive integer on every record.
- [ ] Deliberately causing a parse failure (e.g., by injecting a bad `FakeLLMClient` response) produces a record with `parse_success: false`.
- [ ] A pipeline crash mid-run (e.g., killing the process) leaves a readable partial `llm_calls.jsonl`.
- [ ] With `LANGCHAIN_TRACING_V2=true`, LangSmith spans are labeled with the agent name (e.g., `"theme_extractor"`) rather than a generic method name.
- [ ] `pytest` passes with no new failures.
- [ ] `FakeLLMClient` with `on_call=None` (the default used in most existing tests) continues to work without modification.