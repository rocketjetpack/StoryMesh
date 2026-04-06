import shutil
from pathlib import Path
from typing import Any, cast

import orjson


def persist_node_output(
    artifact_store: "ArtifactStore",
    run_id: str,
    stage_name: str,
    output: object,
) -> None:
    """Write a node's output as a JSON artifact within its run directory.

    Called by node wrapper functions immediately after agent execution so
    that artifacts are written incrementally rather than only after the full
    graph completes. This means a crash mid-pipeline still leaves partial
    artifacts on disk for post-mortem inspection.

    No-ops silently when ``output`` is ``None`` or not serialisable as a dict.

    Args:
        artifact_store: The ``ArtifactStore`` instance for this run.
        run_id: Unique run identifier from ``StoryMeshState``.
        stage_name: Pipeline stage name (e.g. ``'genre_normalizer'``).
        output: Stage output — a Pydantic model, a plain dict, or ``None``.
    """
    if output is None:
        return
    if hasattr(output, "model_dump"):
        data: dict[str, Any] = output.model_dump()
    elif isinstance(output, dict):
        data = output
    else:
        return
    artifact_store.save_run_file(run_id, f"{stage_name}_output.json", data)


class ArtifactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(f"{Path.home()}/.storymesh")
        self.stages_dir = self.root / "stages"
        self.runs_dir = self.root / "runs"
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def stage_path(self, stage_name: str, stage_hash: str) -> Path:
        """
        Retrieve the path for storage of artifacts based on the stage name and hash.
        
        :param self: Description
        :param stage_name: Description
        :type stage_name: str
        :param stage_hash: Description
        :type stage_hash: str
        :return: Description
        :rtype: Path
        """
        stage_dir = self.stages_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / f"{stage_hash}.json"
    
    def save_stage(self, stage_name: str, stage_hash: str, data: dict[str, Any]) -> None:
        """
        Save stage artifacts to the appropriate path based on the stage name and hash.
        
        :param self: Description
        :param stage_name: Description
        :type stage_name: str
        :param stage_hash: Description
        :type stage_hash: str
        :param data: Description
        :type data: Dict[str, Any]
        """
        path = self.stage_path(stage_name, stage_hash)
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def save_run_file(self, run_hash: str, filename: str, data: dict[str, Any]) -> None:
        """
        Save a JSON file within a run directory.

        Args:
            run_hash: Unique identifier for a specific run.
            filename: Name of the file to write
            data: The data to write as JSON
        """
        run_dir = self.runs_dir / run_hash
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / filename).write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def load_stage(self, stage_name: str, stage_hash: str) -> dict[str, Any] | None:
        """
        Load stage artifacts from the appropriate path based on the stage name and hash.
        
        :param self: Description
        :param stage_name: Description
        :type stage_name: str
        :param stage_hash: Description
        :type stage_hash: str
        :return: Description
        :rtype: Dict[str, Any] | None
        """
        path = self.stage_path(stage_name, stage_hash)
        if path.exists():
            raw = orjson.loads(path.read_bytes())
            return cast(dict[str, Any], raw)
        return None
    
    def save_run(self, run_hash: str, data: dict[str, Any]) -> None:
        """
        Save run artifacts to the appropriate path based on the run hash.

        :param self: Description
        :param run_hash: Description
        :type run_hash: str
        :param data: Description
        :type data: Dict[str, Any]
        """

        self.save_run_file(run_hash, "run_metadata.json", data)

    def load_run_file(self, run_id: str, filename: str) -> bytes | None:
        """Return raw bytes from a file within a run directory, or None if absent.

        Args:
            run_id: Unique run identifier (matches the run directory name).
            filename: Name of the file to read.

        Returns:
            Raw bytes of the file, or ``None`` if the file does not exist.
        """
        path = self.runs_dir / run_id / filename
        return path.read_bytes() if path.exists() else None

    def list_run_ids(self) -> list[str]:
        """Return all run IDs sorted by modification time, newest first.

        Returns:
            List of run directory names (run IDs), most recently modified first.
            Returns an empty list when ``runs_dir`` does not exist or is empty.
        """
        if not self.runs_dir.exists():
            return []
        subdirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return [d.name for d in subdirs]

    def purge_stage_cache(self) -> int:
        """Delete all cached stage output files.

        Removes every per-stage subdirectory (and its JSON files) from
        ``stages_dir``.  The parent directory itself is preserved so that the
        next pipeline run can write into it without having to recreate the
        directory structure.

        Returns:
            Number of cached JSON files removed.
        """
        count = 0
        if not self.stages_dir.exists():
            return count
        for stage_subdir in self.stages_dir.iterdir():
            if stage_subdir.is_dir():
                count += sum(1 for _ in stage_subdir.glob("*.json"))
                shutil.rmtree(stage_subdir)
        return count

    def purge_runs(self) -> int:
        """Delete all per-run artifact directories.

        Removes every run subdirectory from ``runs_dir``.  The parent
        directory itself is preserved.

        Returns:
            Number of run directories removed.
        """
        count = 0
        if not self.runs_dir.exists():
            return count
        for run_subdir in self.runs_dir.iterdir():
            if run_subdir.is_dir():
                shutil.rmtree(run_subdir)
                count += 1
        return count

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
