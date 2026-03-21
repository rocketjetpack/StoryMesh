from pathlib import Path
from typing import Any, cast

import orjson


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
