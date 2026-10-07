"""File-bound stages for input preparation, including partial-attempt recovery."""

from dataclasses import dataclass
from pathlib import Path
import json
import re
import time
from typing import Callable

from v2d.common.artifacts import artifact_record, atomic_json


@dataclass(frozen=True)
class PreparationStages:
    root: Path
    identity_path: Path

    def run(self, name: str, inputs: tuple[Path, ...], outputs: tuple[Path, ...],
            operation: Callable[[], None]) -> dict:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not outputs:
            raise ValueError("A preparation stage needs a simple name and declared outputs")
        if {p.resolve() for p in outputs} & {p.resolve() for p in (self.identity_path, *inputs)}:
            raise ValueError("Preparation stages cannot overwrite their inputs")
        signature = [artifact_record(path) for path in (self.identity_path, *inputs)]
        marker = self.root / ".stages" / f"{name}.json"
        if any(not path.resolve().is_relative_to(self.root.resolve()) for path in outputs):
            raise ValueError("Preparation stage outputs must stay inside their episode directory")
        if marker.exists():
            previous = json.loads(marker.read_text())
            actual = [artifact_record(p) if p.is_file() else None for p in outputs]
            if (previous.get("schema") == "v2d.track1.preparation_stage.v1"
                    and previous.get("inputs") == signature and previous.get("outputs") == actual):
                return {"name": name, "reused": True, "elapsed_seconds": 0.0}
            raise ValueError(f"Stage {name} identity changed; use a new preparation output_dir")
        # No success marker: only these declared files can belong to an
        # interrupted attempt. Never adopt or delete a completed stage's data.
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
        started = time.monotonic()
        operation()
        identities = [artifact_record(path) for path in outputs]
        if any(item["size"] == 0 for item in identities):
            raise ValueError(f"Stage {name} produced an empty output")
        if signature != [artifact_record(path) for path in (self.identity_path, *inputs)]:
            raise ValueError(f"Inputs changed during {name}; use a new preparation output_dir")
        atomic_json(marker, {"schema": "v2d.track1.preparation_stage.v1", "inputs": signature,
                             "outputs": identities})
        return {"name": name, "reused": False, "elapsed_seconds": time.monotonic() - started}
