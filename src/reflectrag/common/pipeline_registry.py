from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reflectrag.common.paths import get_paths


@dataclass(frozen=True)
class StepDefinition:
    number: int
    name: str
    kind: str
    script: str
    outputs: list[str]
    multi_key: bool
    note: str = ""


@dataclass(frozen=True)
class DatasetPipeline:
    dataset: str
    display_name: str
    source_subdir: str
    default_workdir: str
    completeness: str
    recommended_for_refactor: bool
    resources: list[dict[str, str]]
    steps: list[StepDefinition]

    def step_map(self) -> dict[int, StepDefinition]:
        return {step.number: step for step in self.steps}

    @property
    def workdir(self) -> Path:
        return get_paths().root / self.default_workdir


def _config_dir() -> Path:
    return get_paths().configs / "datasets"


def available_datasets() -> list[str]:
    return sorted(path.stem for path in _config_dir().glob("*.yaml"))


def load_pipeline(dataset: str) -> DatasetPipeline:
    path = _config_dir() / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown dataset config: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    steps = [
        StepDefinition(
            number=int(step["number"]),
            name=str(step["name"]),
            kind=str(step["kind"]),
            script=str(step["script"]),
            outputs=[str(x) for x in step.get("outputs", [])],
            multi_key=bool(step.get("multi_key", False)),
            note=str(step.get("note", "")),
        )
        for step in raw.get("steps", [])
    ]

    return DatasetPipeline(
        dataset=str(raw["dataset"]),
        display_name=str(raw.get("display_name", raw["dataset"])),
        source_subdir=str(raw["source_subdir"]),
        default_workdir=str(raw["default_workdir"]),
        completeness=str(raw.get("completeness", "unknown")),
        recommended_for_refactor=bool(raw.get("recommended_for_refactor", False)),
        resources=[dict(item) for item in raw.get("resources", [])],
        steps=steps,
    )
