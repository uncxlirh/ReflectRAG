from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from reflectrag.common.paths import get_paths


@dataclass(frozen=True)
class RunProfile:
    name: str
    description: str
    base_url: str
    keys_file: str
    skip_existing: bool


def _profiles_dir() -> Path:
    return get_paths().configs / "runs"


def available_profiles() -> list[str]:
    return sorted(path.stem for path in _profiles_dir().glob("*.yaml"))


def load_profile(name: str) -> RunProfile:
    path = _profiles_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown run profile: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    return RunProfile(
        name=str(raw["run_name"]),
        description=str(raw.get("description", "")),
        base_url=str(raw.get("base_url", "")).strip(),
        keys_file=str(raw.get("keys_file", "")).strip(),
        skip_existing=bool(raw.get("skip_existing", True)),
    )

