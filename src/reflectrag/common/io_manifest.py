from __future__ import annotations

from pathlib import Path
from typing import Iterable


def file_state(base_dir: Path, rel_paths: Iterable[str]) -> list[dict]:
    states: list[dict] = []
    for rel in rel_paths:
        path = base_dir / rel
        exists = path.exists()
        states.append(
            {
                "path": rel,
                "exists": exists,
                "size": path.stat().st_size if exists else 0,
            }
        )
    return states

