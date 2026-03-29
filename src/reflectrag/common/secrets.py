from __future__ import annotations

import os
from pathlib import Path

from reflectrag.common.paths import get_paths


def _parse_key_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    if len(lines) == 1 and "," in lines[0]:
        lines = [item.strip() for item in lines[0].split(",") if item.strip()]

    seen: set[str] = set()
    out: list[str] = []
    for key in lines:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def load_api_keys(keys_file: str | Path | None = None) -> list[str]:
    env_keys = _parse_key_text(os.getenv("OPENAI_API_KEYS", ""))
    if env_keys:
        return env_keys

    target = Path(keys_file) if keys_file else get_paths().default_keys_file
    if not target.exists():
        return []
    return _parse_key_text(target.read_text(encoding="utf-8", errors="ignore"))


def primary_api_key(keys_file: str | Path | None = None) -> str | None:
    keys = load_api_keys(keys_file)
    return keys[0] if keys else None

