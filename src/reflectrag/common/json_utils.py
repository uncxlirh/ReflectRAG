from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_json_loose(text: str) -> Any:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def load_json_or_jsonl(path: str | Path) -> list[Any]:
    target = Path(path)
    if not target.exists():
        alt = target.with_suffix(".jsonl")
        if alt.exists():
            target = alt
    if not target.exists():
        raise FileNotFoundError(target)

    if target.suffix == ".jsonl":
        rows: list[Any] = []
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    return [data]


def dump_json(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def dump_jsonl(path: str | Path, rows: list[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

