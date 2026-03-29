from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModuleCheck:
    name: str
    available: bool
    version: str
    detail: str


def python_summary() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_command(name: str) -> bool:
    return shutil.which(name) is not None


def check_module(name: str) -> ModuleCheck:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return ModuleCheck(name=name, available=True, version=str(version), detail="")
    except Exception as exc:
        return ModuleCheck(
            name=name,
            available=False,
            version="",
            detail=f"{exc.__class__.__name__}: {exc}",
        )


def check_modules(names: Iterable[str]) -> list[ModuleCheck]:
    return [check_module(name) for name in names]


def openai_keys_present(keys_file: str | None = None) -> bool:
    env_keys = os.getenv("OPENAI_API_KEYS", "").strip()
    if env_keys:
        return True
    if not keys_file:
        return False
    if not os.path.exists(keys_file):
        return False
    with open(keys_file, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                return True
    return False

