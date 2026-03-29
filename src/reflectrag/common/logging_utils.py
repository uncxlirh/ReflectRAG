from __future__ import annotations

from datetime import datetime


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def info(message: str) -> None:
    print(f"[{timestamp()}] {message}")

