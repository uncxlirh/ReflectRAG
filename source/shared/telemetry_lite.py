"""Lightweight per-step telemetry for the source pipeline scripts.

Usage (drop-in, zero config):
    from shared.telemetry_lite import StepTelemetry
    tel = StepTelemetry("2-gptrerank")          # creates telemetry_2-gptrerank.jsonl
    tel.record(response)                         # extracts usage from OpenAI response
    ...
    tel.save_summary()                           # writes telemetry_2-gptrerank_summary.json
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CallRecord:
    stage: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    success: bool
    error: str = ""


class StepTelemetry:
    def __init__(self, step_name: str, output_dir: str = "."):
        self.step_name = step_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.output_dir / f"telemetry_{step_name}.jsonl"
        self._records: list[CallRecord] = []
        self._t0 = time.perf_counter()

    def record_call(self, response: Any, model: str = "", latency_s: float = 0.0,
                    success: bool = True, error: str = "") -> None:
        """Extract usage from an OpenAI ChatCompletion response and record it."""
        usage = getattr(response, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or 0)
        rec = CallRecord(
            stage=self.step_name, latency_s=round(latency_s, 4),
            prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
            model=model, success=success, error=error,
        )
        self._records.append(rec)
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def record_failure(self, model: str = "", latency_s: float = 0.0, error: str = "") -> None:
        self.record_call(None, model=model, latency_s=latency_s, success=False, error=error)

    def save_summary(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self._t0
        summary = {
            "step": self.step_name,
            "wall_clock_s": round(elapsed, 2),
            "total_calls": len(self._records),
            "successful_calls": sum(1 for r in self._records if r.success),
            "failed_calls": sum(1 for r in self._records if not r.success),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self._records),
            "total_completion_tokens": sum(r.completion_tokens for r in self._records),
            "total_tokens": sum(r.total_tokens for r in self._records),
            "total_api_latency_s": round(sum(r.latency_s for r in self._records), 2),
        }
        out_path = self.output_dir / f"telemetry_{self.step_name}_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
