from __future__ import annotations

import sys
from pathlib import Path


def run(dataset_name: str, adapter_dir_default: str) -> int:
    here = Path(__file__).resolve()
    source_root = here.parents[1]
    repo_root = here.parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(source_root))

    from shared.telemetry_lite import StepTelemetry
    from reflectrag.pipelines.iterative_reflection_runtime import parse_runtime_args, run_dataset

    telemetry = StepTelemetry("10-reflect-iterative")
    runtime = parse_runtime_args(dataset_name=dataset_name, adapter_dir_default=adapter_dir_default)
    return run_dataset(runtime, telemetry=telemetry)

