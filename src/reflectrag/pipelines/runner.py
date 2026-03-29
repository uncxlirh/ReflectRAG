from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reflectrag.common.logging_utils import info
from reflectrag.common.paths import ensure_runtime_dirs, get_paths
from reflectrag.common.pipeline_registry import DatasetPipeline, StepDefinition, available_datasets, load_pipeline
from reflectrag.common.run_profiles import RunProfile
from reflectrag.common.secrets import load_api_keys


@dataclass(frozen=True)
class StepRunResult:
    step: int
    return_code: int
    log_path: Path
    status: str


def list_datasets() -> list[str]:
    return available_datasets()


def resolve_source_dir(dataset: str, source_root: str | None = None) -> Path:
    paths = get_paths()
    pipe = load_pipeline(dataset)
    root = Path(source_root).expanduser().resolve() if source_root else paths.source_root
    return root / pipe.source_subdir


def resolve_workdir(dataset: str, workdir: str | None = None) -> Path:
    pipe = load_pipeline(dataset)
    if workdir:
        return Path(workdir).expanduser().resolve()
    return (get_paths().root / pipe.default_workdir).resolve()


def ensure_resources(dataset: str, workdir: Path, refresh: bool = False, source_root: str | None = None) -> None:
    pipe = load_pipeline(dataset)
    source_dir = resolve_source_dir(dataset, source_root=source_root)
    for resource in pipe.resources:
        src = source_dir / resource["source"]
        dst = workdir / resource["target"]
        if not src.exists():
            continue
        if dst.exists() and refresh:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def selected_steps(dataset: str, requested: list[int] | None = None) -> list[StepDefinition]:
    pipe = load_pipeline(dataset)
    if requested is None:
        return pipe.steps
    step_map = pipe.step_map()
    missing = [step for step in requested if step not in step_map]
    if missing:
        raise ValueError(f"Unsupported steps for dataset={dataset}: {missing}")
    return [step_map[num] for num in requested]


def build_env(keys_file: str | None = None, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    paths = get_paths()
    keys = load_api_keys(keys_file)
    if keys:
        env["OPENAI_API_KEYS"] = ",".join(keys)
    env.setdefault("INIT_MODEL_DIR", str(paths.shared_init_lora))
    env.setdefault("PEFT_MODEL_NAME", str(paths.shared_init_lora))
    if extra_env:
        env.update(extra_env)
    return env


def step_outputs_exist(workdir: Path, outputs: Iterable[str]) -> bool:
    output_list = [Path(item) for item in outputs]
    if not output_list:
        return False
    return all((workdir / output).exists() for output in output_list)


def _skip_log_path(workdir: Path, step: int) -> Path:
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"step{step}.log"


def run_step(
    dataset: str,
    step: StepDefinition,
    python_bin: str,
    workdir: Path,
    keys_file: str | None = None,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
    source_root: str | None = None,
    skip_existing: bool = True,
) -> StepRunResult:
    if skip_existing and step_outputs_exist(workdir, step.outputs):
        log_path = _skip_log_path(workdir, step.number)
        info(f"dataset={dataset} step={step.number} skipped because outputs already exist")
        return StepRunResult(step=step.number, return_code=0, log_path=log_path, status="skipped")

    source_dir = resolve_source_dir(dataset, source_root=source_root)
    script = source_dir / step.script
    if not script.exists():
        raise FileNotFoundError(f"Missing source script: {script}")

    env = build_env(keys_file=keys_file, extra_env=extra_env)
    logs_dir = workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"step{step.number}.log"
    cmd = [python_bin, str(script)]

    info(f"dataset={dataset} step={step.number} kind={step.kind}")
    info(f"cwd={workdir}")
    info(f"cmd={' '.join(cmd)}")

    if dry_run:
        return StepRunResult(step=step.number, return_code=0, log_path=log_path, status="dry-run")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# dataset={dataset}\n")
        log.write(f"# step={step.number}\n")
        log.write(f"# kind={step.kind}\n")
        log.write(f"# cwd={workdir}\n")
        log.write(f"# cmd={' '.join(cmd)}\n\n")
        rc = subprocess.call(cmd, cwd=str(workdir), stdout=log, stderr=subprocess.STDOUT, env=env)
    return StepRunResult(step=step.number, return_code=rc, log_path=log_path, status="executed")


def run_pipeline(
    dataset: str,
    steps: list[int] | None = None,
    workdir: str | None = None,
    python_bin: str | None = None,
    keys_file: str | None = None,
    dry_run: bool = False,
    refresh_resources: bool = False,
    keep_going: bool = False,
    extra_env: dict[str, str] | None = None,
    source_root: str | None = None,
    skip_existing: bool = True,
    profile: RunProfile | None = None,
) -> list[StepRunResult]:
    ensure_runtime_dirs()
    pipe = load_pipeline(dataset)
    target_workdir = resolve_workdir(dataset, workdir)
    target_workdir.mkdir(parents=True, exist_ok=True)
    ensure_resources(dataset, target_workdir, refresh=refresh_resources, source_root=source_root)

    interpreter = python_bin or sys.executable
    effective_keys_file = keys_file
    effective_skip_existing = skip_existing
    merged_env = dict(extra_env or {})
    if profile:
        if profile.base_url and "OPENAI_BASE_URL" not in merged_env:
            merged_env["OPENAI_BASE_URL"] = profile.base_url
        if not effective_keys_file and profile.keys_file:
            effective_keys_file = profile.keys_file
        if skip_existing is True:
            effective_skip_existing = profile.skip_existing

    results: list[StepRunResult] = []
    failures = 0
    for step in selected_steps(dataset, steps):
        result = run_step(
            dataset=dataset,
            step=step,
            python_bin=interpreter,
            workdir=target_workdir,
            keys_file=effective_keys_file,
            dry_run=dry_run,
            extra_env=merged_env,
            source_root=source_root,
            skip_existing=effective_skip_existing,
        )
        results.append(result)
        if result.return_code != 0:
            failures += 1
            info(f"step {step.number} failed -> {result.log_path}")
            if not keep_going:
                break
        else:
            outputs = ", ".join(step.outputs) if step.outputs else "(no declared outputs)"
            info(f"step {step.number} {result.status} -> {outputs}")

    if failures and not keep_going:
        raise RuntimeError(f"Pipeline stopped after {failures} failed step(s)")
    return results


def describe_pipeline(dataset: str) -> DatasetPipeline:
    return load_pipeline(dataset)
