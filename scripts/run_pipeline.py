#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reflectrag.common.run_profiles import available_profiles, load_profile
from reflectrag.pipelines.runner import describe_pipeline, list_datasets, resolve_source_dir, resolve_workdir, run_pipeline


def parse_steps(text: str | None) -> list[int] | None:
    if not text:
        return None
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReflectRAG source pipelines from the clean project workspace.")
    sub = parser.add_subparsers(dest="command", required=True)

    lp = sub.add_parser("list", help="List available datasets.")

    sp = sub.add_parser("show", help="Show dataset pipeline details.")
    sp.add_argument("--dataset", required=True, choices=list_datasets())

    rp = sub.add_parser("run", help="Run selected steps for one dataset.")
    rp.add_argument("--dataset", required=True, choices=list_datasets())
    rp.add_argument("--steps", default="", help="Comma-separated steps. Omit to run the full declared chain.")
    rp.add_argument("--workdir", default="", help="Override workdir.")
    rp.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    rp.add_argument("--keys-file", default="", help="Optional API key file override.")
    rp.add_argument("--base-url", default="", help="OpenAI-compatible base URL override.")
    rp.add_argument("--profile", default="", choices=available_profiles(), help="Optional run profile from configs/runs.")
    rp.add_argument("--source-root", default="", help="Override source tree root.")
    rp.add_argument("--dry-run", action="store_true", help="Print what would run without executing it.")
    rp.add_argument("--refresh-resources", action="store_true", help="Re-copy source-side resources into the workdir.")
    rp.add_argument("--keep-going", action="store_true", help="Continue after failed steps.")
    rp.add_argument("--force", action="store_true", help="Re-run steps even if declared outputs already exist.")
    rp.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment variable in KEY=VALUE form. Can be passed multiple times.",
    )
    return parser


def parse_env_items(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value
    return out


def cmd_list() -> int:
    for dataset in list_datasets():
        print(dataset)
    return 0


def cmd_show(dataset: str) -> int:
    pipe = describe_pipeline(dataset)
    print(f"dataset: {pipe.dataset}")
    print(f"display_name: {pipe.display_name}")
    print(f"source_dir: {resolve_source_dir(dataset)}")
    print(f"default_workdir: {resolve_workdir(dataset)}")
    print(f"completeness: {pipe.completeness}")
    print(f"recommended_for_refactor: {pipe.recommended_for_refactor}")
    if pipe.resources:
        print("resources:")
        for item in pipe.resources:
            print(f"  - {item['source']} -> {item['target']}")
    print("steps:")
    for step in pipe.steps:
        outputs = ", ".join(step.outputs) if step.outputs else "-"
        print(f"  {step.number:>2}  {step.name:<24} kind={step.kind:<28} multi_key={step.multi_key}")
        print(f"      script={step.script}")
        print(f"      outputs={outputs}")
        if step.note:
            print(f"      note={step.note}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    steps = parse_steps(args.steps)
    profile = load_profile(args.profile) if args.profile else None
    extra_env = parse_env_items(args.env)
    if args.base_url:
        extra_env["OPENAI_BASE_URL"] = args.base_url
    results = run_pipeline(
        dataset=args.dataset,
        steps=steps,
        workdir=args.workdir or None,
        python_bin=args.python,
        keys_file=args.keys_file or None,
        dry_run=args.dry_run,
        refresh_resources=args.refresh_resources,
        keep_going=args.keep_going,
        extra_env=extra_env,
        source_root=args.source_root or None,
        skip_existing=not args.force,
        profile=profile,
    )
    if args.dry_run:
        print(f"dry-run complete: {len(results)} step(s)")
        return 0
    failures = [item for item in results if item.return_code != 0]
    if failures:
        print(f"completed with failures: {len(failures)}")
        return 1
    executed = sum(1 for item in results if item.status == "executed")
    skipped = sum(1 for item in results if item.status == "skipped")
    print(f"completed successfully: {len(results)} step(s), executed={executed}, skipped={skipped}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list":
        return cmd_list()
    if args.command == "show":
        return cmd_show(args.dataset)
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
