#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reflectrag.common.paths import get_paths
from reflectrag.common.run_profiles import available_profiles, load_profile
from reflectrag.pipelines.ablation_marco import AblationConfig, run_ablation


PRESETS = {
    "prompt_only_reflection": dict(reflection_mode="prompt_only", iterations=3, use_grpo_reflector=False, enable_filtering=True, enable_yesno_norm=True, use_gold_in_reflection=False, use_gold_score_in_reflection=False),
    "grpo_single_pass": dict(reflection_mode="trained", iterations=1, use_grpo_reflector=True, enable_filtering=True, enable_yesno_norm=True, use_gold_in_reflection=False, use_gold_score_in_reflection=False),
    "iter_t1": dict(reflection_mode="trained", iterations=1, use_grpo_reflector=True, enable_filtering=True, enable_yesno_norm=True, use_gold_in_reflection=False, use_gold_score_in_reflection=False),
    "iter_t2": dict(reflection_mode="trained", iterations=2, use_grpo_reflector=True, enable_filtering=True, enable_yesno_norm=True, use_gold_in_reflection=False, use_gold_score_in_reflection=False),
    "iter_t3": dict(reflection_mode="trained", iterations=3, use_grpo_reflector=True, enable_filtering=True, enable_yesno_norm=True, use_gold_in_reflection=False, use_gold_score_in_reflection=False),
    "heuristics_off": dict(reflection_mode="trained", iterations=3, use_grpo_reflector=True, enable_filtering=False, enable_yesno_norm=False, use_gold_in_reflection=False, use_gold_score_in_reflection=False),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run reviewer-driven dataset ablations with cost/latency and IO recording.")
    p.add_argument("--dataset", default="marco", choices=["marco", "nq", "tqa"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--profile", default="third_party_resume", choices=available_profiles())
    p.add_argument("--workdir", default="", help="Path containing dataset step outputs, especially answers_plan.json")
    p.add_argument("--output-dir", default="", help="Directory for ablation artifacts")
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=0, help="Override sample-level concurrency for ablation runs")
    return p


def main() -> int:
    args = parser().parse_args()
    profile = load_profile(args.profile)
    preset = PRESETS[args.preset]
    paths = get_paths()
    
    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        base_runs = getattr(paths, f"{args.dataset}_runs")
        subdir = f"seed{args.seed}" if args.seed is not None else "default"
        workdir = base_runs / subdir

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = paths.experiments / args.dataset / "ablations" / args.preset
        if args.seed is not None:
            output_dir = output_dir / f"seed{args.seed}"

    config = AblationConfig(
        name=args.preset,
        model=args.model,
        base_url=profile.base_url,
        max_items=args.max_items,
        max_workers=args.max_workers if args.max_workers > 0 else 8,
        **preset,
    )
    summary = run_ablation(config=config, workdir=workdir, output_dir=output_dir, keys_file=profile.keys_file or None)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
