#!/usr/bin/env python3
"""
SelfRAG baseline reimplemented with GPT-3.5-turbo.

This keeps the baseline simple:
  1. grade passage relevance over the reranked top-k passages
  2. answer from the retained passages
  3. check support
  4. regenerate once from all passages if needed

Unlike the earlier broken run, this version validates API keys up front and
removes bad keys at runtime so fixed subsets are not poisoned by 401 failures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reflectrag.common.baseline_api_pool import ChatKeyPool, mask_key, probe_working_keys


BASE_URL = os.getenv("OPENAI_BASE_URL", "https://apic.littlewheat.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
KEYS_FILE = os.getenv(
    "OPENAI_API_KEYS_FILE",
    str(Path(__file__).resolve().parents[2] / ".secrets" / "openai_api_keys.txt"),
)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
TEMPERATURE = 0.0
MAX_TOKENS = 256
TOP_K_PASSAGES = 5
MAX_RETRIES = 3


def chat(api_pool, system, user, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    return api_pool.complete(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(gt)
    return 2 * p * r / (p + r)


def max_f1(pred, golds):
    return max((f1_score(pred, g) for g in golds), default=0.0)


SYSTEM_GENERATE = (
    "You are a question-answering assistant. Given a question and supporting "
    "passages, provide a concise, direct answer. Do not explain or elaborate."
)
SYSTEM_RELEVANCE = (
    "You are a relevance grader. Given a question and a passage, determine if "
    'the passage is relevant to answering the question. Reply with ONLY "relevant" or "irrelevant".'
)
SYSTEM_SUPPORT = (
    "You are a grounding checker. Given a question, an answer, and supporting "
    'passages, determine if the answer is supported by the passages. Reply with ONLY "supported" or "not supported".'
)


def user_generate(query, passages):
    ctx = "\n\n".join(f"[Passage {i+1}] {p}" for i, p in enumerate(passages))
    return f"""Passages:
{ctx}

Question: {query}

Answer (concise, directly answer the question):"""


def user_relevance(query, passage):
    return f"""Question: {query}

Passage: {passage}

Is this passage relevant to answering the question? Reply with ONLY "relevant" or "irrelevant"."""


def user_support(query, answer, passages):
    ctx = "\n\n".join(f"[Passage {i+1}] {p}" for i, p in enumerate(passages))
    return f"""Question: {query}

Answer: {answer}

Passages:
{ctx}

Is this answer supported by the passages? Reply with ONLY "supported" or "not supported"."""


def run_selfrag_single(api_pool, query, all_passages, gold_answers):
    telemetry = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _chat(sys_prompt, user_prompt, **kw):
        resp, usage = chat(api_pool, sys_prompt, user_prompt, **kw)
        telemetry["calls"] += 1
        if usage:
            telemetry["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            telemetry["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return resp

    passages = all_passages[:TOP_K_PASSAGES]
    relevant_passages = []
    for passage in passages:
        verdict = _chat(SYSTEM_RELEVANCE, user_relevance(query, passage), max_tokens=10)
        if "relevant" in verdict.lower() and "irrelevant" not in verdict.lower():
            relevant_passages.append(passage)

    if not relevant_passages:
        relevant_passages = passages

    answer = _chat(SYSTEM_GENERATE, user_generate(query, relevant_passages))
    support = _chat(SYSTEM_SUPPORT, user_support(query, answer, relevant_passages), max_tokens=10)

    if "not supported" in support.lower():
        regen_prompt = user_generate(query, passages) + "\nIMPORTANT: Base your answer strictly on the passages above."
        answer = _chat(SYSTEM_GENERATE, regen_prompt)

    norm = normalize_answer(answer)
    f1 = max_f1(norm, gold_answers)
    return {
        "query": query,
        "raw_answer": answer,
        "normalized_answer": norm,
        "gold_answers": gold_answers,
        "n_relevant_passages": len(relevant_passages),
        "supported": "not supported" not in support.lower(),
        "f1": f1,
        "telemetry": telemetry,
    }


def load_data(workdir):
    reranked_file = workdir / "reranked_gpt3.5.json"
    if not reranked_file.exists():
        sys.exit(f"Missing {reranked_file}")
    reranked = json.load(open(reranked_file, "r", encoding="utf-8"))

    plan_file = workdir / "answers_plan.json"
    if not plan_file.exists():
        sys.exit(f"Missing {plan_file}")
    plan_data = json.load(open(plan_file, "r", encoding="utf-8"))

    query_to_gold = {}
    for item in plan_data:
        query_to_gold[item.get("query", "")] = item.get("gold_answers", [])

    items = []
    for entry in reranked:
        q = entry.get("query", "")
        passages = []
        refs = entry.get("hits", entry.get("references", entry.get("reranked_passages", [])))
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict):
                    passages.append((ref.get("content") or ref.get("text") or ref.get("passage") or "")[:800])
                elif isinstance(ref, str):
                    passages.append(ref[:800])
        items.append({"query": q, "passages": passages, "gold_answers": query_to_gold.get(q, [])})
    return items


def metrics_path_for(workdir, output_name):
    output_path = Path(output_name)
    if output_path.name == "answers_selfrag.json":
        return workdir / "metrics_selfrag.json"
    return workdir / f"{output_path.stem}_metrics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True, help="Pipeline workdir (e.g. experiments/marco/runs/seed99)")
    parser.add_argument("--n", type=int, default=0, help="Limit to first N queries (0=all)")
    parser.add_argument("--output", default="answers_selfrag.json", help="Output filename")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    healthy_keys, probe_results = probe_working_keys(
        KEYS_FILE,
        base_url=BASE_URL,
        model=MODEL,
        max_workers=MAX_WORKERS,
    )
    bad_keys = [item for item in probe_results if not item.ok]
    print(f"[SelfRAG-GPT] keys={len(probe_results)} healthy={len(healthy_keys)} model={MODEL} base_url={BASE_URL}")
    if bad_keys:
        masked = ", ".join(mask_key(item.key) for item in bad_keys)
        print(f"[SelfRAG-GPT] dropped bad keys: {masked}")
    if not healthy_keys:
        sys.exit("No healthy API keys available after validation.")

    items = load_data(workdir)
    if args.n > 0:
        items = items[:args.n]
    print(f"[SelfRAG-GPT] Loaded {len(items)} queries")

    api_pool = ChatKeyPool(
        healthy_keys,
        base_url=BASE_URL,
        model=MODEL,
        timeout=60,
        max_retries=MAX_RETRIES,
    )

    results = [None] * len(items)
    total_telemetry = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()

    def process(idx, item):
        return idx, run_selfrag_single(api_pool, item["query"], item["passages"], item["gold_answers"])

    done = 0
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, api_pool.healthy_count))) as pool:
        futures = {pool.submit(process, i, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result
            tel = result.pop("telemetry")
            for key in total_telemetry:
                total_telemetry[key] += tel[key]
            done += 1
            if done % 50 == 0 or done == len(items):
                elapsed = time.time() - t0
                avg_f1 = sum(r["f1"] for r in results[:done] if r) / done
                print(f"  [{done}/{len(items)}] avg_f1={avg_f1:.4f} elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    avg_f1 = sum(r["f1"] for r in results if r) / max(1, len(results))

    out_path = workdir / args.output
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    metrics = {
        "method": "SelfRAG-GPT",
        "model": MODEL,
        "count": len(results),
        "f1": avg_f1,
        "elapsed_sec": elapsed,
        "qps": len(results) / max(elapsed, 1e-6),
        **total_telemetry,
    }
    metrics_path = metrics_path_for(workdir, args.output)
    json.dump(metrics, open(metrics_path, "w", encoding="utf-8"), indent=2)
    print(f"\n[SelfRAG-GPT] F1={avg_f1:.4f} | {len(results)} queries | {elapsed:.0f}s")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {metrics_path}")


if __name__ == "__main__":
    main()
