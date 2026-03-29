#!/usr/bin/env python3
"""
ReAct baseline reimplemented with GPT-3.5-turbo.

This version intentionally stays plain: single-trajectory, local-doc Search /
Lookup / Finish, and no extra voting tricks by default.

Usage:
    python run_react.py --workdir experiments/marco/runs/seed99
    python run_react.py --workdir experiments/marco/runs/seed99 --n 50  # quick test
"""
import argparse, json, os, sys, time, re, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reflectrag.common.baseline_api_pool import ChatKeyPool, mask_key, probe_working_keys

# ── config ───────────────────────────────────────────────────────
BASE_URL     = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL        = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
KEYS_FILE    = os.getenv("OPENAI_API_KEYS_FILE",
                          str(Path(__file__).resolve().parents[2] / ".secrets" / "openai_api_keys.txt"))
MAX_WORKERS  = int(os.getenv("MAX_WORKERS", "8"))
MAX_STEPS    = 7
N_TRAJ       = int(os.getenv("N_TRAJ", "1"))     # keep the default ordinary
TOP_K        = 10                                  # passages provided
DOC_TRUNC    = 600                                 # chars per doc in context
MAX_RETRIES  = 3

# ── helpers ──────────────────────────────────────────────────────
def chat(api_pool, messages, temperature=0.0, max_tokens=256):
    return api_pool.complete(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\nObservation"],
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

# ── ReAct system prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps.

You are given a set of documents. You can use the following actions:
(1) Search[Doc N] - Select and read document N (e.g., "Search[Doc 3]")
(2) Lookup[keyword] - Find the next sentence containing keyword in the last selected document
(3) Finish[your final answer] - Return the final answer (short and concise)

Rules:
- Only use information from the given documents
- Keep your answer as concise as possible (a few words or a short phrase)
- You MUST eventually call Finish[your final answer] to provide your answer
- Do not output the literal placeholder "answer" inside Finish[...]
- If you cannot find the answer after searching, call Finish with your best guess based on the documents"""

def build_docs_block(passages):
    lines = []
    for i, p in enumerate(passages):
        text = p[:DOC_TRUNC]
        lines.append(f"Doc {i+1}: {text}")
    return "\n\n".join(lines)

# ── single trajectory ────────────────────────────────────────────
def run_trajectory(api_pool, query, passages, temperature=0.0):
    """Run one ReAct trajectory, return (answer, n_calls, trajectory_text)."""
    docs_block = build_docs_block(passages)
    selected_doc = None
    selected_doc_text = ""
    lookup_pos = {}  # keyword -> last position

    user_msg = f"""Documents:
{docs_block}

Question: {query}

Begin! Remember to use Finish[your final answer] to provide your final answer."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    trajectory = []
    n_calls = 0
    answer = ""

    for step in range(1, MAX_STEPS + 1):
        # Ask model for next Thought + Action
        prompt_suffix = f"Thought {step}:"
        messages.append({"role": "assistant", "content": prompt_suffix})

        resp, usage = chat(api_pool, messages, temperature=temperature, max_tokens=200)
        n_calls += 1

        # Merge the prompt_suffix with response
        full_step = prompt_suffix + " " + (resp or "")
        messages[-1] = {"role": "assistant", "content": full_step}
        trajectory.append(full_step)

        # Parse action
        action_match = re.search(r"Action\s*\d*\s*:\s*(\w+)\[([^\]]*)\]", full_step)
        if not action_match:
            # No action found, try to extract Finish from the text
            finish_match = re.search(r"Finish\[([^\]]*)\]", full_step)
            if finish_match:
                answer = finish_match.group(1).strip()
                break
            # Force continue
            observation = "Invalid action format. Please use Search[Doc N], Lookup[keyword], or Finish[your final answer]."
        else:
            action_type = action_match.group(1).strip()
            action_arg = action_match.group(2).strip()

            if action_type.lower() == "finish":
                candidate = action_arg.strip()
                if candidate.lower() in {"answer", "your answer", "final answer", "your final answer"}:
                    observation = (
                        f"Observation {step}: Invalid Finish placeholder. "
                        "Replace it with the actual short answer from the documents."
                    )
                else:
                    answer = candidate
                    break
            elif action_type.lower() == "search":
                # Parse doc number
                doc_num_match = re.search(r"\d+", action_arg)
                if doc_num_match:
                    doc_idx = int(doc_num_match.group()) - 1
                    if 0 <= doc_idx < len(passages):
                        selected_doc = doc_idx
                        selected_doc_text = passages[doc_idx]
                        observation = f"Observation {step}: {selected_doc_text[:500]}"
                    else:
                        observation = f"Observation {step}: Document {doc_idx+1} not found. Available: Doc 1 to Doc {len(passages)}."
                else:
                    # Keyword-based search
                    kw = action_arg.lower()
                    found = False
                    for i, p in enumerate(passages):
                        if kw in p.lower():
                            selected_doc = i
                            selected_doc_text = p
                            observation = f"Observation {step}: {p[:500]}"
                            found = True
                            break
                    if not found:
                        observation = f"Observation {step}: No document found matching '{action_arg}'."
            elif action_type.lower() == "lookup":
                if selected_doc_text:
                    keyword = action_arg.lower()
                    sentences = re.split(r'[.!?]+', selected_doc_text)
                    start = lookup_pos.get(keyword, 0)
                    found = False
                    for i in range(start, len(sentences)):
                        if keyword in sentences[i].lower():
                            observation = f"Observation {step}: (Result) {sentences[i].strip()}"
                            lookup_pos[keyword] = i + 1
                            found = True
                            break
                    if not found:
                        observation = f"Observation {step}: No more results for '{action_arg}' in current document."
                else:
                    observation = f"Observation {step}: No document selected. Use Search first."
            else:
                observation = f"Observation {step}: Unknown action '{action_type}'. Use Search, Lookup, or Finish."

        messages.append({"role": "user", "content": observation})
        trajectory.append(observation)

    # If we ran out of steps without Finish
    if not answer:
        answer = resp.split("Finish[")[-1].rstrip("]") if "Finish[" in (resp or "") else ""
        if not answer:
            # Last resort: extract any answer-like content
            answer = ""

    return answer, n_calls, "\n".join(trajectory)

# ── majority voting ──────────────────────────────────────────────
def majority_vote(answers):
    """Pick the most common normalized answer."""
    if not answers:
        return ""
    normed = [normalize_answer(a) for a in answers]
    counts = Counter(normed)
    best_norm = counts.most_common(1)[0][0]
    # Return the raw version of the best normalized answer
    for raw, n in zip(answers, normed):
        if n == best_norm:
            return raw
    return answers[0]

# ── per-query pipeline ───────────────────────────────────────────
def run_react_single(api_pool, query, passages, gold_answers, n_traj=N_TRAJ):
    telemetry = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    answers = []
    all_trajs = []
    total_calls = 0

    for t in range(n_traj):
        temp = 0.0 if t == 0 else 0.3
        ans, n_calls, traj = run_trajectory(api_pool, query, passages, temperature=temp)
        answers.append(ans)
        all_trajs.append(traj)
        total_calls += n_calls

    telemetry["calls"] = total_calls

    # Vote
    final_answer = majority_vote(answers) if n_traj > 1 else answers[0]
    norm = normalize_answer(final_answer)
    f1 = max_f1(norm, gold_answers)

    return {
        "query": query,
        "raw_answer": final_answer,
        "normalized_answer": norm,
        "gold_answers": gold_answers,
        "n_trajectories": n_traj,
        "trajectory_answers": answers,
        "total_calls": total_calls,
        "f1": f1,
        "telemetry": telemetry,
    }

# ── data loading ─────────────────────────────────────────────────
def load_data(workdir):
    workdir = Path(workdir)
    reranked_file = workdir / "reranked_gpt3.5.json"
    if not reranked_file.exists():
        sys.exit(f"Missing {reranked_file}")
    reranked = json.load(open(reranked_file))

    plan_file = workdir / "answers_plan.json"
    if not plan_file.exists():
        sys.exit(f"Missing {plan_file}")
    plan_data = json.load(open(plan_file))

    query_to_gold = {item["query"]: item.get("gold_answers", []) for item in plan_data}

    items = []
    for entry in reranked:
        q = entry.get("query", "")
        golds = query_to_gold.get(q, [])
        passages = []
        refs = entry.get("hits", entry.get("references", entry.get("reranked_passages", [])))
        if isinstance(refs, list):
            for r in refs:
                if isinstance(r, dict):
                    passages.append(r.get("content", r.get("text", r.get("passage", str(r))))[:800])
                elif isinstance(r, str):
                    passages.append(r[:800])
        items.append({"query": q, "passages": passages, "gold_answers": golds})
    return items


def metrics_path_for(workdir, output_name):
    output_path = Path(output_name)
    if output_path.name == "answers_react.json":
        return workdir / "metrics_react.json"
    return workdir / f"{output_path.stem}_metrics.json"

# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--n", type=int, default=0, help="Limit queries (0=all)")
    parser.add_argument("--n-traj", type=int, default=N_TRAJ)
    parser.add_argument("--output", default="answers_react.json")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    healthy_keys, probe_results = probe_working_keys(
        KEYS_FILE,
        base_url=BASE_URL,
        model=MODEL,
        max_workers=MAX_WORKERS,
    )
    bad_keys = [item for item in probe_results if not item.ok]
    print(f"[ReAct-GPT] keys={len(probe_results)} healthy={len(healthy_keys)} model={MODEL} n_traj={args.n_traj}")
    if bad_keys:
        masked = ", ".join(mask_key(item.key) for item in bad_keys)
        print(f"[ReAct-GPT] dropped bad keys: {masked}")
    if not healthy_keys:
        sys.exit("No healthy API keys available after validation.")

    items = load_data(workdir)
    if args.n > 0:
        items = items[:args.n]
    print(f"[ReAct-GPT] Loaded {len(items)} queries")

    api_pool = ChatKeyPool(
        healthy_keys,
        base_url=BASE_URL,
        model=MODEL,
        timeout=60,
        max_retries=MAX_RETRIES,
    )

    results = [None] * len(items)
    t0 = time.time()

    def process(idx, item):
        return idx, run_react_single(
            api_pool,
            item["query"],
            item["passages"],
            item["gold_answers"],
            n_traj=args.n_traj,
        )

    done = 0
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, api_pool.healthy_count))) as pool:
        futures = {pool.submit(process, i, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result
            result.pop("telemetry", None)
            done += 1
            if done % 50 == 0 or done == len(items):
                elapsed = time.time() - t0
                avg_f1 = sum(r["f1"] for r in results[:done] if r) / done
                print(f"  [{done}/{len(items)}] avg_f1={avg_f1:.4f} elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    f1s = [r["f1"] for r in results if r]
    avg_f1 = sum(f1s) / len(f1s)

    out_path = workdir / args.output
    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)

    metrics = {
        "method": "ReAct-GPT",
        "model": MODEL,
        "n_traj": args.n_traj,
        "count": len(results),
        "f1": avg_f1,
        "elapsed_sec": elapsed,
        "qps": len(results) / elapsed,
    }
    metrics_path = metrics_path_for(workdir, args.output)
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"\n[ReAct-GPT] F1={avg_f1:.4f} | {len(results)} queries | {elapsed:.0f}s")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {metrics_path}")

if __name__ == "__main__":
    main()
