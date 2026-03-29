from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from openai import OpenAI
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from reflectrag.common.metrics import bert_score_single as cached_bert_score_single


YESNO_PREFIX = r"^(is|are|do|does|did|can|could|was|were|has|have|had|will|would|should|must|may|might)\b"


@dataclass
class RuntimeArgs:
    dataset_name: str
    base_model: str
    adapter_dir: str
    model_gpt: str
    base_url: str
    iterations: int
    disable_filtering: bool
    disable_verifier: bool
    use_base_model: bool
    input_file: str
    output_ans_file: str
    output_metrics: str
    max_retries: int
    max_workers: int
    reflection_candidates: int
    answer_candidates: int
    max_items: int


CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "5"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=3, help="Max iterations (T)")
    p.add_argument("--disable-filtering", action="store_true", help="Disable early-stop and filtering")
    p.add_argument("--disable-verifier", action="store_true", help="Disable the verifier sub-step")
    p.add_argument("--use-base-model", action="store_true", help="Use base model without PEFT adapter")
    p.add_argument("--input-file", type=str, default="answers_plan.json")
    p.add_argument("--output-ans-file", type=str, default="answers_final.json")
    p.add_argument("--output-metrics", type=str, default="metrics_summary.json")
    p.add_argument("--reflection-candidates", type=int, default=2, help="Number of sampled reflection candidates")
    p.add_argument("--answer-candidates", type=int, default=2, help="Number of final-answer candidates to rank")
    p.add_argument("--max-items", type=int, default=0, help="Smoke-test cap on examples")
    return p


def parse_runtime_args(dataset_name: str, adapter_dir_default: str) -> RuntimeArgs:
    parsed = build_arg_parser().parse_args()
    return RuntimeArgs(
        dataset_name=dataset_name,
        base_model="google/gemma-2-2b-it",
        adapter_dir=os.getenv("ADAPTER_DIR", adapter_dir_default),
        model_gpt=os.getenv("MODEL_GPT", "gpt-3.5-turbo"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        iterations=parsed.iterations,
        disable_filtering=parsed.disable_filtering,
        disable_verifier=parsed.disable_verifier,
        use_base_model=parsed.use_base_model,
        input_file=parsed.input_file,
        output_ans_file=parsed.output_ans_file,
        output_metrics=parsed.output_metrics,
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        max_workers=int(os.getenv("MAX_WORKERS", "5")),
        reflection_candidates=max(1, parsed.reflection_candidates),
        answer_candidates=max(1, parsed.answer_candidates),
        max_items=max(0, parsed.max_items),
    )


def simple_preprocess(text: str) -> list[str]:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s]+", "", value)
    return value.split()


def f1_score(prediction: str, reference: str) -> float:
    pt = simple_preprocess(prediction)
    rt = simple_preprocess(reference)
    if not pt or not rt:
        return 0.0
    common = Counter(pt) & Counter(rt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(rt)
    return (2 * precision * recall) / (precision + recall)


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_score(prediction: str, reference: str) -> float:
    pt = " ".join(simple_preprocess(prediction))
    rt = " ".join(simple_preprocess(reference))
    return _ROUGE.score(rt, pt)["rougeL"].fmeasure if pt and rt else 0.0


def bert_score_single(prediction: str, reference: str) -> float:
    return cached_bert_score_single(prediction, reference)


def is_yesno(query: str) -> bool:
    return bool(re.match(YESNO_PREFIX, (query or "").strip().lower()))


def strip_to_json(text: str) -> dict[str, Any] | None:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?", "", value, flags=re.I).strip("` \n")
    try:
        return json.loads(value)
    except Exception:
        match = re.search(r"\{.*\}", value, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        lines = [ln.strip("-* \t") for ln in value.splitlines() if ln.strip()]
        if lines:
            suggestions = []
            for line in lines:
                low = line.lower()
                if low.startswith(
                    (
                        "suggestion",
                        "revise",
                        "improve",
                        "fix",
                        "use ",
                        "keep ",
                        "replace ",
                        "clarify ",
                        "extract ",
                        "mention ",
                        "check ",
                    )
                ):
                    suggestions.append(line.split(":", 1)[-1].strip())
            if not suggestions:
                suggestions = [ln for ln in lines if len(ln) >= 12][:3]
            return {
                "plan_analysis": lines[0] if lines else "",
                "answer_analysis": lines[1] if len(lines) > 1 else "",
                "suggestions": suggestions[:3],
                "verification_points": suggestions[:2],
            }
        return None


def load_api_keys() -> list[str]:
    raw = os.getenv("OPENAI_API_KEYS", "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_plan_indices(item: dict[str, Any]) -> list[int]:
    plan = item.get("plan") or {}
    idxs = []
    if isinstance(plan.get("plan_indices"), list):
        idxs = plan["plan_indices"]
    elif isinstance(item.get("plan_indices"), list):
        idxs = item["plan_indices"]
    elif isinstance(plan.get("plan"), list):
        idxs = [v.get("index") for v in plan["plan"] if isinstance(v, dict) and "index" in v]
    return [int(v) for v in idxs if str(v).isdigit()]


def pick_passages(item: dict[str, Any], per_doc_chars: int = 300) -> list[str]:
    ev = item.get("evidence") or []
    if isinstance(ev, list) and ev:
        return [str(x)[:per_doc_chars] for x in ev if isinstance(x, str)]
    refs = item.get("references") or []
    idxs = get_plan_indices(item)
    if not refs or not idxs:
        return [str(x)[:per_doc_chars] for x in refs[:3]]
    out: list[str] = []
    for i in idxs:
        j = i - 1
        if 0 <= j < len(refs):
            out.append((refs[j] or "")[:per_doc_chars])
    return out if out else [str(x)[:per_doc_chars] for x in refs[:3]]


def normalize_yesno_answer(raw: str, normalized: str) -> tuple[str, str]:
    raw_text = (raw or "").strip().lower()
    norm_text = (normalized or "").strip().lower()
    if "yes" in norm_text and "no" not in norm_text:
        return "yes", "yes"
    if "no" in norm_text and "yes" not in norm_text:
        return "no", "no"
    if "yes" in raw_text and "no" not in raw_text:
        return "yes", "yes"
    if "no" in raw_text and "yes" not in raw_text:
        return "no", "no"
    return raw, normalized


def normalize_short_answer(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \n\t.,;:!?\"'")


def is_concise_answer(text: str, max_words: int = 8) -> bool:
    return len(simple_preprocess(text)) <= max_words


def lexical_overlap_ratio(answer: str, evidence_items: list[str]) -> float:
    ans = set(simple_preprocess(answer))
    if not ans:
        return 0.0
    evidence = set()
    for item in evidence_items:
        evidence.update(simple_preprocess(item))
    if not evidence:
        return 0.0
    return len(ans & evidence) / max(1, len(ans))


def fact_coverage_count(answer: str, facts: list[str]) -> int:
    answer_tokens = set(simple_preprocess(answer))
    count = 0
    for fact in facts:
        fact_tokens = set(simple_preprocess(fact))
        if not fact_tokens:
            continue
        overlap = len(answer_tokens & fact_tokens) / max(1, len(fact_tokens))
        if overlap >= 0.4:
            count += 1
    return count


def reflection_quality(reflection: dict[str, Any]) -> tuple[int, float, bool]:
    if not isinstance(reflection, dict):
        return 0, 0.0, False
    suggestions = reflection.get("suggestions") or []
    if not isinstance(suggestions, list):
        return 0, 0.0, False
    lens = [len(s.strip()) for s in suggestions if isinstance(s, str) and s.strip()]
    count = len(lens)
    avg_len = statistics.mean(lens) if lens else 0.0
    actionable_kws = (
        "improve", "revise", "correct", "fix", "refine", "specify",
        "include", "focus", "add", "remove", "replace", "clarify",
        "update", "extract", "mention", "check",
    )
    has_kw = any(
        any(k in (s or "").lower() for k in actionable_kws)
        for s in suggestions if isinstance(s, str)
    )
    return count, avg_len, has_kw


def normalize_reflection(reflection: dict[str, Any] | None) -> dict[str, Any]:
    obj = reflection if isinstance(reflection, dict) else {}
    def clean_field(value: Any) -> str:
        text = str(value or "").strip().strip(",")
        text = re.sub(r'^"?[a-z_]+"?\s*:\s*', "", text, flags=re.I)
        text = text.strip(" \"'")
        if text in {"{", "}", "[", "]", '["', '"]'} or len(text) < 2:
            return ""
        return text
    out = {
        "plan_analysis": clean_field(obj.get("plan_analysis", "")),
        "answer_analysis": clean_field(obj.get("answer_analysis", "")),
        "suggestions": [clean_field(x) for x in (obj.get("suggestions") or []) if clean_field(x)],
        "verification_points": [clean_field(x) for x in (obj.get("verification_points") or []) if clean_field(x)],
    }
    if not out["verification_points"]:
        out["verification_points"] = out["suggestions"][:2]
    return out


def choose_best_reflection(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best = {"plan_analysis": "", "answer_analysis": "", "suggestions": [], "verification_points": []}
    best_score = -10**9
    for cand in candidates:
        sc, sl, has_kw = reflection_quality(cand)
        score = sc * 3 + min(len(cand.get("verification_points", [])), 2) * 2
        score += 2 if has_kw else 0
        score += min(sl, 120) / 120.0
        if score > best_score:
            best_score = score
            best = cand
    return best


def choose_best_reflection_bundle(
    bundles: list[tuple[dict[str, Any], dict[str, Any]]],
    current_answer: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_refl = {"plan_analysis": "", "answer_analysis": "", "suggestions": [], "verification_points": []}
    best_ver = {"supported_points": [], "unsupported_points": [], "verified_facts": []}
    best_score = -10**9
    for cand, ver in bundles:
        sugg_count, avg_len, has_kw = reflection_quality(cand)
        verified = len(ver.get("verified_facts") or [])
        supported = len(ver.get("supported_points") or [])
        unsupported = len(ver.get("unsupported_points") or [])
        delta = 1 if simple_preprocess(current_answer) != simple_preprocess(" ".join(ver.get("verified_facts") or [])) else 0
        score = verified * 5 + supported * 2 - unsupported * 2 + sugg_count * 2 + delta
        score += 2 if has_kw else 0
        score += min(avg_len, 120) / 120.0
        if score > best_score:
            best_score = score
            best_refl = cand
            best_ver = ver
    return best_refl, best_ver


def build_reflection_prompt(query: str, plan: dict[str, Any], current_answer: str, passages: list[str]) -> str:
    pass_block = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    return f"""
[Query]
{query}

[SELECTED PASSAGES]
{pass_block}

[Plan]
{json.dumps(plan or {{}}, ensure_ascii=False, indent=2)}

[Current Answer]
{current_answer}

[Task]
You are a reflection assistant for a RAG QA pipeline.
Analyze weaknesses in the current plan and answer using ONLY the selected passages.

Return strict JSON with the exact keys:
- plan_analysis
- answer_analysis
- suggestions
- verification_points

Rules:
- suggestions should be concrete and actionable
- verification_points should contain 1-2 short factual points that can be checked directly against the passages
- If the current answer already looks correct, concise, and well-supported, explain that briefly and keep suggestions empty.
- do not output the final answer
- start directly with {{

JSON:
{{
  "plan_analysis": "...",
  "answer_analysis": "...",
  "suggestions": ["...", "..."],
  "verification_points": ["...", "..."]
}}
""".strip()


def build_verifier_prompt(query: str, passages: list[str], verification_points: list[str], current_answer: str) -> str:
    pass_block = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    points = "\n".join(f"- {p}" for p in verification_points) if verification_points else "- none"
    return f"""
You are an evidence verifier for a RAG QA pipeline.
Use ONLY the selected passages.

[Query]
{query}

[SELECTED PASSAGES]
{pass_block}

[Current Answer]
{current_answer}

[Verification Points]
{points}

Return strict JSON:
{{
  "supported_points": ["..."],
  "unsupported_points": ["..."],
  "verified_facts": ["..."]
}}
""".strip()


def build_final_prompt(
    query: str,
    plan: dict[str, Any],
    reflection: dict[str, Any],
    verification: dict[str, Any],
    passages: list[str],
    current_answer: str,
) -> str:
    pass_block = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    yn_rule = '- If the question is yes/no, answer with EXACTLY "yes" or "no".\n' if is_yesno(query) else ""
    return f"""
You are an assistant writing FINAL answers for a RAG QA system.
Use ONLY facts from [SELECTED PASSAGES].
Prefer [VERIFIED FACTS] when available.

[Query]
{query}

[SELECTED PASSAGES]
{pass_block}

[Plan]
{json.dumps(plan or {{}}, ensure_ascii=False, indent=2)}

[Current Answer]
{current_answer}

[Reflection]
{json.dumps(reflection or {{}}, ensure_ascii=False, indent=2)}

[Verification]
{json.dumps(verification or {{}}, ensure_ascii=False, indent=2)}

Constraints:
{yn_rule}- Use verified facts as highest-priority evidence.
- Ignore unsupported points.
- Keep the final answer concise.
- If not yes/no, prefer a short extractive answer (ideally <= 8 words).
- If no verified fact or supporting evidence is available, output exactly: insufficient evidence.
- If the current answer is already supported and concise, keep it rather than expanding it.

Return STRICT JSON only:
{{
  "raw_answer": "...",
  "normalized_answer": "..."
}}
""".strip()


class GptCaller:
    def __init__(self, base_url: str, model: str, max_retries: int, telemetry: Any | None):
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.telemetry = telemetry
        self.keys = load_api_keys()
        self._key_idx = random.randrange(0, len(self.keys)) if self.keys else 0
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {k: 0.0 for k in self.keys}

    def _next_key(self) -> str:
        if not self.keys:
            raise RuntimeError("OPENAI_API_KEYS not set")
        with self._lock:
            now = time.time()
            n = len(self.keys)
            for _ in range(n):
                key = self.keys[self._key_idx % n]
                self._key_idx += 1
                if self._cooldown_until.get(key, 0.0) <= now:
                    return key
            soonest = min(self._cooldown_until.values()) if self._cooldown_until else now
        time.sleep(max(0.05, min(1.0, soonest - now)))
        with self._lock:
            key = self.keys[self._key_idx % len(self.keys)]
            self._key_idx += 1
        return key

    def _mark_key_cooldown(self, key: str, error: str) -> None:
        message = (error or "").lower()
        cooldown = 0.2
        if any(token in message for token in ("quota", "pre_consume_token_quota_failed", "insufficient_quota")):
            cooldown = 15.0
        elif any(token in message for token in ("rate limit", "429", "too many requests")):
            cooldown = 5.0
        elif any(token in message for token in ("timed out", "timeout", "connection error", "server disconnected", "remoteprotocolerror")):
            cooldown = 1.5
        cooldown += random.random() * 0.5
        with self._lock:
            self._cooldown_until[key] = max(self._cooldown_until.get(key, 0.0), time.time() + cooldown)

    def call_json(self, prompt: str) -> dict[str, Any] | None:
        attempts, total = 0, len(self.keys) * self.max_retries
        while attempts < total:
            key = self._next_key()
            attempts += 1
            t0 = time.perf_counter()
            try:
                client = OpenAI(base_url=self.base_url, api_key=key)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    stream=False,
                    timeout=45,
                )
                if self.telemetry:
                    self.telemetry.record_call(resp, model=self.model, latency_s=time.perf_counter() - t0)
                text = resp.choices[0].message.content.strip()
                return strip_to_json(text)
            except Exception as exc:
                self._mark_key_cooldown(key, str(exc))
                if self.telemetry and hasattr(self.telemetry, "record_failure"):
                    self.telemetry.record_failure(model=self.model, latency_s=time.perf_counter() - t0, error=str(exc))
                time.sleep(0.2 + random.random() * 0.4)
        return None


def load_reflector(base_model: str, adapter_dir: str, use_base_model: bool):
    print(f"⚙️ Loading base: {base_model}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not use_base_model and os.path.exists(adapter_dir):
        print(f"🔩 Attaching PEFT adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        print("⚠️ Using BASE model only (No PEFT adapter)")
        model = base
    model.eval()
    gen = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
    return {"pipeline": gen, "tokenizer": tok}


def run_reflection_candidates(gen: Any, prompt: str, n: int) -> list[dict[str, Any]]:
    pipe = gen["pipeline"]
    tokenizer = gen["tokenizer"]
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        rendered = prompt
    out = []
    for _ in range(n):
        result = pipe(
            rendered,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0]["generated_text"]
        raw = result[len(rendered):].strip()
        out.append(normalize_reflection(strip_to_json(raw)))
    return out


def should_skip_reflection(query: str, disable_filtering: bool) -> bool:
    return False if disable_filtering else is_yesno(query)


def should_early_stop(
    query: str,
    reflection: dict[str, Any],
    verification: dict[str, Any],
    prev_answer: str,
    new_answer: str,
    disable_filtering: bool,
) -> tuple[bool, str]:
    if disable_filtering:
        return False, ""
    suggestions = reflection.get("suggestions") or []
    verified = verification.get("verified_facts") or []
    unsupported = verification.get("unsupported_points") or []
    if not suggestions and not verified:
        return True, "empty_reflection_and_no_verified_facts"
    if not verified and unsupported:
        return True, "no_verified_facts"
    if prev_answer and new_answer and simple_preprocess(prev_answer) == simple_preprocess(new_answer):
        return True, "answer_unchanged"
    if verified and is_concise_answer(new_answer) and lexical_overlap_ratio(new_answer, verified) >= 0.8:
        return True, "supported_concise_answer"
    if is_yesno(query):
        return True, "yesno_shortcut"
    return False, ""


def should_keep_current_answer(
    current_answer: str,
    verification: dict[str, Any],
    reflection: dict[str, Any],
    disable_filtering: bool,
) -> tuple[bool, str]:
    if disable_filtering or not current_answer:
        return False, ""
    verified = verification.get("verified_facts") or []
    suggestions = reflection.get("suggestions") or []
    if verified and not suggestions and is_concise_answer(current_answer) and lexical_overlap_ratio(current_answer, verified) >= 0.6:
        return True, "current_answer_already_supported"
    return False, ""


def score_answer_candidate(
    candidate: str,
    current_answer: str,
    verification: dict[str, Any],
    passages: list[str],
    query: str,
) -> float:
    cand = normalize_short_answer(candidate)
    if not cand:
        return -10**6
    score = 0.0
    verified = verification.get("verified_facts") or []
    supported = verification.get("supported_points") or []
    unsupported = verification.get("unsupported_points") or []
    if verified:
        score += lexical_overlap_ratio(cand, verified) * 8
        score += fact_coverage_count(cand, verified) * 2.5
    score += lexical_overlap_ratio(cand, passages) * 3
    if cand.lower() == "insufficient evidence":
        score -= 3 if verified else 0
    if is_yesno(query):
        score += 2 if cand.lower() in ("yes", "no") else -2
    else:
        score += 0.75 if is_concise_answer(cand) else max(-1.5, -0.08 * max(0, len(simple_preprocess(cand)) - 8))
    if current_answer and simple_preprocess(cand) == simple_preprocess(current_answer):
        score += 1.5
    if supported:
        score += 1.0
    if unsupported:
        score -= 1.0
        for point in unsupported:
            if point and point.lower() in cand.lower():
                score -= 2
        if current_answer and simple_preprocess(cand) == simple_preprocess(current_answer) and not supported:
            score -= 4
    return score


def generate_final_answer_candidates(
    caller: GptCaller,
    final_prompt: str,
    current_raw: str,
    current_norm: str,
    verification: dict[str, Any],
    passages: list[str],
    query: str,
    n: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    base_current = current_norm or current_raw
    candidates.append(
        {
            "raw_answer": current_raw,
            "normalized_answer": current_norm or current_raw,
            "score": score_answer_candidate(base_current, base_current, verification, passages, query),
            "source": "current_answer",
        }
    )
    for idx in range(max(1, n)):
        final_obj = caller.call_json(final_prompt)
        if not isinstance(final_obj, dict):
            continue
        raw = normalize_short_answer(final_obj.get("raw_answer", current_raw))
        norm = normalize_short_answer(final_obj.get("normalized_answer", raw or current_norm or current_raw))
        score = score_answer_candidate(norm or raw, base_current, verification, passages, query)
        candidates.append(
            {
                "raw_answer": raw or current_raw,
                "normalized_answer": norm or raw or current_norm or current_raw,
                "score": score,
                "source": f"generated_{idx+1}",
            }
        )
    best = max(candidates, key=lambda x: x["score"])
    current = next((c for c in candidates if c["source"] == "current_answer"), best)
    if best["source"] != "current_answer" and best["score"] < current["score"] + 1.5:
        best = current
    return best["raw_answer"], best["normalized_answer"], candidates


def summarize_results(results: list[dict[str, Any]], args: RuntimeArgs) -> dict[str, Any]:
    f1s = [r["final_f1_score"] for r in results]
    rgs = [r["final_rouge_l_score"] for r in results]
    bss = [r["final_bert_score"] for r in results]
    iters = [r["iterations_used"] for r in results]
    stops = Counter(r.get("stop_reason", "") for r in results)
    iter_hist = Counter(iters)
    nonempty_reflections = sum(1 for r in results if (r.get("reflection_final") or {}).get("suggestions"))
    verified_nonempty = sum(1 for r in results if (r.get("verification_final") or {}).get("verified_facts"))
    improved = 0
    degraded = 0
    unchanged = 0
    for r in results:
        plan_answer = r.get("plan_answer", "")
        gold_answers = r.get("gold_answers", []) or []
        gold = gold_answers[0] if gold_answers else ""
        before = f1_score(plan_answer, gold) if plan_answer else 0.0
        after = r["final_f1_score"]
        if after > before:
            improved += 1
        elif after < before:
            degraded += 1
        else:
            unchanged += 1
    return {
        "ITERATIVE_REFLECT": {
            "max_T": args.iterations,
            "reflection_candidates": args.reflection_candidates,
            "answer_candidates": args.answer_candidates,
            "verifier_enabled": not args.disable_verifier,
            "filtering_enabled": not args.disable_filtering,
            "used_base_model": args.use_base_model,
            "avg_iterations_used": round(statistics.mean(iters), 2) if iters else 0,
            "iteration_histogram": {str(k): v for k, v in sorted(iter_hist.items())},
            "stop_reasons": dict(stops),
            "reflection_nonempty_rate": round(nonempty_reflections / max(1, len(results)), 4),
            "verification_nonempty_rate": round(verified_nonempty / max(1, len(results)), 4),
            "improved_count": improved,
            "degraded_count": degraded,
            "unchanged_count": unchanged,
            "F1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "ROUGE": round(sum(rgs) / len(rgs), 4) if rgs else 0.0,
            "BERT": round(sum(bss) / len(bss), 4) if bss else 0.0,
            "count": len(results),
        }
    }


def save_checkpoint(results: list[dict[str, Any]], runtime: RuntimeArgs, telemetry: Any | None = None) -> None:
    json.dump(results, open(runtime.output_ans_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    summary = summarize_results(results, runtime)
    json.dump(summary, open(runtime.output_metrics, "w", encoding="utf-8"), indent=2)
    if telemetry and hasattr(telemetry, "save_summary"):
        telemetry.save_summary()


def item_uid(item: dict[str, Any]) -> str:
    qid = item.get("qid")
    if qid is not None and str(qid).strip():
        return str(qid)
    return str(item.get("query", "")).strip()


def run_dataset(runtime: RuntimeArgs, telemetry: Any | None = None) -> int:
    if not load_api_keys():
        print("❌ OPENAI_API_KEYS not set")
        return 1
    in_path = runtime.input_file
    if not os.path.exists(in_path):
        alt = os.path.splitext(in_path)[0] + ".jsonl"
        if os.path.exists(alt):
            in_path = alt
    print(f"📥 Loading {in_path} ...")
    if in_path.endswith(".jsonl"):
        data = []
        for line in open(in_path, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    else:
        data = json.load(open(in_path, "r", encoding="utf-8"))
    if runtime.max_items > 0:
        data = data[: runtime.max_items]

    results: list[dict[str, Any]] = []
    if os.path.exists(runtime.output_ans_file):
        try:
            existing = json.load(open(runtime.output_ans_file, "r", encoding="utf-8"))
            if isinstance(existing, list):
                results = existing
                done_ids = {item_uid(item) for item in results if isinstance(item, dict)}
                if 0 < len(done_ids) < len(data):
                    print(f"♻️ Resuming from existing checkpoint: {len(done_ids)}/{len(data)} done")
                    data = [item for item in data if item_uid(item) not in done_ids]
                elif len(done_ids) >= len(data):
                    print(f"✅ Existing output already complete: {len(done_ids)}/{len(data)}")
                    save_checkpoint(results, runtime, telemetry)
                    return 0
        except Exception:
            results = []

    gen = load_reflector(runtime.base_model, runtime.adapter_dir, runtime.use_base_model)
    caller = GptCaller(runtime.base_url, runtime.model_gpt, runtime.max_retries, telemetry)
    t0 = time.time()
    try:
        start_done = len(results)
        for idx, item in enumerate(tqdm(data, desc=f"🧠 Iterative Answering ({runtime.dataset_name}, T={runtime.iterations})"), start=1):
            q = item.get("query", "")
            plan = item.get("plan", {}) or {}
            passages = pick_passages(item, per_doc_chars=300)
            golds = item.get("gold_answers", []) or []
            gold = golds[0] if golds else ""
            current_raw = item.get("raw_answer", "")
            current_norm = item.get("normalized_answer", "")
            current_reflection = {"plan_analysis": "", "answer_analysis": "", "suggestions": [], "verification_points": []}
            current_verification = {"supported_points": [], "unsupported_points": [], "verified_facts": []}
            trace = []
            iterations_used = 0
            stop_reason = ""

            if should_skip_reflection(q, runtime.disable_filtering):
                final_prompt = build_final_prompt(q, plan, current_reflection, current_verification, passages, current_norm)
                parsed = caller.call_json(final_prompt) or {}
                current_raw = parsed.get("raw_answer", current_raw)
                current_norm = parsed.get("normalized_answer", current_raw)
                current_raw, current_norm = normalize_yesno_answer(current_raw, current_norm)
                iterations_used = 0
                stop_reason = "yesno_shortcut"
            else:
                for t in range(runtime.iterations):
                    answer_before = current_norm
                    refl_prompt = build_reflection_prompt(q, plan, current_norm, passages)
                    candidates = run_reflection_candidates(gen, refl_prompt, runtime.reflection_candidates)
                    if not any(c.get("suggestions") or c.get("verification_points") for c in candidates):
                        candidates = [choose_best_reflection(candidates)]
                    bundles: list[tuple[dict[str, Any], dict[str, Any]]] = []
                    if runtime.disable_verifier:
                        bundles = [(cand, {"supported_points": [], "unsupported_points": [], "verified_facts": []}) for cand in candidates]
                    else:
                        for cand in candidates:
                            verifier_prompt = build_verifier_prompt(q, passages, cand.get("verification_points", []), current_norm)
                            verifier_obj = caller.call_json(verifier_prompt)
                            verification = {"supported_points": [], "unsupported_points": [], "verified_facts": []}
                            if isinstance(verifier_obj, dict):
                                verification = {
                                    "supported_points": [str(x).strip() for x in (verifier_obj.get("supported_points") or []) if str(x).strip()],
                                    "unsupported_points": [str(x).strip() for x in (verifier_obj.get("unsupported_points") or []) if str(x).strip()],
                                    "verified_facts": [str(x).strip() for x in (verifier_obj.get("verified_facts") or []) if str(x).strip()],
                                }
                            bundles.append((cand, verification))
                    chosen, verification = choose_best_reflection_bundle(bundles, current_norm)
                    keep_current, keep_reason = should_keep_current_answer(
                        current_norm, verification, chosen, runtime.disable_filtering
                    )
                    answer_candidates_trace = []
                    final_prompt = build_final_prompt(q, plan, chosen, verification, passages, current_norm)
                    if keep_current:
                        stop_reason = keep_reason
                    else:
                        current_raw, current_norm, answer_candidates_trace = generate_final_answer_candidates(
                            caller=caller,
                            final_prompt=final_prompt,
                            current_raw=current_raw,
                            current_norm=current_norm,
                            verification=verification,
                            passages=passages,
                            query=q,
                            n=runtime.answer_candidates,
                        )
                    if is_yesno(q):
                        current_raw, current_norm = normalize_yesno_answer(current_raw, current_norm)
                    iterations_used = t + 1
                    current_reflection = chosen
                    current_verification = verification
                    stop = False
                    if keep_current:
                        stop = True
                    else:
                        stop, stop_reason = should_early_stop(
                            q, chosen, verification, answer_before, current_norm, runtime.disable_filtering
                        )
                    trace.append(
                        {
                            "iteration": t + 1,
                            "answer_before": answer_before,
                            "reflection_candidates": candidates,
                            "chosen_reflection": chosen,
                            "verification": verification,
                            "answer_candidates": answer_candidates_trace,
                            "answer_after": current_norm,
                            "stop_reason": stop_reason if stop else "",
                        }
                    )
                    if stop:
                        break

            results.append(
                {
                    "qid": item.get("qid"),
                    "query": q,
                    "plan": plan,
                    "plan_answer": item.get("normalized_answer", ""),
                    "iterations_used": iterations_used,
                    "stop_reason": stop_reason,
                    "iteration_trace": trace,
                    "reflection_final": current_reflection,
                    "verification_final": current_verification,
                    "final_raw_answer": current_raw,
                    "final_normalized_answer": current_norm,
                    "gold_answers": golds,
                    "final_f1_score": f1_score(current_norm, gold),
                    "final_rouge_l_score": rouge_l_score(current_norm, gold),
                    "final_bert_score": bert_score_single(current_norm, gold),
                }
            )

            if (start_done + idx) % CHECKPOINT_EVERY == 0:
                save_checkpoint(results, runtime, telemetry)

    except KeyboardInterrupt:
        print("⚠️ Interrupted. Saving partial checkpoint ...")
        save_checkpoint(results, runtime, telemetry)
        raise
    except Exception:
        save_checkpoint(results, runtime, telemetry)
        raise

    save_checkpoint(results, runtime, telemetry)
    print(f"✅ Saved answers to {runtime.output_ans_file}")
    print(f"✅ Saved metrics to {runtime.output_metrics}: {json.dumps(summarize_results(results, runtime), indent=2)}")
    print(f"⏱️ Time: {time.time()-t0:.2f}s")
    return 0
