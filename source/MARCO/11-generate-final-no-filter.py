#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from bert_score import BERTScorer
from openai import OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.telemetry_lite import StepTelemetry

_tel = StepTelemetry("11-generate-final-no-filter")

INPUT_PLAN_FILE = os.getenv("INPUT_PLAN_FILE", "answers_plan.json")
INPUT_REFL_FILE = os.getenv("INPUT_REFL_FILE", "reflections_heuristics_off.json")
OUTPUT_ANS_FILE = os.getenv("OUTPUT_ANS_FILE", "answers_final_heuristics_off_fast.json")
OUTPUT_METRICS = os.getenv("OUTPUT_METRICS", "metrics_summary_heuristics_off_fast.json")

MODEL_GPT = os.getenv("MODEL_GPT", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "0"))
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://apic.littlewheat.com/v1")


def _load_api_keys() -> list[str]:
    raw = os.getenv("OPENAI_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


_API_KEYS = _load_api_keys()
_key_idx = 0


def _rotate():
    global _key_idx
    if _API_KEYS:
        _key_idx = (_key_idx + 1) % len(_API_KEYS)


def _key():
    return _API_KEYS[_key_idx] if _API_KEYS else None


def preflight_or_exit():
    print(f"[PREFLIGHT][11-no-filter] base_url={BASE_URL} model={MODEL_GPT} keys={len(_API_KEYS)}")
    if not _API_KEYS:
        print("[PREFLIGHT][11-no-filter] FAILED: OPENAI_API_KEYS is empty.", file=sys.stderr)
        sys.exit(1)


def simple_preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.split()


def f1_score(prediction, reference):
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


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_score(pred, ref):
    pt = " ".join(simple_preprocess(pred))
    rt = " ".join(simple_preprocess(ref))
    return _rouge.score(rt, pt)["rougeL"].fmeasure if pt and rt else 0.0


@lru_cache(maxsize=1)
def _bert_scorer():
    return BERTScorer(
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=False,
    )


YESNO_PREFIX = r"^(is|are|do|does|did|can|could|was|were|has|have|had|will|would|should|must|may|might)\b"


def is_yesno(q):
    return bool(re.match(YESNO_PREFIX, (q or "").strip().lower()))


def pick_passages(item, per_doc_chars=300):
    refs = item.get("references") or []
    plan = item.get("plan") or {}
    idxs = plan.get("plan_indices", []) or item.get("plan_indices", []) or []
    out = []
    if isinstance(idxs, list) and idxs:
        for v in idxs:
            try:
                i = int(v) - 1
                if 0 <= i < len(refs):
                    out.append((refs[i] or "")[:per_doc_chars])
            except Exception:
                pass
    if not out:
        out = [str(x)[:per_doc_chars] for x in refs[:3]]
    return out


def gpt_call(prompt):
    if not _API_KEYS:
        raise RuntimeError("OPENAI_API_KEYS is empty.")
    attempts, total = 0, len(_API_KEYS) * MAX_RETRIES
    while attempts < total:
        attempts += 1
        try:
            client = OpenAI(base_url=BASE_URL, api_key=_key())
            t0 = time.perf_counter()
            r = client.chat.completions.create(
                model=MODEL_GPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                stream=False,
                timeout=45,
            )
            _tel.record_call(r, model=MODEL_GPT, latency_s=time.perf_counter() - t0)
            return r.choices[0].message.content.strip()
        except Exception:
            _rotate()
            time.sleep(1.0)
    raise RuntimeError("All API keys failed.")


def strip_to_json(s):
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip("` \n")
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def normalize_yesno(query, raw, norm):
    if not is_yesno(query):
        return raw, norm
    txt = (norm or "").strip().lower()
    rtxt = (raw or "").strip().lower()
    if "yes" in txt and "no" not in txt:
        return "yes", "yes"
    if "no" in txt and "yes" not in txt:
        return "no", "no"
    if "yes" in rtxt and "no" not in rtxt:
        return "yes", "yes"
    if "no" in rtxt and "yes" not in rtxt:
        return "no", "no"
    return raw, norm


def build_final_prompt(query, plan, reflection, passages, enforce_yesno):
    pass_block = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    plan_json = json.dumps(plan or {}, ensure_ascii=False, indent=2)
    refl_json = json.dumps(reflection or {}, ensure_ascii=False, indent=2)
    yn_rule = '- If the question is yes/no, answer with EXACTLY "yes" or "no". Do not add any other words.\n' if enforce_yesno else ""
    return f"""
You are an assistant writing FINAL answers for a RAG QA system.
Use ONLY facts from [SELECTED PASSAGES].
Prefer short, direct answers (≤12 words). Avoid hallucination.

[Query]
{query}

[SELECTED PASSAGES]
{pass_block}

[Plan]
{plan_json}

[Reflection] (unfiltered)
{refl_json}

Constraints:
{yn_rule}- If not yes/no, give a short answer (≤12 words).
- If evidence is insufficient, output exactly: insufficient evidence.

Return STRICT JSON only:
{{
  "raw_answer": "...",
  "normalized_answer": "..."
}}
""".strip()


def process_one(i, plans, refls):
    item, ritem = plans[i], refls[i]
    q = item.get("query", "")
    plan = item.get("plan", {}) or {}
    passages = pick_passages(item, per_doc_chars=300)
    golds = item.get("gold_answers") or []
    gold = golds[0] if golds else ""
    reflection = ritem.get("reflection") or {"plan_analysis": "", "answer_analysis": "", "suggestions": []}
    prompt = build_final_prompt(q, plan, reflection, passages, enforce_yesno=is_yesno(q))
    gpt_out = gpt_call(prompt)
    parsed = strip_to_json(gpt_out) or {"raw_answer": gpt_out, "normalized_answer": gpt_out}
    raw = parsed.get("raw_answer", "")
    norm = parsed.get("normalized_answer", raw)
    raw, norm = normalize_yesno(q, raw, norm)

    f1 = f1_score(norm, gold)
    rg = rouge_l_score(norm, gold)
    try:
        _, _, f1_val = _bert_scorer().score([norm], [gold], batch_size=1)
        bs = float(f1_val.tolist()[0])
    except Exception:
        bs = 0.0

    return i, {
        "query": q,
        "plan": plan,
        "reflection": reflection,
        "final_raw_answer": raw,
        "final_normalized_answer": norm,
        "gold_answers": golds,
        "final_f1_score": f1,
        "final_rouge_l_score": rg,
        "final_bert_score": bs,
    }


def main():
    t0 = time.time()
    print(f"📥 Loading {INPUT_PLAN_FILE} & {INPUT_REFL_FILE} ...")
    preflight_or_exit()
    plans = json.load(open(INPUT_PLAN_FILE, "r", encoding="utf-8"))
    refls = json.load(open(INPUT_REFL_FILE, "r", encoding="utf-8"))
    n = min(len(plans), len(refls))
    if MAX_ITEMS > 0:
        n = min(n, MAX_ITEMS)

    results = [None] * n
    f1s, rgs, bss = [], [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_one, i, plans, refls): i for i in range(n)}
        for fut in tqdm(as_completed(futs), total=n, desc="🧠 Final answering (no filter)"):
            i, res = fut.result()
            results[i] = res
            f1s.append(res["final_f1_score"])
            rgs.append(res["final_rouge_l_score"])
            bss.append(res["final_bert_score"])

    json.dump(results, open(OUTPUT_ANS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    summary = {
        "HEURISTICS_OFF_FAST": {
            "F1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "ROUGE": round(sum(rgs) / len(rgs), 4) if rgs else 0.0,
            "BERT": round(sum(bss) / len(bss), 4) if bss else 0.0,
            "count": n,
        }
    }
    json.dump(summary, open(OUTPUT_METRICS, "w", encoding="utf-8"), indent=2)
    _tel.save_summary()
    print(f"✅ Saved answers to {OUTPUT_ANS_FILE}")
    print(f"✅ Saved metrics to {OUTPUT_METRICS}: {summary}")
    print(f"⏱️ Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
