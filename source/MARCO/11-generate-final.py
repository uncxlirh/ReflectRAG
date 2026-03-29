#!/usr/bin/env python
# coding: utf-8
"""
11-generate-final.py  — BASE+PLAN+REFLECT（干净版）
- 读取 answers_plan.json + reflections_full.json
- 使用 plan 选段（无则回退 top-3）
- 对 reflection 做“无泄漏”质量筛选（不看 gold）
- Prompt 约束：短答；yes/no 只允许 "yes"/"no"；非 yes/no 可在证据不足时输出 exact "insufficient evidence"
- 并行调用；稳健解析；计算 F1/ROUGE-L/BERT；写出 answers_final.json + metrics_summary.json
"""

import json, os, re, time, statistics
import sys
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from rouge_score import rouge_scorer
from bert_score import BERTScorer
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.telemetry_lite import StepTelemetry
_tel = StepTelemetry("11-generate-final")

# ================== 配置 ================== #
INPUT_PLAN_FILE   = os.getenv("INPUT_PLAN_FILE", "answers_plan.json")
INPUT_REFL_FILE   = os.getenv("INPUT_REFL_FILE", "reflections_full.json")
OUTPUT_ANS_FILE   = os.getenv("OUTPUT_ANS_FILE", "answers_final.json")
OUTPUT_METRICS    = os.getenv("OUTPUT_METRICS", "metrics_summary.json")

MODEL_GPT   = os.getenv("MODEL_GPT", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "0"))

# OpenAI 协议 Endpoint（默认对齐 test/test.py；可用环境变量覆盖）
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://apic.littlewheat.com/v1")

# 多 Key：环境变量 OPENAI_API_KEYS="k1,k2,k3"
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

# ================== 文本/指标工具 ================== #
def simple_preprocess(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.split()

def f1_score(prediction, reference):
    pt = simple_preprocess(prediction); rt = simple_preprocess(reference)
    if not pt or not rt: return 0.0
    common = Counter(pt) & Counter(rt); num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pt); recall = num_same / len(rt)
    return (2*precision*recall)/(precision+recall)

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
def rouge_l_score(pred, ref):
    pt, rt = " ".join(simple_preprocess(pred)), " ".join(simple_preprocess(ref))
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

def bucket_answer_len(s):
    n = len(simple_preprocess(s))
    if n <= 3: return "s"
    if n <= 8: return "m"
    return "l"

# ================== 选段：基于 plan_indices，回退 top-3 ================== #
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

# ================== OpenAI Chat ================== #
def gpt_call(prompt):
    if not _API_KEYS:
        return "{}"
    attempts, total = 0, len(_API_KEYS) * MAX_RETRIES
    while attempts < total:
        attempts += 1
        try:
            client = OpenAI(base_url=BASE_URL, api_key=_key())
            _t0 = time.perf_counter()
            r = client.chat.completions.create(
                model=MODEL_GPT,
                messages=[{"role":"user","content":prompt}],
                temperature=TEMPERATURE,
                stream=False,
                timeout=45
            )
            _tel.record_call(r, model=MODEL_GPT, latency_s=time.perf_counter() - _t0)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Key Failure] {_key()[:8] if _key() else 'NOKEY'} => {e}")
            _rotate(); time.sleep(1.0)
    return "{}"

def strip_to_json(s):
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip("` \n")
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return None
        return None

def _validate_final_answer_json(obj):
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("raw_answer"), str)
        and isinstance(obj.get("normalized_answer"), str)
    )

def preflight_or_exit():
    print(f"[PREFLIGHT][11-generate-final] base_url={BASE_URL} model={MODEL_GPT} ...")
    prompt = (
        "Return STRICT JSON only (no code fences, no extra text):\n"
        '{ "raw_answer": "paris", "normalized_answer": "paris" }'
    )
    last_error = None
    if not _API_KEYS:
        print("[PREFLIGHT][11-generate-final] FAILED: OPENAI_API_KEYS is empty.", file=sys.stderr)
        sys.exit(1)
    for key in _API_KEYS:
        try:
            client = OpenAI(base_url=BASE_URL, api_key=key)
            resp = client.chat.completions.create(
                model=MODEL_GPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                stream=False,
                timeout=30,
            )
            txt = (resp.choices[0].message.content or "").strip()
            obj = strip_to_json(txt)
            if not _validate_final_answer_json(obj):
                raise ValueError(f"JSON schema mismatch: {txt[:200]}")
            print("[PREFLIGHT][11-generate-final] OK")
            return
        except Exception as e:
            last_error = e
            continue
    print(f"[PREFLIGHT][11-generate-final] FAILED: {last_error}", file=sys.stderr)
    sys.exit(1)

# ================== Reflection 无泄漏预处理 ================== #
def reflection_stats(reflection):
    if not isinstance(reflection, dict): return (0, 0, False)
    suggs = reflection.get("suggestions") or []
    if not isinstance(suggs, list): return (0, 0, False)
    lens = [len(s.strip()) for s in suggs if isinstance(s, str)]
    count = len(lens)
    avg_len = statistics.mean(lens) if lens else 0
    kws = ("improve","revise","correct","fix","refine","specify","include","focus")
    has_kw = any(any(k in (s or "").lower() for k in kws) for s in (suggs or []) if isinstance(s, str))
    return count, avg_len, has_kw

def filter_reflection(query, reflection):
    """
    无泄漏（不看 gold/答案）的反思筛选规则：
    - yes/no 问题 → 直接清空（经验上反思常误导）
    - wh/define 开头问题 → 清空（减少长篇发挥）
    - 其它问题：仅保留“2 条建议 & 平均长度≥80 & 含改进关键词”的反思；否则清空
    """
    q = (query or "").strip().lower()
    sc, sl, has_kw = reflection_stats(reflection)
    if is_yesno(query):
        return {"plan_analysis":"","answer_analysis":"","suggestions":[]}
    if q.startswith(("what","who","where","when","which","why","how","define","definition","explain","meaning")):
        return {"plan_analysis":"","answer_analysis":"","suggestions":[]}
    if sc == 2 and sl >= 80 and has_kw:
        return reflection
    return {"plan_analysis":"","answer_analysis":"","suggestions":[]}

# ================== Prompt ================== #
def build_final_prompt(query, plan, reflection, passages, enforce_yesno):
    pass_block = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    plan_json = json.dumps(plan or {}, ensure_ascii=False, indent=2)
    refl_json = json.dumps(reflection or {}, ensure_ascii=False, indent=2)
    yn_rule = ""
    if enforce_yesno:
        yn_rule = '- If the question is yes/no, answer with EXACTLY "yes" or "no". Do not add any other words.\n'
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

[Reflection] (filtered)
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

# ================== Yes/No 归一化（不看 gold） ================== #
def normalize_yesno(query, raw, norm):
    if not is_yesno(query): 
        return raw, norm
    txt, rtxt = (norm or "").strip().lower(), (raw or "").strip().lower()
    if "yes" in txt and "no" not in txt: return "yes","yes"
    if "no"  in txt and "yes" not in txt: return "no","no"
    if "yes" in rtxt and "no"  not in rtxt: return "yes","yes"
    if "no"  in rtxt and "yes" not in rtxt: return "no","no"
    # 否则不做强制覆盖，交给原输出（保持独立性/公正性）
    return raw, norm

# ================== 单样本处理 ================== #
def process_one(i, plans, refls):
    item, ritem = plans[i], refls[i]
    q = item.get("query","")
    plan = item.get("plan",{}) or {}
    passages = pick_passages(item, per_doc_chars=300)
    golds = (item.get("gold_answers") or [])
    gold  = golds[0] if golds else ""

    reflection = ritem.get("reflection") or {"plan_analysis":"","answer_analysis":"","suggestions":[]}
    reflection = filter_reflection(q, reflection)

    prompt = build_final_prompt(q, plan, reflection, passages, enforce_yesno=is_yesno(q))
    gpt_out = gpt_call(prompt)
    parsed = strip_to_json(gpt_out) or {"raw_answer": gpt_out, "normalized_answer": gpt_out}
    raw, norm = parsed.get("raw_answer",""), parsed.get("normalized_answer", parsed.get("raw_answer",""))

    # yes/no 仅做二值归一，不借助 gold
    raw, norm = normalize_yesno(q, raw, norm)

    # 评测（只在离线评估阶段使用 gold，不反哺生成逻辑）
    f1 = f1_score(norm, gold)
    rg = rouge_l_score(norm, gold)
    try:
        _,_,f1_val = _bert_scorer().score([norm], [gold], batch_size=1)
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
        "final_bert_score": bs
    }

# ================== 主流程 ================== #
def main():
    t0 = time.time()
    print(f"📥 Loading {INPUT_PLAN_FILE} & {INPUT_REFL_FILE} ...")
    plan_path = INPUT_PLAN_FILE
    if not os.path.exists(plan_path):
        alt = os.path.splitext(plan_path)[0] + ".jsonl"
        if os.path.exists(alt):
            plan_path = alt
    if plan_path.endswith(".jsonl"):
        plans = []
        for line in open(plan_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                plans.append(json.loads(line))
            except Exception:
                continue
    else:
        plans = json.load(open(plan_path, "r", encoding="utf-8"))
    refls = json.load(open(INPUT_REFL_FILE, "r", encoding="utf-8"))
    n = min(len(plans), len(refls))
    if MAX_ITEMS > 0:
        n = min(n, MAX_ITEMS)

    preflight_or_exit()

    results = [None]*n
    f1s, rgs, bss = [], [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_one, i, plans, refls): i for i in range(n)}
        for fut in tqdm(as_completed(futs), total=n, desc="🧠 Final answering"):
            i, res = fut.result()
            results[i] = res
            f1s.append(res["final_f1_score"]); rgs.append(res["final_rouge_l_score"]); bss.append(res["final_bert_score"])

    json.dump(results, open(OUTPUT_ANS_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    summary = {
        "BASE+PLAN+REFLECT_clean": {
            "F1":   round(sum(f1s)/len(f1s), 3) if f1s else 0.0,
            "ROUGE":round(sum(rgs)/len(rgs), 3) if rgs else 0.0,
            "BERT": round(sum(bss)/len(bss), 3) if bss else 0.0,
            "count": n
        }
    }
    json.dump(summary, open(OUTPUT_METRICS,"w",encoding="utf-8"), indent=2)
    print(f"✅ Saved answers to {OUTPUT_ANS_FILE}")
    print(f"✅ Saved metrics to {OUTPUT_METRICS}: {summary}")
    _tel.save_summary()
    print(f"⏱️ Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
