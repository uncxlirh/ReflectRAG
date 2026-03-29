#!/usr/bin/env python
# coding: utf-8

import json
import os
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


INPUT_FILE = os.getenv("INPUT_PLAN_FILE", "answers_plan.json")
OUTPUT_FILE = os.getenv("OUTPUT_REFL_FILE", "reflections_heuristics_off.json")
BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2-2b-it")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "gemma-2-2b-reflection-final")
USE_BASE_MODEL = os.getenv("USE_BASE_MODEL", "0").strip().lower() in ("1", "true", "yes")
MAX_NEW_TOKENS = int(os.getenv("REFLECTION_MAX_NEW_TOKENS", "128"))
DO_SAMPLE = os.getenv("REFLECTION_DO_SAMPLE", "1").strip().lower() in ("1", "true", "yes")
TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("REFLECTION_TOP_P", "0.9"))
BATCH_SIZE = int(os.getenv("REFLECTION_BATCH_SIZE", "8"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "0"))

torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"


def load_reflector():
    print(f"⚙️ Loading base: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if USE_BASE_MODEL or not os.path.exists(ADAPTER_DIR):
        print("⚠️ Using BASE model only (no PEFT adapter)")
        model = base
    else:
        print(f"🔩 Attaching PEFT adapter: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    gen = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
    return gen


def get_plan_indices(item):
    p = item.get("plan") or {}
    idxs = []
    if isinstance(p.get("plan_indices"), list):
        idxs = p["plan_indices"]
    elif isinstance(item.get("plan_indices"), list):
        idxs = item["plan_indices"]
    elif isinstance(p.get("plan"), list):
        idxs = [v.get("index") for v in p["plan"] if isinstance(v, dict) and "index" in v]
    out = []
    for v in idxs:
        try:
            out.append(int(v))
        except Exception:
            pass
    return out


def pick_selected_passages(item, per_doc_chars=300):
    ev = item.get("evidence") or []
    if isinstance(ev, list) and ev:
        return [str(x)[:per_doc_chars] for x in ev if isinstance(x, str)]
    refs = item.get("references") or []
    idxs = get_plan_indices(item)
    if not refs or not idxs:
        return [str(x)[:per_doc_chars] for x in refs[:3]]
    out = []
    for i in idxs:
        j = i - 1
        if 0 <= j < len(refs):
            out.append((refs[j] or "")[:per_doc_chars])
    return out


def build_reflection_prompt(query, plan, current_answer, passages):
    pass_block = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    return f"""
[Query]
{query}

[SELECTED PASSAGES]
{pass_block}

[Plan]
{json.dumps(plan, ensure_ascii=False, indent=2)}

[Current Answer]
{current_answer}

[Task]
You are a REFLECTION assistant for a RAG QA pipeline.
Analyze issues in the Plan and in the current Answer using ONLY the selected passages.
Propose concrete, actionable improvements.

Rules:
- Use ONLY facts implied by [SELECTED PASSAGES].
- DO NOT output the final answer. Produce REFLECTION ONLY.
- Output MUST be strict JSON with the exact keys: plan_analysis, answer_analysis, suggestions.
- No extra text before/after the JSON.

Valid JSON schema:
{{
  "plan_analysis": "...",
  "answer_analysis": "...",
  "suggestions": ["...", "..."]
}}
""".strip()


def strip_to_json(s: str):
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


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]


def main():
    t0 = time.time()
    print(f"📥 Loading {INPUT_FILE} ...")
    in_path = INPUT_FILE
    if not os.path.exists(in_path):
        alt = os.path.splitext(in_path)[0] + ".jsonl"
        if os.path.exists(alt):
            in_path = alt
    if in_path.endswith(".jsonl"):
        data = []
        for line in open(in_path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    else:
        data = json.load(open(in_path, "r", encoding="utf-8"))
    if MAX_ITEMS > 0:
        data = data[:MAX_ITEMS]

    gen = load_reflector()
    prompts = []
    meta = []
    for uid, item in enumerate(data):
        q = item.get("query", "")
        plan = item.get("plan", {}) or {}
        ans = item.get("normalized_answer", "")
        passages = pick_selected_passages(item, per_doc_chars=300)
        prompts.append(build_reflection_prompt(q, plan, ans, passages))
        meta.append((uid, q, plan))

    out = [None] * len(prompts)
    for start, batch_prompts in tqdm(list(batched(prompts, BATCH_SIZE)), desc="🧠 Batch reflections"):
        outputs = gen(
            batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            batch_size=BATCH_SIZE,
        )
        for j, prompt in enumerate(batch_prompts):
            uid, q, plan = meta[start + j]
            full = outputs[j][0]["generated_text"]
            refl_txt = full[len(prompt):].strip()
            parsed = strip_to_json(refl_txt) or {"plan_analysis": "", "answer_analysis": "", "suggestions": []}
            out[start + j] = {"uid": uid, "query": q, "plan": plan, "reflection": parsed}

    json.dump(out, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(out)} reflections to {OUTPUT_FILE}")
    print(f"⏱️ Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
