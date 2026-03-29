# nohup python -u 7-prepare_reflection_data.py > 7_reflect.log 2>&1 &
# tail -f 7_reflect.log  (Ctrl+C 结束查看)
import json
import time
from tqdm import tqdm
from transformers import pipeline
from openai import OpenAI
import os
import sys
import torch
import numpy as np
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.telemetry_lite import StepTelemetry
_tel = StepTelemetry("7-prepare_reflection")

YESNO_PREFIX = r"^(is|are|do|does|did|can|could|was|were|has|have|had|will|would|should|must|may|might)\b"

def _is_yesno(q: str) -> bool:
    return bool(re.match(YESNO_PREFIX, (q or "").strip().lower()))

def _infer_yesno_from_passages(passages):
    """超轻权重启发式：证据里若出现明显否定，就判 no；若有明确存在/肯定且无否定，就判 yes；否则 None。"""
    t = " ".join([p.lower() for p in (passages or [])])
    neg = any(z in t for z in [" is not ", " are not ", " isn't ", " aren't ", " does not ", " didn't ", " no ", " not "])
    pos = any(z in t for z in [" is a ", " is the ", " are ", " there is ", " there are ", " exists ", " is ", " yes "])
    if neg: return "no"
    if pos: return "yes"
    return None

############################################################################
# 配置区域（保持你的原值）
############################################################################
def _load_api_keys() -> list[str]:
    raw = os.getenv("OPENAI_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]

API_KEYS = _load_api_keys()
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2
MAX_RETRIES = 4
BATCH_SIZE = 4
MAX_ITERATIONS = 2

INPUT_FILE = "RL_train.json"
OUTPUT_FILE_V1 = "reflection_data_v1.json"
OUTPUT_FILE_V2 = "reflection_data_v2.json"

# OpenAI 协议 Endpoint（默认对齐 test/test.py；可用环境变量覆盖）
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 稳定性设置（保持）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

############################################################################
# 通用函数（原样 + 更稳 JSON 解析）
############################################################################
def simple_preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.split()

def f1_score(prediction, reference):
    from collections import Counter
    pred_tokens = simple_preprocess(prediction)
    ref_tokens = simple_preprocess(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)

############################################################################
# GPT 多 Key 安全调用（保持）
############################################################################
def gpt_call(prompt, api_keys):
    for key in api_keys:
        client = OpenAI(base_url=BASE_URL, api_key=key)
        for attempt in range(MAX_RETRIES):
            try:
                _t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    stream=False,
                    timeout=30
                )
                _tel.record_call(resp, model=MODEL, latency_s=time.perf_counter() - _t0)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[{key}][Retry {attempt+1}] LLM error: {str(e)}")
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    break
                time.sleep(2 ** attempt)
    return "[Error] All API keys failed or exhausted."

def safe_parse_json_response(raw_text):
    """更稳：兼容 ```json 围栏 + 抓首个 JSON 块"""
    import re
    s = (raw_text or "").strip()
    try:
        s0 = re.sub(r'^```(?:json)?\s*', '', s, flags=re.I).rstrip('`').strip()
        return json.loads(s0)
    except Exception:
        try:
            m = re.search(r'(\{.*\}|\[.*\])', s, re.S)
            if m:
                return json.loads(m.group(1))
        except Exception:
            return None
    return None

def parse_or_retry_gpt_output(gpt_response, max_attempts=2):
    for _ in range(max_attempts):
        parsed = safe_parse_json_response(gpt_response)
        if parsed is not None and "Plan" in parsed and "Answer" in parsed:
            return parsed["Plan"], parsed["Answer"]
    return "N/A", "Failed to generate answer."

def preflight_or_exit():
    print(f"[PREFLIGHT][7-prepare_reflection_data] base_url={BASE_URL} model={MODEL} ...")
    if not API_KEYS:
        print(
            "[PREFLIGHT][7-prepare_reflection_data] FAILED: OPENAI_API_KEYS is empty; set it to comma-separated keys.",
            file=sys.stderr,
        )
        sys.exit(1)
    prompt = (
        "Return ONLY valid JSON (no code fences, no extra text).\n"
        "Schema:\n"
        '{ "Plan": {"plan": [], "instruction": "..."}, "Answer": "insufficient evidence" }'
    )
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEYS[0])
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            stream=False,
            timeout=30,
        )
        txt = (resp.choices[0].message.content or "").strip()
        parsed = safe_parse_json_response(txt)
        if not isinstance(parsed, dict) or "Plan" not in parsed or "Answer" not in parsed:
            raise ValueError(f"JSON schema mismatch: {txt[:200]}")
        print("[PREFLIGHT][7-prepare_reflection_data] OK")
    except Exception as e:
        print(f"[PREFLIGHT][7-prepare_reflection_data] FAILED: {e}", file=sys.stderr)
        sys.exit(1)

############################################################################
# 兼容 RL_train.json 的 plan 索引 + 选段证据（必备）
############################################################################
def get_plan_indices(item):
    """兼容 plan.plan_indices / 顶层 plan_indices / plan.plan[{index}] → 统一成 1-based int 列表"""
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
    """
    优先用 6 号产物中的 item['evidence']；
    否则按 plan 索引从 references 里取原文；做轻截断，避免上下文爆长。
    """
    # 1) 优先 evidence（已裁剪的轻量证据）
    ev = item.get("evidence") or []
    if isinstance(ev, list) and ev:
        return [str(x)[:per_doc_chars] for x in ev if isinstance(x, str)]
    # 2) 回退到 references + plan_indices
    refs = item.get("references") or []
    idxs = get_plan_indices(item)
    if not refs or not idxs:
        return []
    out = []
    for i in idxs:
        j = i - 1
        if 0 <= j < len(refs):
            out.append((refs[j] or "")[:per_doc_chars])
    return out

############################################################################
# LLM + Plan处理
############################################################################
# 设备自适应 + 更稳反思（必备）
generator = pipeline(
    "text-generation",
    model="gemma-2-2b-lora-init",
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_reflection_batch(prompts):
    try:
        # 更稳：不采样，输出更短
        outputs = generator(prompts, max_new_tokens=60, do_sample=False, batch_size=BATCH_SIZE)
        reflections = [output[0]["generated_text"][len(prompt):].strip() for prompt, output in zip(prompts, outputs)]
        torch.cuda.empty_cache()
        return reflections
    except Exception as e:
        print(f"⚠️ Gemma 生成异常：{str(e)}")
        return ["[Error] Gemma failed" for _ in prompts]

def _max_f1_over_golds(ans, golds):
    if not golds: 
        return 0.0
    return max(f1_score(ans, g) for g in golds)


def process_item(item, api_keys, iteration, prev_answer=None, prev_plan=None):
    current_answer = prev_answer or item.get('normalized_answer', '')
    current_plan = prev_plan or item.get('plan', {})
    gold_answer = item['gold_answers'][0] if item.get('gold_answers') else ""
    current_score = f1_score(current_answer, gold_answer)

    selected_passages = pick_selected_passages(item, per_doc_chars=300)
    pass_block = "\n".join(f"- {t}" for t in selected_passages) if selected_passages else "(no passages)"

    # 反思提示：加入选段证据
    reflection_prompt = (
        "[Query]\n"
        f"{item.get('query','')}\n\n"
        "[SELECTED PASSAGES]\n"
        f"{pass_block}\n\n"
        "[Plan]\n"
        f"{json.dumps(current_plan, indent=2, ensure_ascii=False)}\n\n"
        "[Current Answer]\n"
        f"{current_answer}\n\n"
        "[Gold Answer]\n"
        f"{gold_answer}\n\n"
        "[Score]\n"
        f"{current_score}\n\n"
        "[Task]\n"
        "You are an assistant that improves the Plan and Answer.\n"
        "Ground your reflection ONLY on [SELECTED PASSAGES]. Be concise and actionable.\n\n"
        "Output JSON:\n"
        "{\n"
        "  \"plan_analysis\": \"...\",\n"
        "  \"answer_analysis\": \"...\",\n"
        "  \"suggestions\": [\"...\", \"...\"]\n"
        "}"
    )

    # 生成新 Plan/Answer：同样喂证据 + 强答案约束（≤6词 / yes-no / 尽量抽取式）
    yesno_rule = "- If the question is yes/no, output exactly \"yes\" or \"no\".\n" if _is_yesno(item.get('query','')) else ""
    gpt_prompt_template = (
        "You refine the plan and answer. Use ONLY facts from [SELECTED PASSAGES].\n"
        "Return valid JSON with keys Plan and Answer. No extra text.\n\n"
        f"[Query]\n{item.get('query','')}\n\n"
        f"[SELECTED PASSAGES]\n{pass_block}\n\n"
        f"[Previous Plan]\n{json.dumps(current_plan, indent=2, ensure_ascii=False)}\n\n"
        f"[Previous Answer]\n{current_answer}\n\n"
        "[Reflection]\n"
        "{{REFLECTION_OUTPUT}}\n\n"
        "Constraints:\n"
        "- Base the new Answer only on [SELECTED PASSAGES].\n"
        f"{yesno_rule}"  # 原有这一行保留
        "- If NOT a yes/no question, the Answer MUST be a VERBATIM substring from [SELECTED PASSAGES],"
        " lowercased, with at most 6 words, and no punctuation.\n"
        "- If such a substring cannot be found, output exactly: insufficient evidence\n"
        "- JSON shape:\n"
        "{\n"
        '  "Plan": { "plan": [ {"document1": ["...","..."]}, {"document2": ["...","..."]} ],'
        '            "instruction": "..." },\n'
        '  "Answer": "..." \n'
        "}"

    )

    return reflection_prompt, gpt_prompt_template, item

############################################################################
# 主循环入口
############################################################################
def main():
    preflight_or_exit()
    start = time.time()
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_iteration = 0
    results = []
    if os.path.exists(OUTPUT_FILE_V1):
        with open(OUTPUT_FILE_V1, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        if len(v1_data) == len(data):
            print(f"✅ 检测到完整的 {OUTPUT_FILE_V1}，跳过第一轮")
            results = v1_data
            start_iteration = 1
        else:
            print(f"⚠️ {OUTPUT_FILE_V1} 数据不完整，将重新跑第一轮")

    torch.cuda.empty_cache()
    print(f"🚀 处理 {len(data)} 条数据...")
    for iteration in range(start_iteration, MAX_ITERATIONS):
        print(f"🔄 第 {iteration+1} 轮反思优化")
        reflection_prompts, gpt_prompts, items = [], [], []

        for i, item_data in enumerate(data):
            prev_answer = results[i]["new_answer"] if iteration > 0 else None
            prev_plan = results[i]["new_plan"] if iteration > 0 else None
            r_prompt, gpt_template, updated_item = process_item(item_data, API_KEYS, iteration, prev_answer, prev_plan)
            reflection_prompts.append(r_prompt)
            gpt_prompts.append(gpt_template)
            items.append(updated_item)

        print("🧠 批量生成反思...")
        iteration_results = []
        for batch_start in tqdm(range(0, len(reflection_prompts), BATCH_SIZE), desc=f"Gemma 生成 (轮 {iteration+1})"):
            batch_prompts = reflection_prompts[batch_start:batch_start + BATCH_SIZE]
            reflections = generate_reflection_batch(batch_prompts)

            for j, reflection in enumerate(reflections):
                idx = batch_start + j

                # 若证据为空，直接沿用旧答案，reward=0（防止被空证据拖负）
                if "[SELECTED PASSAGES]\n(no passages)" in reflection_prompts[idx]:
                    prev_ans = items[idx].get("normalized_answer","") if iteration == 0 else results[idx]["new_answer"]
                    prev_plan = items[idx].get("plan", {}) if iteration == 0 else results[idx]["new_plan"]
                    iteration_results.append({
                        "reflection_prompt": reflection_prompts[idx],
                        "reflection_output": reflection,
                        "gpt_prompt": "(skipped: no passages)",
                        "gpt_response_raw": "(skipped)",
                        "new_plan": prev_plan,
                        "new_answer": prev_ans,
                        "reward": 0.0,
                        "gold_answers": items[idx].get("gold_answers", []),
                        "iteration": iteration + 1,
                        "qid": items[idx].get("qid",""),
                        "query": items[idx].get("query","")
                    })
                    continue

                # 必备修改：占位符用双大括号
                gpt_prompt = gpt_prompts[idx].replace("{{REFLECTION_OUTPUT}}", reflection)
                gpt_response = gpt_call(gpt_prompt, API_KEYS)
                new_plan, new_answer = parse_or_retry_gpt_output(gpt_response, max_attempts=2)
                if isinstance(new_answer, str) and new_answer.startswith("Failed to"):
                    new_plan = items[idx].get("plan", {}) if iteration == 0 else results[idx]["new_plan"]
                    new_answer = items[idx].get("normalized_answer","") if iteration == 0 else results[idx]["new_answer"]

                ###########################################
                # —— yes/no 后处理兜底（最小侵入）
                if _is_yesno(items[idx].get("query","")):
                    sel = pick_selected_passages(items[idx], per_doc_chars=300)
                    yn = _infer_yesno_from_passages(sel)
                    if yn:
                        new_answer = yn
                ###########################################
                golds = items[idx].get("gold_answers", []) or [""]
                prev_ans = items[idx].get("normalized_answer","") if iteration == 0 else results[idx]["new_answer"]
                new_score = _max_f1_over_golds(new_answer, golds)
                prev_score = _max_f1_over_golds(prev_ans, golds)
                reward = new_score - prev_score


                iteration_results.append({
                    "reflection_prompt": reflection_prompts[idx],
                    "reflection_output": reflection,
                    "gpt_prompt": gpt_prompt,
                    "gpt_response_raw": gpt_response,
                    "new_plan": new_plan,
                    "new_answer": new_answer,
                    "reward": reward,
                    "gold_answers": items[idx].get("gold_answers", []),
                    "iteration": iteration + 1,
                    "qid": items[idx].get("qid",""),
                    "query": items[idx].get("query","")
                })

        results = iteration_results
        output_file = OUTPUT_FILE_V1 if iteration == 0 else OUTPUT_FILE_V2
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        avg_reward = float(np.mean([r["reward"] for r in results])) if results else 0.0
        print(f"📈 第 {iteration+1} 轮平均 Reward: {avg_reward:.4f}")
        print(f"📄 保存到 {output_file}")

    _tel.save_summary()
    print(f"✅ 所有轮完成，耗时 {time.time() - start:.2f} 秒")

if __name__ == "__main__":
    main()
