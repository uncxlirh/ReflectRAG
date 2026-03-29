# nohup env PYTHONUNBUFFERED=1 TQDM_DISABLE=1 python -u 4-generate-plan.py > step4.log 2>&1 & echo $!
# tail -f step4.log  (Ctrl+C 结束查看)

import json
import time
import os
import re
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.telemetry_lite import StepTelemetry
_tel = StepTelemetry("4-generate-plan")

############################################################################
# 配置（安全化 + 稳健默认）
############################################################################
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://apic.littlewheat.com/v1")
MODEL = os.getenv("PLAN_MODEL", "gpt-3.5-turbo")
def _load_api_keys() -> List[str]:
    raw = os.getenv("OPENAI_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]

API_KEYS = _load_api_keys()

TEMPERATURE = 0.0
MAX_RETRIES = 4
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
TOP_K = 8                         # 统一 Top-K
MAX_CHARS_PER_PASSAGE = 600       # 上下文截断，减小跑偏
MAX_PLAN_SEGMENTS = 5             # plan 最多选几段
FALLBACK_TOP_N = 3                # plan 为空时回退段数
MAX_WORKERS = 15
DEBUG = False

INPUT_FILE = "reranked_gpt3.5.json"
OUTPUT_FILE = "answers_plan.jsonl"      # JSONL：实时写入，支持断点续传
OUTPUT_FILE_JSON = "answers_plan.json"  # 兼容：最终汇总成 JSON array

############################################################################
# 数据（保持现有来源）
############################################################################
dataset = load_dataset("ms_marco", "v2.1")
train_data = dataset["train"]

def get_gold_answers() -> Dict[str, List[str]]:
    return {str(sample["query_id"]): sample["answers"] for sample in train_data}

def load_reranked_data(path: str) -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            alt = os.path.splitext(path)[0] + ".jsonl"
            if os.path.exists(alt):
                path = alt
        if path.endswith(".jsonl"):
            out: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        continue
            return out
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 读取 {path} 失败：{e}")
        return []

def _load_done_qids_from_jsonl(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("qid")
                if qid is not None:
                    done.add(str(qid))
            except Exception:
                continue
    return done

def _jsonl_to_json_array(jsonl_path: str, json_path: str) -> int:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(jsonl_path):
        return 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    return len(items)

############################################################################
# 工具：稳健 LLM 解析 + 文本归一化
############################################################################
def to_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "choices") and getattr(resp, "choices"):
        ch0 = resp.choices[0]
        if hasattr(ch0, "message") and hasattr(ch0.message, "content"):
            return ch0.message.content
        if hasattr(ch0, "text"):
            return ch0.text
    if hasattr(resp, "output_text"):
        try:
            return resp.output_text
        except Exception:
            pass
    if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
        ch0 = resp["choices"][0]
        if isinstance(ch0, dict):
            if "message" in ch0 and isinstance(ch0["message"], dict) and "content" in ch0["message"]:
                return ch0["message"]["content"]
            if "text" in ch0:
                return ch0["text"]
    return str(resp)

def extract_json_array_or_obj(s: Any) -> Any:
    """
    兼容：dict/list 直接返回；严格 JSON；```json 围栏；轻微噪音。
    失败返回 {}。
    """
    import ast
    if isinstance(s, (dict, list)):
        return s
    if not s:
        return {}
    s = s.replace("\ufeff", "").replace("\u200b", "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # 尝试截取首个 JSON 片段
    start = s.find("{")
    if start == -1:
        start = s.find("[")
    if start != -1:
        for end in range(len(s), start, -1):
            chunk = s[start:end]
            try:
                return json.loads(chunk)
            except Exception:
                try:
                    return ast.literal_eval(chunk)
                except Exception:
                    pass
    return {}

def coerce_to_text(x: Any) -> str:
    """把任意值尽量压扁为字符串。"""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        for v in x:
            s = coerce_to_text(v)
            if s:
                return s
        return ""
    if isinstance(x, dict):
        for key in ("text", "answer", "value", "content", "normalized", "raw"):
            if key in x:
                s = coerce_to_text(x[key])
                if s:
                    return s
        for v in x.values():
            s = coerce_to_text(v)
            if s:
                return s
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)

def normalize_text(x: Any) -> str:
    s = coerce_to_text(x).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

############################################################################
# LLM 请求（多 key 轮询 + 指数退避）
############################################################################
def gpt_call(prompt: str, key_idx_start: int = 0) -> str:
    n_keys = len(API_KEYS)
    last_err = None
    for off in range(n_keys):  # 换 key
        k = API_KEYS[(key_idx_start + off) % n_keys]
        client = OpenAI(base_url=BASE_URL, api_key=k)
        for attempt in range(MAX_RETRIES):
            try:
                _t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    stream=False,
                    timeout=REQUEST_TIMEOUT
                )
                _tel.record_call(resp, model=MODEL, latency_s=time.perf_counter() - _t0)
                txt = to_text(resp)
                if not txt or not txt.strip():
                    raise ValueError("Empty content from model.")
                return txt
            except Exception as e:
                last_err = e
                if DEBUG:
                    print(f"[{k}][try {attempt+1}] {e}")
                # 配额/限流 → 换 key
                msg = str(e).lower()
                if "quota" in msg or "rate limit" in msg:
                    # 关键：加入轻微抖动，避免多个线程同一时间“集体换 key/集体 sleep”造成局部拥堵
                    time.sleep(0.2 + random.random() * 0.4)
                    break
                time.sleep(2 ** attempt)
    # 不抛出，让上层兜底
    return f'{{"error": "llm_failed", "detail": "{coerce_to_text(last_err)}"}}'

############################################################################
# Prompt（索引化 plan + 强约束作答）
############################################################################
def build_plan_prompt(query_text: str, doc_contents: List[str], hits: List[Dict[str, Any]]) -> str:
    context = "\n".join(
        [f"{i+1}. DocID={hit['docid']} | BM25Rank={hit['rank']} | Score={hit['score']}\n{content}\n---"
         for i, (content, hit) in enumerate(zip(doc_contents, hits))]
    )
    return f"""
You are a planner for open-domain QA. Select the most useful passages by their 1-based index.

Return ONLY a JSON object with BOTH fields:
- "plan_indices": an array of unique integers (1-based) in descending usefulness (at most {MAX_PLAN_SEGMENTS} items).
- "plan": an array of objects like {{"index": <1-based>, "reason": "<why this passage helps>"}}

Do NOT copy snippets. Do NOT answer. No extra text beyond JSON.

Query: {query_text}

Passages:
{context}

Output example:
{{
  "plan_indices": [3,1],
  "plan": [{{"index": 3, "reason": "directly states the required fact"}},
           {{"index": 1, "reason": "contains a supporting detail"}}]
}}
""".strip()

def build_answer_prompt(query_text: str, selected_ctx: str) -> str:
    return f"""
你是一个精准的问答模型。你只能使用下方 [SELECTED PASSAGES] 的事实作答；不得引用未给出的信息。


[SELECTED PASSAGES]
{selected_ctx}

问题：{query_text}

请严格以 JSON 格式作答，两个字段必须是字符串：
{{
  "raw_answer": "...",
  "normalized_answer": "..."
}}
""".strip()

############################################################################
# 计划解析（兼容旧 {"plan":[{"doc2":[...]}, ...]} 结构）
############################################################################
def indices_from_plan_obj(plan_obj: Any, top_k: int) -> List[int]:
    idx: List[int] = []
    if isinstance(plan_obj, dict) and isinstance(plan_obj.get("plan_indices"), list):
        for x in plan_obj["plan_indices"]:
            try:
                i = int(x)
                if 1 <= i <= top_k:
                    idx.append(i)
            except Exception:
                pass
    if not idx and isinstance(plan_obj, dict) and isinstance(plan_obj.get("plan"), list):
        for it in plan_obj["plan"]:
            if isinstance(it, dict):
                for k in it.keys():
                    m = re.match(r"doc(\d+)$", str(k).strip().lower())
                    if m:
                        i = int(m.group(1))
                        if 1 <= i <= top_k:
                            idx.append(i)
    # 去重 + 截断
    idx = sorted(set(idx))[:MAX_PLAN_SEGMENTS]
    return idx

def _validate_plan_json(obj, top_k=TOP_K):
    if not isinstance(obj, dict):
        return False
    pi = obj.get("plan_indices")
    pl = obj.get("plan")
    if not isinstance(pi, list) or not pi:
        return False
    if not isinstance(pl, list):
        return False
    for x in pi:
        if not isinstance(x, int) or not (1 <= x <= top_k):
            return False
    return True

def _validate_answer_json(obj):
    if not isinstance(obj, dict):
        return False
    ra = obj.get("raw_answer")
    na = obj.get("normalized_answer")
    return isinstance(ra, str) and isinstance(na, str)

def preflight_or_exit():
    print(
        f"[PREFLIGHT][4-generate-plan-new] base_url={BASE_URL} model={MODEL} timeout={REQUEST_TIMEOUT}s ..."
    )
    if not API_KEYS:
        print(
            "[PREFLIGHT][4-generate-plan-new] FAILED: OPENAI_API_KEYS is empty; set it to comma-separated keys.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        key_start = random.randrange(0, len(API_KEYS))

        # 1) connectivity + permissive plan probe
        plan_prompt = (
            "Return ONLY a valid JSON object (no code fences, no extra text).\n"
            "Schema:\n"
            '{ "plan_indices": [1,2], "plan": [{"index":1,"reason":"..."},{"index":2,"reason":"..."}] }\n'
            "Now output:"
        )
        obj1 = extract_json_array_or_obj(gpt_call(plan_prompt, key_idx_start=key_start))
        if not _validate_plan_json(obj1, top_k=TOP_K):
            print(
                f"[PREFLIGHT][4-generate-plan-new] WARN: plan schema mismatch during probe; continuing. probe={coerce_to_text(obj1)[:200]}",
                file=sys.stderr,
            )

        # 2) strict answer schema probe
        ans_prompt = (
            "Return ONLY valid JSON (no code fences, no extra text):\n"
            '{ "raw_answer": "paris", "normalized_answer": "paris" }'
        )
        obj2 = extract_json_array_or_obj(gpt_call(ans_prompt, key_idx_start=key_start))
        if not _validate_answer_json(obj2):
            raise ValueError(f"answer JSON schema mismatch: {coerce_to_text(obj2)[:200]}")

        print("[PREFLIGHT][4-generate-plan-new] OK")
    except Exception as e:
        print(f"[PREFLIGHT][4-generate-plan-new] FAILED: {e}", file=sys.stderr)
        sys.exit(1)

############################################################################
# 单条流程（内部自我兜底，保证不抛异常）
############################################################################
def process_item_safe(item: Dict[str, Any], gold_map: Dict[str, List[str]]) -> Dict[str, Any]:
    try:
        return process_item(item, gold_map)
    except Exception as e:
        # 永不抛出，返回“可评测”的占位结果
        query_text = item.get("query", "")
        qid = str(item["hits"][0]["qid"]) if item.get("hits") else "unknown"
        gold_answers = gold_map.get(qid, ["No gold answer available"])
        return {
            "query": query_text,
            "qid": qid,
            "plan": {"error": "internal_exception", "detail": coerce_to_text(e)},
            "plan_indices": [],
            "used_docids": [],
            "raw_answer": "Failed to generate answer.",
            "normalized_answer": normalize_text("Failed to generate answer."),
            "gold_answers": gold_answers,
            "references": []
        }

def process_item(item: Dict[str, Any], gold_map: Dict[str, List[str]]) -> Dict[str, Any]:
    query_text = item["query"]
    hits_all = sorted(item.get("hits", []), key=lambda h: h.get("rank", 1))[:TOP_K]
    qid = str(hits_all[0]["qid"]) if hits_all else "unknown"

    # 准备上下文
    doc_contents: List[str] = []
    for h in hits_all:
        c = (h.get("content", "") or "")
        c = c[:MAX_CHARS_PER_PASSAGE]
        doc_contents.append(c)

    # 为该样本分配“稳定的 key 起点”，避免所有线程都从 key 0 开始导致局部拥堵
    # 使用 qid hash：同一 qid 每次运行分配一致；不同样本分散到不同 key。
    key_start = 0
    if API_KEYS:
        key_start = (hash(qid) & 0xFFFFFFFF) % len(API_KEYS)

    # 1) 生成计划
    plan_prompt = build_plan_prompt(query_text, doc_contents, hits_all)
    plan_raw = gpt_call(plan_prompt, key_idx_start=key_start)
    plan_obj = extract_json_array_or_obj(plan_raw)
    plan_idx = indices_from_plan_obj(plan_obj, top_k=len(doc_contents))

    # 2) 选段（强约束 + 回退）
    if plan_idx:
        selected_ctx = "\n\n".join(doc_contents[i-1] for i in plan_idx)
        used_docids = [hits_all[i-1]["docid"] for i in plan_idx]
    else:
        selected_ctx = "\n\n".join(doc_contents[:FALLBACK_TOP_N])
        used_docids = [h["docid"] for h in hits_all[:FALLBACK_TOP_N]]

    # 3) 作答（严格 JSON 且值为字符串）
    answer_prompt = build_answer_prompt(query_text, selected_ctx)
    answer_raw = gpt_call(answer_prompt, key_idx_start=key_start)
    answer_obj = extract_json_array_or_obj(answer_raw)

    if isinstance(answer_obj, dict):
        raw_answer = coerce_to_text(answer_obj.get("raw_answer", ""))
        norm_answer = coerce_to_text(answer_obj.get("normalized_answer", raw_answer))
    else:
        raw_answer = coerce_to_text(answer_obj)
        norm_answer = raw_answer

    gold_answers = gold_map.get(qid, ["No gold answer available"])

    return {
        "query": query_text,
        "qid": qid,
        "plan": plan_obj,                       # 原样保存便于审计
        "plan_indices": plan_idx,               # 供覆盖率统计
        "used_docids": used_docids,             # 与 passages 对齐
        "raw_answer": coerce_to_text(raw_answer),
        "normalized_answer": normalize_text(norm_answer),
        "gold_answers": gold_answers,
        "references": doc_contents
    }

############################################################################
# 主程序（无异常打印；失败样本也会返回占位结果）
############################################################################
def main():
    preflight_or_exit()
    t0 = time.time()
    print("📦 加载数据...")
    reranked = load_reranked_data(INPUT_FILE)
    if not reranked:
        print("❌ 无数据，终止。"); return
    gold_map = get_gold_answers()

    done_qids = _load_done_qids_from_jsonl(OUTPUT_FILE)
    if done_qids:
        print(f"[RESUME][4-generate-plan-new] found {len(done_qids)} completed qids in {OUTPUT_FILE}")

    remaining = []
    for it in reranked:
        hits = it.get("hits", [])
        qid = str(it.get("qid") or (hits[0].get("qid") if hits else "unknown"))
        if qid not in done_qids:
            remaining.append(it)

    print(f"🚀 开始并发处理 {len(remaining)} 条查询（总计 {len(reranked)}）...")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_item_safe, it, gold_map) for it in remaining]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="🧠 生成中"):
            pass
        for fut in futs:
            try:
                result = fut.result()
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception:
                # 理论上不会走到这里
                out_f.write(json.dumps({
                    "query": "",
                    "qid": "unknown",
                    "plan": {"error": "future_exception"},
                    "plan_indices": [],
                    "used_docids": [],
                    "raw_answer": "Failed to generate answer.",
                    "normalized_answer": "failed to generate answer.",
                    "gold_answers": ["No gold answer available"],
                    "references": []
                }, ensure_ascii=False) + "\n")
                out_f.flush()

    out_f.close()
    total_written = _jsonl_to_json_array(OUTPUT_FILE, OUTPUT_FILE_JSON)

    _tel.save_summary()
    print(f"✅ 生成完成，共计 {total_written} 条（含历史），耗时 {time.time() - t0:.2f} 秒")
    print(f"📄 输出文件：{OUTPUT_FILE} (jsonl), {OUTPUT_FILE_JSON} (json array)")

if __name__ == "__main__":
    main()
