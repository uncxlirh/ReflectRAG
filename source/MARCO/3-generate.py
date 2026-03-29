#nohup env PYTHONUNBUFFERED=1 TQDM_DISABLE=1 python -u 3-generate.py > step3.log 2>&1 & echo $!
#tail -f step3.log
#ctrl + c


import json
import time
import os
import re
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.telemetry_lite import StepTelemetry
_tel = StepTelemetry("3-generate")

############################################################################
# 配置区域
############################################################################
# OpenAI 协议 Endpoint（默认对齐 test/test.py；可用环境变量覆盖）
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("PIPELINE_MODEL", "gpt-3.5-turbo")
TEMPERATURE = 0.0
MAX_RETRIES = 4
TOP_K = 10
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "15"))  # 并发线程总数

INPUT_FILE = "reranked_gpt3.5.json"
OUTPUT_FILE = "answers_base.jsonl"      # JSONL：实时写入，支持断点续传
OUTPUT_FILE_JSON = "answers_base.json"  # 兼容：最终汇总成 JSON array
DEBUG = False

############################################################################
# 数据加载
############################################################################

def _load_api_keys() -> list[str]:
    raw = os.getenv("OPENAI_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]

API_KEYS = _load_api_keys()

def _parse_answer_json_or_raise(txt: str):
    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("Answer JSON is not an object")
    if "raw_answer" not in obj or "normalized_answer" not in obj:
        raise ValueError("Answer JSON missing required keys")
    if not isinstance(obj["raw_answer"], str) or not isinstance(obj["normalized_answer"], str):
        raise ValueError("Answer JSON values must be strings")
    return obj

def preflight_or_exit():
    if not API_KEYS:
        print(
            "[PREFLIGHT][3-generate] FAILED: OPENAI_API_KEYS is empty; set it to comma-separated keys.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[PREFLIGHT][3-generate] base_url={BASE_URL} model={MODEL} ...")
    prompt = (
        "Return ONLY valid JSON (no code fences, no extra text):\n"
        "{\n"
        '  "raw_answer": "paris",\n'
        '  "normalized_answer": "paris"\n'
        "}"
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
        txt = resp.choices[0].message.content.strip()
        _parse_answer_json_or_raise(txt)
        print("[PREFLIGHT][3-generate] OK")
    except Exception as e:
        print(f"[PREFLIGHT][3-generate] FAILED: {e}", file=sys.stderr)
        sys.exit(1)

def get_gold_answers():
    dataset = load_dataset("ms_marco", "v2.1")["train"]
    return {str(sample["query_id"]): sample["answers"] for sample in dataset}

def load_reranked_data(json_file):
    try:
        if not os.path.exists(json_file):
            alt = os.path.splitext(json_file)[0] + ".jsonl"
            if os.path.exists(alt):
                json_file = alt
        if json_file.endswith(".jsonl"):
            items = []
            with open(json_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
            return items
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
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
    items = []
    if not os.path.exists(jsonl_path):
        return 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(items, fout, ensure_ascii=False, indent=4)
    return len(items)

############################################################################
# 调用 LLM 生成答案（带 Prompt 构造）
############################################################################

def build_prompt(query_text, doc_contents):
    context = "\n".join([f"Doc {i+1}: {content}" for i, content in enumerate(doc_contents)])
    prompt = f"""
You are an AI assistant for question answering and answer normalization.
Given the following query and context, provide both a concise and accurate initial answer and a normalized answer in JSON format.
The normalized answer should be a concise version (e.g., a number, short phrase, or key entity) that matches a reference answer, avoiding extra details.

Query: {query_text}
Context:
{context}

Response (JSON format):
{{
  "raw_answer": "...",
  "normalized_answer": "..."
}}
"""
    return prompt.strip()

def generate_answer_with_llm(query_text, doc_contents, key_idx_start: int = 0):
    prompt = build_prompt(query_text, doc_contents)

    n_keys = len(API_KEYS)
    last_err = None
    for off in range(n_keys):
        api_key = API_KEYS[(key_idx_start + off) % n_keys]
        client = OpenAI(base_url=BASE_URL, api_key=api_key)
        for attempt in range(MAX_RETRIES):
            try:
                _t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    stream=False,
                    timeout=30,
                )
                _tel.record_call(resp, model=MODEL, latency_s=time.perf_counter() - _t0)
                response_text = (resp.choices[0].message.content or "").strip()
                response_text = re.sub(r"^```json\n|\n```$", "", response_text)
                result = _parse_answer_json_or_raise(response_text)
                return result.get("raw_answer", ""), result.get("normalized_answer", "")
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # 配额/限流：直接换 key
                if "quota" in msg or "rate limit" in msg or "429" in msg:
                    break
                if DEBUG or attempt == 0:
                    print(
                        f"[LLM][3-generate][key {key_idx_start+off+1}/{n_keys}] "
                        f"[attempt {attempt+1}/{MAX_RETRIES}] {e}",
                        file=sys.stderr,
                    )
                time.sleep(2 ** attempt)
    return "Failed to generate answer.", "Failed to generate answer."

############################################################################
# 多线程封装
############################################################################

def ask_question(item, key_idx_start, gold_answer_map):
    query_text = item["query"]
    qid = str(item["hits"][0]["qid"]) if item.get("hits") else "unknown"
    doc_contents = [hit["content"] for hit in item.get("hits", [])[:TOP_K]]
    raw_answer, norm_answer = generate_answer_with_llm(query_text, doc_contents, key_idx_start=key_idx_start)
    gold_answers = gold_answer_map.get(qid, ["No gold answer available"])
    return {
        "query": query_text,
        "qid": qid,
        "raw_answer": raw_answer,
        "normalized_answer": norm_answer,
        "gold_answers": gold_answers,
        "references": doc_contents
    }

############################################################################
# 主逻辑：并发处理所有 query
############################################################################

def main():
    preflight_or_exit()
    print(f"[INFO][3-generate] keys={len(API_KEYS)} max_workers={MAX_WORKERS}")
    start_time = time.time()

    print("📦 加载数据中...")
    reranked_data = load_reranked_data(INPUT_FILE)
    gold_answer_map = get_gold_answers()
    if not reranked_data:
        print("❌ 无数据，终止运行")
        return

    print(f"🚀 准备并发处理 {len(reranked_data)} 条查询...")
    futures = []
    future_to_query = {}

    done_qids = _load_done_qids_from_jsonl(OUTPUT_FILE)
    if done_qids:
        print(f"[RESUME][3-generate] found {len(done_qids)} completed qids in {OUTPUT_FILE}")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, item in enumerate(reranked_data):
            qid = str(item.get("qid") or (item["hits"][0]["qid"] if item.get("hits") else "unknown"))
            if qid in done_qids:
                continue
            key_index = i % len(API_KEYS)
            future = executor.submit(ask_question, item, key_index, gold_answer_map)
            future_to_query[future] = i

        for future in tqdm(as_completed(future_to_query), total=len(future_to_query), desc="🧠 生成中"):
            try:
                result = future.result()
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"⚠️ 错误: {e}")

    out_f.close()
    total_written = _jsonl_to_json_array(OUTPUT_FILE, OUTPUT_FILE_JSON)

    _tel.save_summary()
    print(f"✅ 完成！共写入 {total_written} 条，耗时 {time.time() - start_time:.2f} 秒")
    print(f"📄 输出文件：{OUTPUT_FILE} (jsonl), {OUTPUT_FILE_JSON} (json array)")

if __name__ == "__main__":
    main()
