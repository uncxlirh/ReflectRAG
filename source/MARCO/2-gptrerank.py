from openai import OpenAI
import json
import time
from tqdm import tqdm
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.telemetry_lite import StepTelemetry
_tel = StepTelemetry("2-gptrerank")

# 禁用 GPU（可选，避免 TensorFlow 警告）
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# OpenAI 协议 Endpoint（默认对齐 test/test.py；可用环境变量覆盖）
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def _load_api_keys() -> list[str]:
    raw = os.getenv("OPENAI_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]

API_KEYS = _load_api_keys()

# GPT 模型参数
MODEL = os.getenv("PIPELINE_MODEL", "gpt-3.5-turbo")
TEMPERATURE = 0  # 确保一致性
MAX_RETRIES = 3  # 最大重试次数
_default_workers = 12
if API_KEYS:
    _default_workers = min(32, max(4, len(API_KEYS) * 2))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(_default_workers)))  # 并发数（按实际配额/机器调整）

# 文件路径
BM25_RESULTS_PATH = "bm25-msmarcoqa.txt"  # 你的 BM25 结果文件
OUTPUT_FILE = "reranked_gpt3.5.jsonl"  # JSONL 实时写入
OUTPUT_FILE_JSON = "reranked_gpt3.5.json"  # 兼容：最终汇总成 JSON array
FAILED_QUERIES_LOG = "failed_queries.log"  # 保存到当前目录

# Top-K Passages 数量限制
TOP_K = 10

# 加载 MS MARCO 数据集以获取查询文本
dataset = load_dataset("ms_marco", "v2.1")
train_data = dataset["train"]

def _validate_rerank_json(obj, top_k=TOP_K):
    if not isinstance(obj, list) or not obj:
        return False
    for r in obj:
        if not isinstance(r, dict):
            return False
        if "index" not in r or "score" not in r:
            return False
        if not isinstance(r["index"], int) or not (1 <= r["index"] <= top_k):
            return False
        if not isinstance(r["score"], (int, float)) or not (0 <= float(r["score"]) <= 1):
            return False
    return True

def preflight_or_exit():
    """
    在真正大规模调用前先做一次最小 JSON 预检：
    - 必须返回可解析 JSON
    - 必须满足脚本所需 schema
    """
    if not API_KEYS:
        print(
            "[PREFLIGHT][2-gptrerank] FAILED: OPENAI_API_KEYS is empty; set it to comma-separated keys.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[PREFLIGHT][2-gptrerank] base_url={BASE_URL} model={MODEL} ...")
    prompt = (
        "Return ONLY a valid JSON array (no code fences, no extra text).\n"
        "Schema: [{\"index\": <int 1..10>, \"score\": <float 0..1>}, ...]\n"
        "Output exactly 3 items.\n"
        "Now output:"
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
        obj = json.loads(txt)
        if not _validate_rerank_json(obj, top_k=TOP_K):
            raise ValueError(f"JSON schema mismatch: {txt[:200]}")
        print("[PREFLIGHT][2-gptrerank] OK")
    except Exception as e:
        print(f"[PREFLIGHT][2-gptrerank] FAILED: {e}", file=sys.stderr)
        sys.exit(1)

def get_query_map():
    """从 MS MARCO 数据集中构建 query_id 到 query 的映射"""
    query_map = {}
    for sample in train_data:
        query_map[str(sample["query_id"])] = sample["query"]
    return query_map

def parse_bm25_results(file_path):
    """解析 TREC 格式的 BM25 结果，转换为查询-文档对"""
    results = {}
    with open(file_path, "r") as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in results:
                results[qid] = {"query": "", "hits": []}
            results[qid]["hits"].append({"docid": docid, "rank": int(rank), "score": float(score)})
    return results

def fetch_passage_content(searcher, docid):
    """从索引中获取文档内容"""
    try:
        raw_content = searcher.doc(docid).raw()
        content = json.loads(raw_content)
        if "title" in content:
            return "Title: " + content["title"] + " Content: " + content["text"]
        return content["contents"]
    except Exception as e:
        print(f"Failed to fetch content for docid {docid}: {e}")
        return "Content unavailable"

def generate_prompt(query, passages, initial_hits):
    """构建 GPT 的 Prompt，包含所有内容并要求返回带置信度分数的 JSON"""
    # 将 BM25 的信息和文档内容一起传递
    input_passages = "\n".join([
        f"{i + 1}. DocID: {hit['docid']}, BM25 Rank: {hit['rank']}, BM25 Score: {hit['score']}, Content: {passage}"
        for i, (passage, hit) in enumerate(zip(passages, initial_hits))
    ])
    # 使用多行字符串，避免 f-string 解析问题
    prompt = (
        "Examples:\n"
        "- Query: What is the capital of France?\n"
        "  Passages:\n"
        "  1. DocID: 123, BM25 Rank: 1, BM25 Score: 13.78, Content: Paris is the capital of France.\n"
        "  2. DocID: 124, BM25 Rank: 2, BM25 Score: 13.41, Content: France is located in Europe.\n"
        "  3. DocID: 125, BM25 Rank: 3, BM25 Score: 12.40, Content: The Eiffel Tower is located in Paris.\n"
        "  Answer: [\n"
        '    {"index": 1, "score": 0.95},\n'
        '    {"index": 3, "score": 0.85},\n'
        '    {"index": 2, "score": 0.70}\n'
        "  ]\n"
        "\n"
        "- Query: Who wrote Hamlet?\n"
        "  Passages:\n"
        "  1. DocID: 200, BM25 Rank: 1, BM25 Score: 14.50, Content: Hamlet is a tragedy written by William Shakespeare.\n"
        "  2. DocID: 201, BM25 Rank: 2, BM25 Score: 13.20, Content: Shakespeare was a famous playwright.\n"
        "  3. DocID: 202, BM25 Rank: 3, BM25 Score: 12.10, Content: Many people consider Hamlet a masterpiece.\n"
        "  Answer: [\n"
        '    {"index": 1, "score": 0.98},\n'
        '    {"index": 2, "score": 0.90},\n'
        '    {"index": 3, "score": 0.75}\n'
        "  ]\n"
        "\n"
        f"Query: {query}\n"
        "Passages:\n"
        f"{input_passages}\n"
        "\n"
        "Instructions:\n"
        "- Rank the passages in descending order of relevance to the query, considering all provided information (DocID, BM25 Rank, BM25 Score, and Content).\n"
        "- Return a valid JSON array of objects, each containing 'index' (1-based) and 'score' (a confidence score between 0 and 1, with 2 decimal places).\n"
        "- Example output: [{\"index\": 3, \"score\": 0.95}, {\"index\": 1, \"score\": 0.85}, {\"index\": 2, \"score\": 0.60}]\n"
        "- Do not include any explanations or additional text beyond the JSON array."
    )
    return prompt

def _chat_call_json(prompt: str, key_idx_start: int = 0, retries: int = MAX_RETRIES):
    """多 key 安全调用：失败（尤其限流/配额）会换 key；必须返回可解析 JSON。"""
    n_keys = len(API_KEYS)
    last_err = None
    for off in range(n_keys):
        k = API_KEYS[(key_idx_start + off) % n_keys]
        client = OpenAI(base_url=BASE_URL, api_key=k)
        for attempt in range(retries):
            try:
                _t0 = time.perf_counter()
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are an AI assistant designed for passage ranking."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    stream=False,
                    timeout=30,
                )
                _tel.record_call(response, model=MODEL, latency_s=time.perf_counter() - _t0)
                generated_text = (response.choices[0].message.content or "").strip()
                return json.loads(generated_text)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # 配额/限流：直接换 key
                if "quota" in msg or "rate limit" in msg or "429" in msg:
                    break
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after key rotation: {last_err}")

def rerank_with_gpt(query, passages, qid, initial_hits, key_idx_start: int):
    """使用 GPT 对段落进行重新排序，并返回置信度分数"""
    prompt = generate_prompt(query, passages, initial_hits)
    try:
        ranked_results = _chat_call_json(prompt, key_idx_start=key_idx_start)
        # 验证返回结果有效性
        if not isinstance(ranked_results, list) or not all(
            isinstance(r, dict) and "index" in r and "score" in r and
            isinstance(r["index"], int) and 1 <= r["index"] <= len(passages) and
            isinstance(r["score"], (int, float)) and 0 <= r["score"] <= 1
            for r in ranked_results
        ):
            raise ValueError("Invalid ranking results")
        # 重新组织 hits
        hits = []
        for rank, result in enumerate(ranked_results, start=1):
            idx = result["index"] - 1  # 转换为 0-based 索引
            original_hit = initial_hits[idx]
            hits.append({
                "content": passages[idx],
                "qid": qid,
                "docid": original_hit["docid"],
                "rank": rank,
                "score": result["score"]  # 使用 GPT 提供的置信度分数
            })
        return hits
    except Exception as e:
        print(f"[ERROR][2-gptrerank] rerank failed. qid={qid} query={query} err={e}")
        with open(FAILED_QUERIES_LOG, "a") as log_file:
            log_file.write(f"Query: {query}\nError: {e}\n\n")
        return None

def save_results_to_json(results, output_file):
    """保存结果到 JSON 文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

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

def _jsonl_to_json_array(jsonl_path: str, json_path: str) -> None:
    items = []
    if not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    save_results_to_json(items, json_path)

def main():
    preflight_or_exit()
    print(f"[INFO][2-gptrerank] keys={len(API_KEYS)} max_workers={MAX_WORKERS}")
    # 初始化 Pyserini 搜索器，使用预构建索引
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    searcher_lock = threading.Lock()

    # 获取 query_id 到 query 的映射
    query_map = get_query_map()

    # 解析 BM25 结果
    bm25_results = parse_bm25_results(BM25_RESULTS_PATH)

    # 为每个查询填充 query 文本
    for qid in bm25_results:
        bm25_results[qid]["query"] = query_map.get(qid, "unknown")

    done_qids = _load_done_qids_from_jsonl(OUTPUT_FILE)
    if done_qids:
        print(f"[RESUME][2-gptrerank] found {len(done_qids)} completed qids in {OUTPUT_FILE}")

    items = [(qid, item) for (qid, item) in bm25_results.items() if str(qid) not in done_qids]
    print(f"[INFO][2-gptrerank] remaining={len(items)} total={len(bm25_results)}")

    def _fetch_passages_threadsafe(initial_hits):
        passages = []
        for hit in initial_hits:
            with searcher_lock:
                passages.append(fetch_passage_content(searcher, hit["docid"]))
        return passages

    def _work(i: int, qid: str, item: dict):
        query = item["query"]
        initial_hits = item["hits"]
        passages = _fetch_passages_threadsafe(initial_hits)
        key_idx_start = i % max(1, len(API_KEYS))
        ranked_hits = rerank_with_gpt(query, passages, qid, initial_hits, key_idx_start=key_idx_start)
        return i, qid, query, ranked_hits

    # JSONL 实时写入：避免 99% 崩溃丢失全部数据
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for i, (qid, item) in enumerate(items):
            futs.append(ex.submit(_work, i, qid, item))
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Re-ranking ({MODEL})"):
            i, qid, query, ranked_hits = fut.result()
            if ranked_hits:
                rec = {"qid": str(qid), "query": query, "hits": ranked_hits}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
    out_f.close()

    # 兼容：汇总成 JSON array，供旧脚本/人工查看
    _jsonl_to_json_array(OUTPUT_FILE, OUTPUT_FILE_JSON)
    _tel.save_summary()
    print(f"Results saved to {OUTPUT_FILE} (jsonl) and {OUTPUT_FILE_JSON} (json array)")

if __name__ == "__main__":
    main()  # 修复末尾语法
