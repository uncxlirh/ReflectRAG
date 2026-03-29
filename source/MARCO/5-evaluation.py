# -*- coding: utf-8 -*-
import json
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
from rouge_score import rouge_scorer
from bert_score import score as bert_score_batch
import logging
import re
from typing import Any, List

# 与原脚本一致：显示 BERTScore 的告警
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)

def _load_json_or_jsonl(path: str) -> List[Any]:
    """兼容 .json（array）与 .jsonl（每行一个对象）。"""
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".jsonl"
        if os.path.exists(alt):
            path = alt
    if path.endswith(".jsonl"):
        out: List[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

############################################################################
# 文本预处理（保持原样）
############################################################################
def simple_preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.split()

############################################################################
# 计算 F1 分数（保持原样）
############################################################################
def f1_score(prediction, reference):
    if isinstance(prediction, dict):
        prediction = prediction.get("normalized_answer", "")
    if isinstance(reference, dict):
        reference = reference.get("normalized_answer", "")
    if isinstance(prediction, list):
        prediction = prediction[0] if prediction else ""
    if isinstance(reference, list):
        reference = reference[0] if reference else ""

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
# 计算 ROUGE-L（保持原样）
############################################################################
rouge_scorer_instance = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l_score(prediction, reference):
    if isinstance(prediction, dict):
        prediction = prediction.get("normalized_answer", "")
    if isinstance(reference, dict):
        reference = reference.get("normalized_answer", "")
    if isinstance(prediction, list):
        prediction = prediction[0] if prediction else ""
    if isinstance(reference, list):
        reference = reference[0] if reference else ""

    pred_tokens = simple_preprocess(prediction)
    ref_tokens = simple_preprocess(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_text = " ".join(pred_tokens)
    ref_text = " ".join(ref_tokens)
    scores = rouge_scorer_instance.score(ref_text, pred_text)
    return scores["rougeL"].fmeasure

############################################################################
# 主评估逻辑（新增 emit_cases 控制是否输出好/坏样本）
############################################################################
def evaluate_file(INPUT_FILE, OUTPUT_FILE, RESULT_FILE,
                  BAD_CASES_FILE=None, GOOD_CASES_FILE=None,
                  tag="", emit_cases=True):
    if not os.path.exists(INPUT_FILE):
        alt = os.path.splitext(INPUT_FILE)[0] + ".jsonl"
        if not os.path.exists(alt):
            print(f"❌ 文件 {INPUT_FILE} 不存在，跳过 {tag}！")
            return
        INPUT_FILE = alt

    data = _load_json_or_jsonl(INPUT_FILE)

    results = []
    f1_raw_list, f1_norm_list = [], []
    rouge_raw_list, rouge_norm_list = [], []

    # 收集全部预测/参考对，用于后续批量 BERTScore
    pred_raw_list, gold_raw_list = [], []
    pred_norm_list, gold_norm_list = [], []

    for item in tqdm(data, desc=f"🔍 收集数据 ({tag})"):
        query = item["query"]
        raw_answer = item["raw_answer"]
        norm_answer = item["normalized_answer"]
        gold_answers = item["gold_answers"]

        best_f1_raw = max([f1_score(raw_answer, g) for g in gold_answers], default=0)
        best_f1_norm = max([f1_score(norm_answer, g) for g in gold_answers], default=0)
        best_rouge_raw = max([rouge_l_score(raw_answer, g) for g in gold_answers], default=0)
        best_rouge_norm = max([rouge_l_score(norm_answer, g) for g in gold_answers], default=0)

        f1_raw_list.append(best_f1_raw)
        f1_norm_list.append(best_f1_norm)
        rouge_raw_list.append(best_rouge_raw)
        rouge_norm_list.append(best_rouge_norm)

        # BERTScore - 只选第一个 gold（保持原实现）
        gold_ref = gold_answers[0] if gold_answers else ""

        # 与原实现一致：强制转字符串，避免崩溃
        def normalize_text(x):
            if isinstance(x, dict):
                for v in x.values():
                    if isinstance(v, str):
                        return v
                    elif isinstance(v, list) and v and isinstance(v[0], str):
                        return v[0]
                    elif isinstance(v, dict):
                        return normalize_text(v)
                return ""
            elif isinstance(x, list):
                if x and isinstance(x[0], str):
                    return x[0]
                elif x and isinstance(x[0], dict):
                    return normalize_text(x[0])
                return ""
            return str(x)

        pred_raw_list.append(normalize_text(raw_answer))
        gold_raw_list.append(normalize_text(gold_ref))
        pred_norm_list.append(normalize_text(norm_answer))
        gold_norm_list.append(normalize_text(gold_ref))

        results.append({
            "query": query,
            "raw_answer": raw_answer,
            "normalized_answer": norm_answer,
            "gold_answers": gold_answers,
            "f1_score": {"raw": best_f1_raw, "normalized": best_f1_norm},
            "rouge_l": {"raw": best_rouge_raw, "normalized": best_rouge_norm},
            "bert_score": {}
        })

    print(f"⚙️ 正在批量计算 BERTScore（raw 和 normalized）... [{tag}]")

    P_raw, R_raw, F1_raw = bert_score_batch(
        pred_raw_list, gold_raw_list, lang="en",
        model_type="roberta-large", rescale_with_baseline=False
    )
    P_norm, R_norm, F1_norm = bert_score_batch(
        pred_norm_list, gold_norm_list, lang="en",
        model_type="roberta-large", rescale_with_baseline=False
    )

    bert_raw_list = F1_raw.tolist()
    bert_norm_list = F1_norm.tolist()

    for i in range(len(results)):
        results[i]["bert_score"]["raw"] = bert_raw_list[i]
        results[i]["bert_score"]["normalized"] = bert_norm_list[i]

    avg_scores = {
        "f1_score": {
            "raw": float(np.mean(f1_raw_list)),
            "normalized": float(np.mean(f1_norm_list))
        },
        "rouge_l": {
            "raw": float(np.mean(rouge_raw_list)),
            "normalized": float(np.mean(rouge_norm_list))
        },
        "bert_score": {
            "raw": float(np.mean(bert_raw_list)),
            "normalized": float(np.mean(bert_norm_list))
        }
    }

    print(f"\n🔹 **整体评测结果 [{tag}]**")
    print(json.dumps(avg_scores, indent=4, ensure_ascii=False))

    # 保存详细评测与均值
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": results, "average_scores": avg_scores},
                  f, ensure_ascii=False, indent=4)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, ensure_ascii=False, indent=4)

    # 是否生成好/坏样本清单（PLAN 需要，BASE 不需要）
    if emit_cases and BAD_CASES_FILE and GOOD_CASES_FILE:
        bad_cases, good_cases = [], []
        for result in results:
            f1_raw = result["f1_score"]["raw"]
            rouge_raw = result["rouge_l"]["raw"]
            bert_raw = result["bert_score"]["raw"]
            f1_norm = result["f1_score"]["normalized"]
            rouge_norm = result["rouge_l"]["normalized"]
            bert_norm = result["bert_score"]["normalized"]

            if (f1_raw < 0.2 and rouge_raw < 0.25 and bert_raw < 0.85) or \
               (f1_norm < 0.2 and rouge_norm < 0.25 and bert_norm < 0.85):
                if not ((f1_raw == 1.0 and rouge_raw == 1.0) or (f1_norm == 1.0 and rouge_norm == 1.0)):
                    bad_cases.append(result)
                else:
                    good_cases.append(result)
            else:
                good_cases.append(result)

        with open(BAD_CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(bad_cases, f, ensure_ascii=False, indent=4)
        with open(GOOD_CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(good_cases, f, ensure_ascii=False, indent=4)

        print(f"\n✅ 评测完成 [{tag}]！结果已保存至 {OUTPUT_FILE} 和 {RESULT_FILE}")
        print(f"❗ 低分样本已保存至 {BAD_CASES_FILE}")
        print(f"❗ 高分样本已保存至 {GOOD_CASES_FILE}")
    else:
        print(f"\n✅ 评测完成 [{tag}]！结果已保存至 {OUTPUT_FILE} 和 {RESULT_FILE}")
        print(f"ℹ️ 按需求：{tag} 分支不生成 good/bad cases。")

############################################################################
# 启动执行：PLAN 产出好/坏样本，BASE 只出评测均值/明细
############################################################################
if __name__ == "__main__":
    plan_input = "answers_plan.json"
    base_input = "answers_base.json"

    # PLAN：需要 good/bad cases（供 7–12 使用）
    evaluate_file(
        INPUT_FILE=plan_input,
        OUTPUT_FILE="eval_plan.json",
        RESULT_FILE="result_plan.json",
        BAD_CASES_FILE="bad_cases_plan.json",
        GOOD_CASES_FILE="good_cases_plan.json",
        tag="PLAN",
        emit_cases=True
    )

    # BASE：不需要 good/bad cases（BASE 到 1–6 即止）
    evaluate_file(
        INPUT_FILE=base_input,
        OUTPUT_FILE="eval_base.json",
        RESULT_FILE="result_base.json",
        BAD_CASES_FILE=None,
        GOOD_CASES_FILE=None,
        tag="BASE",
        emit_cases=False
    )
