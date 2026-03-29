from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

from bert_score import BERTScorer
from rouge_score import rouge_scorer


YESNO_PREFIX = r"^(is|are|do|does|did|can|could|was|were|has|have|had|will|would|should|must|may|might)\b"


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


def max_f1_over_golds(prediction: str, golds: Iterable[str]) -> float:
    items = list(golds or [])
    if not items:
        return 0.0
    return max(f1_score(prediction, gold) for gold in items)


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_score(prediction: str, reference: str) -> float:
    pt = " ".join(simple_preprocess(prediction))
    rt = " ".join(simple_preprocess(reference))
    if not pt or not rt:
        return 0.0
    return _ROUGE.score(rt, pt)["rougeL"].fmeasure


def max_rouge_over_golds(prediction: str, golds: Iterable[str]) -> float:
    items = list(golds or [])
    if not items:
        return 0.0
    return max(rouge_l_score(prediction, gold) for gold in items)


@lru_cache(maxsize=1)
def _bert_scorer() -> BERTScorer:
    return BERTScorer(
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=False,
    )


def bert_score_single(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    try:
        _, _, f1 = _bert_scorer().score(
            [prediction],
            [reference],
            batch_size=1,
        )
        return float(f1.tolist()[0])
    except Exception:
        return 0.0


def max_bert_over_golds(prediction: str, golds: Iterable[str]) -> float:
    items = list(golds or [])
    if not items:
        return 0.0
    return max(bert_score_single(prediction, gold) for gold in items)


def is_yesno(query: str) -> bool:
    return bool(re.match(YESNO_PREFIX, (query or "").strip().lower()))


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
