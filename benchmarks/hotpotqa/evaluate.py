"""HotpotQA evaluation metrics.

Adapted from the official evaluation script:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

Metrics implemented:
  - Answer EM / F1 (standard HotpotQA, Yang et al. 2018 §5.2)
  - Contain-Match Accuracy (LinearRAG-style, Zhuang et al. 2025 §4.1)
  - Evidence Recall (title-level: fraction of gold SP titles in retrieved
    article titles — proxy for retrieval quality)

Note on SP EM/F1 and Joint EM/F1:
  The official HotpotQA metrics include Supporting Facts EM/F1 computed on
  predicted vs. gold sets of (title, sent_id) pairs, and Joint metrics that
  combine answer and SP scores. AgenticSearch is a black-box RAG system that
  outputs answers only (no explicit supporting fact predictions), so we
  cannot compute these metrics directly. We report Evidence Recall at the
  document-title level as the closest retrieval quality proxy.
  Ref: Yang et al. (2018) Table 4, Eq. for Joint F1.
"""

import re
import string
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set


def normalize_answer(s: str) -> str:
    """Lower text, remove articles/punctuation/extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def contain_match_score(prediction: str, ground_truth: str) -> int:
    """Bidirectional Contain-Match: 1 if normalized gold appears in prediction
    OR normalized prediction appears in gold.

    Bidirectional matching handles cases where the prediction is more
    specific (e.g. pred="August 15, 1843", gold="15 August 1843") or
    the gold is more verbose.
    """
    np = normalize_answer(prediction)
    ng = normalize_answer(ground_truth)
    if not np or not ng:
        return 0
    return int(ng in np or np in ng)


def _normalize_title(t: str) -> str:
    """Lowercase, strip punctuation and whitespace for fuzzy title matching."""
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _evidence_recall(
    retrieved_titles: List[str],
    supporting_facts: Dict[str, Any],
) -> float:
    """Fraction of gold supporting-fact titles found among retrieved article titles.

    Uses normalized matching to handle minor formatting differences
    (e.g. "Tivoli Gardens" vs "tivoli gardens").
    """
    sf_titles = supporting_facts.get("title", [])
    if not sf_titles:
        return 0.0
    unique_gold: Set[str] = set(sf_titles)
    norm_retrieved: Set[str] = {_normalize_title(t) for t in retrieved_titles}

    found = 0
    for gold_title in unique_gold:
        if _normalize_title(gold_title) in norm_retrieved:
            found += 1
    return found / len(unique_gold)


def evaluate_predictions(
    results: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Answer EM/F1, Contain-Match Acc, and Evidence Recall.

    Provides breakdown by question type (bridge/comparison) and
    difficulty level (easy/medium/hard).
    """
    sample_map = {s["_id"]: s for s in samples}

    metric_keys = ["em", "f1", "contain", "ev_recall"]
    overall = {k: [] for k in metric_keys}
    by_type: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {k: [] for k in metric_keys})
    by_level: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {k: [] for k in metric_keys})
    per_sample: List[Dict[str, Any]] = []

    for r in results:
        qid = r["_id"]
        sample = sample_map.get(qid)
        if not sample or r.get("error"):
            continue
        gold = sample.get("answer", "")
        pred = r.get("prediction", "")
        if not gold:
            continue

        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)
        contain = contain_match_score(pred, gold)
        q_type = sample.get("type", "unknown")
        q_level = sample.get("level", "unknown")

        retrieved_titles = r.get("retrieved_titles", [])
        sf = sample.get("supporting_facts", {})
        ev_recall = _evidence_recall(retrieved_titles, sf)

        for bucket in [overall, by_type[q_type], by_level[q_level]]:
            bucket["em"].append(em)
            bucket["f1"].append(f1)
            bucket["contain"].append(contain)
            bucket["ev_recall"].append(ev_recall)

        per_sample.append({
            "_id": qid,
            "em": em,
            "f1": round(f1, 4),
            "contain": contain,
            "ev_recall": round(ev_recall, 4),
            "type": q_type,
            "level": q_level,
            "gold": gold,
            "pred": pred,
            "question": r.get("question", ""),
        })

    def _avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _bucket(d):
        return {
            k: {
                "ans_em": _avg(v["em"]),
                "ans_f1": _avg(v["f1"]),
                "contain_acc": _avg(v["contain"]),
                "ev_recall": _avg(v["ev_recall"]),
                "count": len(v["em"]),
            }
            for k, v in d.items()
        }

    return {
        "overall": {
            "ans_em": _avg(overall["em"]),
            "ans_f1": _avg(overall["f1"]),
            "contain_acc": _avg(overall["contain"]),
            "ev_recall": _avg(overall["ev_recall"]),
            "count": len(overall["em"]),
        },
        "by_type": _bucket(by_type),
        "by_level": _bucket(by_level),
        "per_sample": per_sample,
    }
