"""Evaluation hooks (stub) for extraction and ranking metrics.
Designed to run without external datasets; supply gold labels at call time.
"""

from typing import Dict, List, Set, Tuple


def precision_recall_f1(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    """Compute P/R/F1 on set overlap; safe for empty inputs."""
    pred = set(pred or [])
    gold = set(gold or [])
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def ndcg_at_k(pred_ordered: List[str], gold_relevant: Set[str], k: int = 5) -> float:
    """Compute a simple nDCG@k given an ordered list of predictions and a set of relevant items."""
    if k <= 0:
        return 0.0
    pred_ordered = pred_ordered[:k]
    gold_relevant = set(gold_relevant or [])
    dcg = 0.0
    for idx, item in enumerate(pred_ordered):
        if item in gold_relevant:
            dcg += 1 / (log2(idx + 2))  # position is idx+1; denom log2(pos+1)
    ideal_hits = min(len(gold_relevant), k)
    idcg = sum(1 / (log2(i + 2)) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def log2(x: float) -> float:
    from math import log2 as _log2
    return _log2(x)


def evaluate_skill_extraction(pred_skills: Dict[str, List[str]], gold_skills: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """Category-wise and micro P/R/F1 for skills; expects dict of lists."""
    categories = set(pred_skills.keys()) | set(gold_skills.keys())
    results = {}
    all_pred = []
    all_gold = []
    for cat in categories:
        pset = set(pred_skills.get(cat, []))
        gset = set(gold_skills.get(cat, []))
        metrics = precision_recall_f1(pset, gset)
        results[cat] = metrics
        all_pred.extend(pset)
        all_gold.extend(gset)
    results['micro'] = precision_recall_f1(set(all_pred), set(all_gold))
    return results


def evaluate_ranking(pred_ordered_resumes: List[str], gold_relevant_resumes: Set[str], k: int = 5) -> Dict[str, float]:
    """Compute nDCG@k for ranking evaluation."""
    return {
        'ndcg@k': ndcg_at_k(pred_ordered_resumes, gold_relevant_resumes, k)
    }
