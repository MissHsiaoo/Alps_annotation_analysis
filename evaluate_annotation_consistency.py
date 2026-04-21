from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import pearsonr

try:
    from bert_score import score as bert_score_fn
except Exception:  # pragma: no cover - optional dependency path
    bert_score_fn = None


TASK1 = "task1"
TASK2 = "task2"
TASK3 = "task3"
TASK4 = "task4"
ALL_TASKS = (TASK1, TASK2, TASK3, TASK4)

ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


@dataclass
class MatchedPair:
    item_a: dict[str, Any]
    item_b: dict[str, Any]
    score: float
    match_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two annotation directories or files.")
    parser.add_argument("--dir-a", type=Path, required=True, help="Directory containing annotator A session JSON files.")
    parser.add_argument("--dir-b", type=Path, required=True, help="Directory containing annotator B session JSON files.")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of paired sessions to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for paired-session sampling.")
    parser.add_argument(
        "--bertscore",
        choices=("on", "off"),
        default="off",
        help="Whether to compute BERTScore metrics.",
    )
    parser.add_argument(
        "--matching-threshold",
        type=float,
        default=0.5,
        help="Minimum semantic similarity score for non-exact memory/query matching.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save final JSON results.")
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def safe_mean(values: list[float]) -> float:
    valid = [value for value in values if not is_nan(value)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def safe_pearson(xs: list[float | None], ys: list[float | None]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float("nan")
    unique_x = {x for x, _ in pairs}
    unique_y = {y for _, y in pairs}
    if len(unique_x) <= 1 or len(unique_y) <= 1:
        return float("nan")
    x_values = [x for x, _ in pairs]
    y_values = [y for _, y in pairs]
    return float(pearsonr(x_values, y_values).statistic)


def compute_rouge_l_f1(text_a: Any, text_b: Any) -> float:
    normalized_a = normalize_text(text_a)
    normalized_b = normalize_text(text_b)
    if not normalized_a and not normalized_b:
        return 1.0
    if not normalized_a or not normalized_b:
        return 0.0
    return float(ROUGE_SCORER.score(normalized_a, normalized_b)["rougeL"].fmeasure)


def compute_bertscore_batch(texts_a: list[str], texts_b: list[str], enabled: bool) -> list[float]:
    if not texts_a:
        return []
    if not enabled:
        return [float("nan")] * len(texts_a)
    if bert_score_fn is None:
        raise RuntimeError("BERTScore requested but bert_score is not available.")
    _, _, f1 = bert_score_fn(
        texts_a,
        texts_b,
        model_type="bert-base-multilingual-cased",
        device="cpu",
        verbose=False,
    )
    return [float(value) for value in f1.tolist()]


def singleton_f1(value_a: str, value_b: str) -> float:
    set_a = {value_a} if value_a else set()
    set_b = {value_b} if value_b else set()
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    tp = len(set_a & set_b)
    precision = tp / len(set_a) if set_a else 0.0
    recall = tp / len(set_b) if set_b else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def set_f1(items_a: list[str], items_b: list[str]) -> float:
    set_a = {item for item in items_a if item}
    set_b = {item for item in items_b if item}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    tp = len(set_a & set_b)
    precision = tp / len(set_a) if set_a else 0.0
    recall = tp / len(set_b) if set_b else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_memory_value(memory: dict[str, Any]) -> str:
    return str(memory.get("value") or "")


def exact_match_pairs(
    memories_a: list[dict[str, Any]],
    memories_b: list[dict[str, Any]],
) -> tuple[list[MatchedPair], list[dict[str, Any]], list[dict[str, Any]]]:
    buckets_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_b in memories_b:
        buckets_b[normalize_text(extract_memory_value(item_b))].append(item_b)

    matched: list[MatchedPair] = []
    unmatched_a: list[dict[str, Any]] = []
    matched_b_ids: set[int] = set()

    for item_a in memories_a:
        key = normalize_text(extract_memory_value(item_a))
        candidates = buckets_b.get(key, [])
        while candidates and id(candidates[0]) in matched_b_ids:
            candidates.pop(0)
        if candidates:
            item_b = candidates.pop(0)
            matched_b_ids.add(id(item_b))
            matched.append(MatchedPair(item_a=item_a, item_b=item_b, score=1.0, match_type="exact"))
        else:
            unmatched_a.append(item_a)

    unmatched_b = [item_b for item_b in memories_b if id(item_b) not in matched_b_ids]
    return matched, unmatched_a, unmatched_b


def greedy_semantic_match(
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    threshold: float,
) -> list[MatchedPair]:
    if not items_a or not items_b:
        return []

    candidates: list[tuple[float, int, int]] = []
    for index_a, item_a in enumerate(items_a):
        value_a = extract_memory_value(item_a)
        for index_b, item_b in enumerate(items_b):
            value_b = extract_memory_value(item_b)
            score = compute_rouge_l_f1(value_a, value_b)
            if score >= threshold:
                candidates.append((score, index_a, index_b))

    candidates.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[MatchedPair] = []

    for score, index_a, index_b in candidates:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matched.append(
            MatchedPair(
                item_a=items_a[index_a],
                item_b=items_b[index_b],
                score=score,
                match_type="semantic",
            )
        )

    return matched


def match_memories(
    memories_a: list[dict[str, Any]],
    memories_b: list[dict[str, Any]],
    matching_threshold: float,
) -> dict[str, Any]:
    exact_pairs, unmatched_a, unmatched_b = exact_match_pairs(memories_a, memories_b)
    semantic_pairs = greedy_semantic_match(unmatched_a, unmatched_b, matching_threshold)

    matched_a_ids = {id(pair.item_a) for pair in semantic_pairs}
    matched_b_ids = {id(pair.item_b) for pair in semantic_pairs}
    remaining_a = [item for item in unmatched_a if id(item) not in matched_a_ids]
    remaining_b = [item for item in unmatched_b if id(item) not in matched_b_ids]

    pairs = exact_pairs + semantic_pairs
    return {
        "pairs": pairs,
        "unmatched_a": remaining_a,
        "unmatched_b": remaining_b,
    }


def compute_memory_metrics(
    memories_a: list[dict[str, Any]],
    memories_b: list[dict[str, Any]],
    use_bertscore: bool,
    matching_threshold: float,
) -> dict[str, Any]:
    matched = match_memories(memories_a, memories_b, matching_threshold)
    pairs: list[MatchedPair] = matched["pairs"]

    matched_count = len(pairs)
    precision = matched_count / len(memories_a) if memories_a else 1.0 if not memories_b else 0.0
    recall = matched_count / len(memories_b) if memories_b else 1.0 if not memories_a else 0.0
    if precision + recall == 0:
        memory_set_f1 = 0.0
    else:
        memory_set_f1 = 2 * precision * recall / (precision + recall)

    rouge_values = [compute_rouge_l_f1(extract_memory_value(pair.item_a), extract_memory_value(pair.item_b)) for pair in pairs]
    bert_values = compute_bertscore_batch(
        [extract_memory_value(pair.item_a) for pair in pairs],
        [extract_memory_value(pair.item_b) for pair in pairs],
        enabled=use_bertscore,
    )

    confidences_a = [safe_float(pair.item_a.get("confidence")) for pair in pairs]
    confidences_b = [safe_float(pair.item_b.get("confidence")) for pair in pairs]

    return {
        "memory_set_f1": float(memory_set_f1),
        "matched_memory_bertscore": safe_mean(bert_values),
        "matched_memory_rougeL": safe_mean(rouge_values),
        "confidence_pearson": safe_pearson(confidences_a, confidences_b),
        "memory_count_diff": abs(len(memories_a) - len(memories_b)),
        "matched_count": matched_count,
        "unmatched_a_count": len(matched["unmatched_a"]),
        "unmatched_b_count": len(matched["unmatched_b"]),
    }


def extract_task_payload(sample: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    return sample.get("tasks", {}).get(task_name)


def extract_memory_list(task_payload: dict[str, Any] | None, field_name: str) -> list[dict[str, Any]]:
    if not task_payload:
        return []
    annotation = task_payload.get("annotation") or {}
    return list(annotation.get(field_name) or [])


def selected_memory_value(task_payload: dict[str, Any] | None) -> str:
    if not task_payload:
        return ""
    annotation = task_payload.get("annotation") or {}
    selected = annotation.get("editableSelectedMemory") or {}
    return str(selected.get("value") or "")


def compute_task3_metrics(sample_a: dict[str, Any], sample_b: dict[str, Any], use_bertscore: bool) -> dict[str, Any]:
    task_a = extract_task_payload(sample_a, TASK3)
    task_b = extract_task_payload(sample_b, TASK3)
    if not task_a or not task_b:
        return {
            "query_memory_pair_f1": float("nan"),
            "selected_memory_agreement": float("nan"),
            "selected_memory_bertscore": float("nan"),
            "selected_memory_rougeL": float("nan"),
        }

    annotation_a = task_a.get("annotation") or {}
    annotation_b = task_b.get("annotation") or {}
    query_a = normalize_text(annotation_a.get("queryText"))
    query_b = normalize_text(annotation_b.get("queryText"))
    memory_a = normalize_text(selected_memory_value(task_a))
    memory_b = normalize_text(selected_memory_value(task_b))

    pair_a = f"{query_a} || {memory_a}" if query_a or memory_a else ""
    pair_b = f"{query_b} || {memory_b}" if query_b or memory_b else ""

    bert_value = compute_bertscore_batch([memory_a], [memory_b], enabled=use_bertscore)[0]
    return {
        "query_memory_pair_f1": singleton_f1(pair_a, pair_b),
        "selected_memory_agreement": 1.0 if memory_a == memory_b else 0.0,
        "selected_memory_bertscore": bert_value,
        "selected_memory_rougeL": compute_rouge_l_f1(memory_a, memory_b),
    }


def task4_alignment_key(sub_annotation: dict[str, Any]) -> str:
    query_id = str(sub_annotation.get("queryId") or "").strip()
    if query_id:
        return query_id
    ability = normalize_text(sub_annotation.get("ability"))
    query_text = normalize_text(sub_annotation.get("queryText"))
    return f"{ability}::{query_text}"


def compute_task4_metrics(sample_a: dict[str, Any], sample_b: dict[str, Any], use_bertscore: bool) -> dict[str, Any]:
    task_a = extract_task_payload(sample_a, TASK4)
    task_b = extract_task_payload(sample_b, TASK4)
    if not task_a or not task_b:
        return {
            "query_f1": float("nan"),
            "query_bertscore": float("nan"),
            "query_rougeL": float("nan"),
            "supporting_memory_f1": float("nan"),
            "matched_query_count": 0,
        }

    annotation_a = task_a.get("annotation") or {}
    annotation_b = task_b.get("annotation") or {}
    sub_a = list(annotation_a.get("subAnnotations") or [])
    sub_b = list(annotation_b.get("subAnnotations") or [])

    query_texts_a = [normalize_text(item.get("queryText")) for item in sub_a]
    query_texts_b = [normalize_text(item.get("queryText")) for item in sub_b]
    query_f1 = set_f1(query_texts_a, query_texts_b)

    map_a = {task4_alignment_key(item): item for item in sub_a}
    map_b = {task4_alignment_key(item): item for item in sub_b}
    shared_keys = sorted(set(map_a) & set(map_b))

    query_rouges: list[float] = []
    query_bertscore_inputs_a: list[str] = []
    query_bertscore_inputs_b: list[str] = []
    support_f1_values: list[float] = []

    for key in shared_keys:
        item_a = map_a[key]
        item_b = map_b[key]
        query_a = str(item_a.get("queryText") or "")
        query_b = str(item_b.get("queryText") or "")
        query_rouges.append(compute_rouge_l_f1(query_a, query_b))
        query_bertscore_inputs_a.append(normalize_text(query_a))
        query_bertscore_inputs_b.append(normalize_text(query_b))

        selected_a = normalize_text((item_a.get("editableSelectedMemory") or {}).get("value"))
        selected_b = normalize_text((item_b.get("editableSelectedMemory") or {}).get("value"))
        support_f1_values.append(singleton_f1(selected_a, selected_b))

    query_bertscores = compute_bertscore_batch(query_bertscore_inputs_a, query_bertscore_inputs_b, enabled=use_bertscore)
    return {
        "query_f1": query_f1,
        "query_bertscore": safe_mean(query_bertscores),
        "query_rougeL": safe_mean(query_rouges),
        "supporting_memory_f1": safe_mean(support_f1_values),
        "matched_query_count": len(shared_keys),
    }


def aggregate_task_metrics(per_sample_results: list[dict[str, Any]], task_name: str) -> dict[str, float]:
    if not per_sample_results:
        return {}

    metric_names = [
        key
        for key in per_sample_results[0][task_name].keys()
        if key not in {"matched_count", "unmatched_a_count", "unmatched_b_count", "matched_query_count"}
    ]
    aggregated: dict[str, float] = {}
    for metric_name in metric_names:
        values = [sample[task_name][metric_name] for sample in per_sample_results]
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        aggregated[metric_name] = safe_mean([float(value) for value in numeric_values])
    return aggregated


def load_samples_from_dir(directory: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        session_id = str(payload.get("sessionId") or "")
        if session_id:
            samples[session_id] = payload
    return samples


def sample_session_ids(ids: list[str], sample_size: int, seed: int) -> list[str]:
    if sample_size >= len(ids):
        return sorted(ids)
    rng = random.Random(seed)
    return sorted(rng.sample(ids, sample_size))


def evaluate_session(
    session_id: str,
    sample_a: dict[str, Any],
    sample_b: dict[str, Any],
    use_bertscore: bool,
    matching_threshold: float,
) -> dict[str, Any]:
    task1_metrics = compute_memory_metrics(
        extract_memory_list(extract_task_payload(sample_a, TASK1), "editableGoldMemories"),
        extract_memory_list(extract_task_payload(sample_b, TASK1), "editableGoldMemories"),
        use_bertscore=use_bertscore,
        matching_threshold=matching_threshold,
    )
    task2_metrics = compute_memory_metrics(
        extract_memory_list(extract_task_payload(sample_a, TASK2), "editableUpdatedMemories"),
        extract_memory_list(extract_task_payload(sample_b, TASK2), "editableUpdatedMemories"),
        use_bertscore=use_bertscore,
        matching_threshold=matching_threshold,
    )
    task3_metrics = compute_task3_metrics(sample_a, sample_b, use_bertscore=use_bertscore)
    task4_metrics = compute_task4_metrics(sample_a, sample_b, use_bertscore=use_bertscore)

    return {
        "sessionId": session_id,
        "task1": task1_metrics,
        "task2": task2_metrics,
        "task3": task3_metrics,
        "task4": task4_metrics,
    }


def main() -> None:
    args = parse_args()
    use_bertscore = args.bertscore == "on"

    samples_a = load_samples_from_dir(args.dir_a)
    samples_b = load_samples_from_dir(args.dir_b)
    shared_session_ids = sorted(set(samples_a) & set(samples_b))
    selected_session_ids = sample_session_ids(shared_session_ids, args.sample_size, args.seed)

    per_sample_results = [
        evaluate_session(
            session_id=session_id,
            sample_a=samples_a[session_id],
            sample_b=samples_b[session_id],
            use_bertscore=use_bertscore,
            matching_threshold=args.matching_threshold,
        )
        for session_id in selected_session_ids
    ]

    summary = {
        "settings": {
            "dir_a": str(args.dir_a),
            "dir_b": str(args.dir_b),
            "sample_size": len(selected_session_ids),
            "seed": args.seed,
            "bertscore": args.bertscore,
            "matching_threshold": args.matching_threshold,
        },
        "task1": aggregate_task_metrics(per_sample_results, TASK1),
        "task2": aggregate_task_metrics(per_sample_results, TASK2),
        "task3": aggregate_task_metrics(per_sample_results, TASK3),
        "task4": aggregate_task_metrics(per_sample_results, TASK4),
        "per_sample_results": per_sample_results,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
