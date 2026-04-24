"""
evaluate_annotation_consistency.py

Q1 Inter-annotator agreement evaluation for the Alps benchmark.

Loads paired annotation bundles from batches 1–4 (each batch has
annotation_data_1.json and annotation_data_2.json), plus the single-annotator
batch 5. Randomly samples SAMPLE_SIZE complete sessions (seed=RANDOM_SEED),
then computes agreement metrics for all 4 tasks and prints a summary report.

Metrics computed:
  - memory_set_f1          : set-level F1 on which memories were selected
  - matched_memory_rougeL  : ROUGE-L on the text of matched memory pairs
  - memory_count_diff      : absolute difference in memory counts
  - confidence_pearson/spearman/kendall_tau : rank/linear agreement on confidence scores
  - confidence_MAE         : mean absolute error on confidence scores
  - type/label/time_scope kappa : Cohen's κ on categorical memory attributes
  - Task 3: query-memory pair F1, selected-memory agreement, ROUGE-L
  - Task 4: query set F1, query ROUGE-L, supporting-memory F1
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

# ── Configuration ────────────────────────────────────────────────────────────

RANDOM_SEED = 42       # fixed seed so the 1000-session sample is reproducible
SAMPLE_SIZE = 1000     # number of complete session pairs to evaluate
ROUGE_THRESHOLD = 0.5  # minimum ROUGE-L score to accept a semantic memory match

BASE = Path(__file__).parent   # repo root — batch folders live here
ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


# ── Text utilities ────────────────────────────────────────────────────────────

def norm(text: Any) -> str:
    """Normalise text for comparison: lowercase, collapse whitespace.

    Used before any exact-match or ROUGE-L comparison so that minor
    formatting differences (extra spaces, capitalisation) do not cause
    false negatives.
    """
    if text is None:
        return ''
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def safe_float(v: Any) -> float | None:
    """Convert a value to float, returning None on failure.

    Annotators sometimes store confidence as a string (e.g. '0.9').
    Returning None (rather than raising) lets callers skip bad entries
    instead of crashing.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rouge_l(a: Any, b: Any) -> float:
    """Compute ROUGE-L F1 between two text strings.

    Both inputs are normalised before scoring.
    Returns 1.0 if both are empty (perfect agreement on 'nothing'),
    0.0 if exactly one is empty.
    """
    na, nb = norm(a), norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return float(ROUGE.score(na, nb)['rougeL'].fmeasure)


# ── Aggregate statistics ──────────────────────────────────────────────────────

def safe_mean(vals: list[float]) -> float:
    """Mean of a list, ignoring None and NaN entries.

    Returns NaN if no valid values remain.
    """
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')


def safe_pearson(xs, ys) -> float:
    """Pearson r between two lists, skipping None pairs.

    Returns NaN if fewer than 2 valid pairs exist or if either list
    is constant (zero variance makes the correlation undefined).
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float('nan')
    xa, ya = zip(*pairs)
    if len(set(xa)) < 2 or len(set(ya)) < 2:
        return float('nan')
    return float(pearsonr(xa, ya).statistic)


def safe_spearman(xs, ys) -> float:
    """Spearman ρ between two lists, skipping None pairs.

    Preferred over Pearson for confidence scores because those values
    are discrete (0.5 / 0.7 / 0.8 / 0.9) and skewed toward 0.9.
    Spearman measures rank-order agreement rather than linear fit.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float('nan')
    xa, ya = zip(*pairs)
    return float(spearmanr(xa, ya).statistic)


def safe_kendall(xs, ys) -> float:
    """Kendall τ between two lists, skipping None pairs.

    A more conservative rank-correlation metric than Spearman —
    counts concordant vs discordant pairs directly.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float('nan')
    xa, ya = zip(*pairs)
    return float(kendalltau(xa, ya).statistic)


def safe_mae(xs, ys) -> float:
    """Mean absolute error between two lists, skipping None pairs.

    Used for confidence scores to express practical disagreement magnitude
    in the original 0–1 scale rather than as a correlation coefficient.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if not pairs:
        return float('nan')
    return float(np.mean([abs(x - y) for x, y in pairs]))


def safe_kappa(labels_a: list, labels_b: list, weights=None) -> float:
    """Cohen's κ between two label sequences, skipping None pairs.

    weights=None  → unweighted (nominal categories, e.g. type, label)
    weights='linear' → linear weighting (ordered categories, e.g. memoryDependency)

    Returns NaN if fewer than 2 valid pairs exist or only one unique
    class is present (kappa is undefined in those cases).
    """
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return float('nan')
    la, lb = zip(*pairs)
    classes = sorted(set(la) | set(lb))
    if len(classes) < 2:
        return float('nan')
    try:
        return float(cohen_kappa_score(la, lb, weights=weights, labels=classes))
    except Exception:
        return float('nan')


# ── Set-level F1 ─────────────────────────────────────────────────────────────

def set_f1(a: list[str], b: list[str]) -> float:
    """Set-level F1 between two lists of strings.

    Treats each list as a set and computes:
        precision = |A ∩ B| / |A|
        recall    = |A ∩ B| / |B|
        F1        = 2·P·R / (P+R)

    Both empty → 1.0 (perfect agreement on 'nothing selected').
    One empty  → 0.0 (complete disagreement).
    """
    sa = {x for x in a if x}
    sb = {x for x in b if x}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    tp = len(sa & sb)
    p = tp / len(sa)
    r = tp / len(sb)
    return 2 * p * r / (p + r) if p + r else 0.0


def singleton_f1(a: str, b: str) -> float:
    """F1 for a single-item 'set' (wraps set_f1 for scalar values).

    Used when each annotator produces exactly one value (e.g. Task 3
    selected memory) and we want to express agreement as 0 or 1.
    """
    return set_f1([a], [b])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_batch(batch_idx: int) -> tuple[dict, dict]:
    """Load one batch folder and return two session dicts (annotator 1 & 2).

    Each batch folder (e.g. '1/') contains annotation_data_1.json and
    annotation_data_2.json. Both files are bundle JSONs with a top-level
    'annotations' list — one entry per (session × task) pair.

    Returns:
        (sessions_1, sessions_2) — dicts keyed by sessionId, each value
        is {'sessionId': str, 'tasks': {taskName: annotationEntry}}.
        Returns ({}, {}) if either file is missing (e.g. batch 5 has no
        annotation_data_2.json).
    """
    def bundle_to_sessions(path: Path) -> dict[str, dict]:
        data = json.loads(path.read_text('utf-8'))
        sessions: dict[str, dict] = {}
        for ann in data['annotations']:
            sid = ann['sessionId']
            task = ann['task']
            if sid not in sessions:
                sessions[sid] = {'sessionId': sid, 'tasks': {}}
            sessions[sid]['tasks'][task] = ann
        return sessions

    p1 = BASE / str(batch_idx) / 'annotation_data_1.json'
    p2 = BASE / str(batch_idx) / 'annotation_data_2.json'
    if not p1.exists() or not p2.exists():
        return {}, {}
    return bundle_to_sessions(p1), bundle_to_sessions(p2)


# ── Memory matching ───────────────────────────────────────────────────────────

def exact_match(mems_a, mems_b):
    """Match memories between two annotators by exact normalised text.

    Builds a bucket index on annotator-B memories keyed by normalised
    value. For each annotator-A memory, pops the first matching B entry
    (greedy, one-to-one). Unmatched memories from both sides are returned
    separately for the subsequent semantic matching step.

    Returns:
        matched       : list of (mem_a, mem_b, score=1.0) triples
        unmatched_a   : A memories with no exact B counterpart
        unmatched_b   : B memories with no exact A counterpart
    """
    bucket = defaultdict(list)
    for m in mems_b:
        bucket[norm(m.get('value', ''))].append(m)

    matched, unmatched_a, used_b = [], [], set()
    for m in mems_a:
        key = norm(m.get('value', ''))
        cands = [x for x in bucket.get(key, []) if id(x) not in used_b]
        if cands:
            used_b.add(id(cands[0]))
            matched.append((m, cands[0], 1.0))
        else:
            unmatched_a.append(m)

    unmatched_b = [m for m in mems_b if id(m) not in used_b]
    return matched, unmatched_a, unmatched_b


def greedy_semantic(ua, ub, threshold=ROUGE_THRESHOLD):
    """Match remaining (unmatched) memories by greedy ROUGE-L similarity.

    Handles the common case where two annotators wrote the same memory
    with different phrasing. Scores all (A, B) pairs, then greedily
    assigns the highest-scoring pairs first, ensuring each memory is
    used at most once (one-to-one matching).

    Only pairs with ROUGE-L ≥ threshold are considered; this avoids
    false matches between semantically unrelated memories.

    Returns:
        matched : list of (mem_a, mem_b, rouge_score) triples
    """
    if not ua or not ub:
        return []
    cands = []
    for ia, a in enumerate(ua):
        for ib, b in enumerate(ub):
            s = rouge_l(a.get('value', ''), b.get('value', ''))
            if s >= threshold:
                cands.append((s, ia, ib))
    cands.sort(reverse=True)
    used_a, used_b, matched = set(), set(), []
    for s, ia, ib in cands:
        if ia not in used_a and ib not in used_b:
            used_a.add(ia)
            used_b.add(ib)
            matched.append((ua[ia], ub[ib], s))
    return matched


def match_memories(mems_a, mems_b):
    """Two-stage memory matching: exact first, then semantic.

    Stage 1 — exact_match: fast bucket lookup on normalised text.
    Stage 2 — greedy_semantic: ROUGE-L on the leftovers from stage 1.

    Returns:
        all_matched   : combined list of (mem_a, mem_b, score) triples
        n_unmatched_a : number of A memories not matched to any B memory
        n_unmatched_b : number of B memories not matched to any A memory
    """
    exact, ua, ub = exact_match(mems_a, mems_b)
    semantic = greedy_semantic(ua, ub)
    return exact + semantic, len(ua) - len(semantic), len(ub) - len(semantic)


# ── Per-task metric computation ───────────────────────────────────────────────

def memory_metrics(mems_a, mems_b) -> dict:
    """Compute all Task 1 / Task 2 agreement metrics for one session.

    Runs two-stage matching, then derives:
      - memory_set_f1        : how well the two memory sets overlap
      - matched_memory_rougeL: text similarity of matched pairs
      - memory_count_diff    : absolute count disagreement
      - _conf_a / _conf_b    : confidence scores for correlation (accumulated
                               across sessions before computing Pearson/Spearman)
      - _type/label/scope_a/b: categorical labels for kappa computation

    Private keys (prefixed '_') are intermediate lists that the caller
    accumulates across all sessions before computing corpus-level statistics.
    """
    pairs, n_unmatched_a, n_unmatched_b = match_memories(mems_a, mems_b)
    n = len(pairs)
    prec = n / len(mems_a) if mems_a else (1.0 if not mems_b else 0.0)
    rec  = n / len(mems_b) if mems_b else (1.0 if not mems_a else 0.0)
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    rouge_vals = [rouge_l(a.get('value'), b.get('value')) for a, b, _ in pairs]
    confs_a    = [safe_float(a.get('confidence')) for a, _, _ in pairs]
    confs_b    = [safe_float(b.get('confidence')) for _, b, _ in pairs]
    types_a    = [a.get('type')       for a, _, _ in pairs]
    types_b    = [b.get('type')       for _, b, _ in pairs]
    labels_a   = [a.get('label')      for a, _, _ in pairs]
    labels_b   = [b.get('label')      for _, b, _ in pairs]
    scopes_a   = [a.get('time_scope') for a, _, _ in pairs]
    scopes_b   = [b.get('time_scope') for _, b, _ in pairs]

    return {
        'memory_set_f1':         f1,
        'matched_memory_rougeL': safe_mean(rouge_vals),
        'memory_count_diff':     abs(len(mems_a) - len(mems_b)),
        'matched_count':         n,
        '_conf_a':   confs_a,  '_conf_b':   confs_b,
        '_type_a':   types_a,  '_type_b':   types_b,
        '_label_a':  labels_a, '_label_b':  labels_b,
        '_scope_a':  scopes_a, '_scope_b':  scopes_b,
    }


def task3_metrics(ann_a, ann_b) -> dict:
    """Compute Task 3 agreement metrics for one session.

    Task 3: annotators select a query and link it to a memory entry.
    Metrics:
      - query_memory_pair_f1     : exact-match F1 on the concatenated
                                   'queryText || memory.value' string
      - selected_memory_agreement: binary — did they pick the same memory?
      - selected_memory_rougeL   : ROUGE-L on the selected memory text
                                   (captures near-matches)
    """
    q_a = norm(ann_a.get('queryText'))
    q_b = norm(ann_b.get('queryText'))
    m_a = norm((ann_a.get('editableSelectedMemory') or {}).get('value'))
    m_b = norm((ann_b.get('editableSelectedMemory') or {}).get('value'))
    pair_a = f'{q_a} || {m_a}' if q_a or m_a else ''
    pair_b = f'{q_b} || {m_b}' if q_b or m_b else ''
    return {
        'query_memory_pair_f1':    singleton_f1(pair_a, pair_b),
        'selected_memory_agreement': 1.0 if m_a == m_b else 0.0,
        'selected_memory_rougeL':  rouge_l(m_a, m_b),
    }


def task4_metrics(ann_a, ann_b) -> dict:
    """Compute Task 4 agreement metrics for one session.

    Task 4: annotators construct multiple (query, supporting-memory) pairs.
    Sub-annotations are matched by queryId (a stable identifier assigned
    during annotation). For matched pairs, both query text and supporting
    memory selection are compared.

    Metrics:
      - query_f1             : set-level F1 on query texts (did they write
                               the same queries?)
      - query_rougeL         : ROUGE-L on matched query texts (wording similarity)
      - supporting_memory_f1 : exact-match F1 on the selected memory per query
      - matched_query_count  : number of queryId-matched sub-annotation pairs
                               (used to weight corpus-level averages)
    """
    subs_a = ann_a.get('subAnnotations') or []
    subs_b = ann_b.get('subAnnotations') or []
    qt_a = [norm(s.get('queryText')) for s in subs_a]
    qt_b = [norm(s.get('queryText')) for s in subs_b]
    map_a = {s['queryId']: s for s in subs_a if s.get('queryId')}
    map_b = {s['queryId']: s for s in subs_b if s.get('queryId')}
    shared = sorted(set(map_a) & set(map_b))

    rouge_vals, support_f1s = [], []
    for key in shared:
        sa, sb = map_a[key], map_b[key]
        rouge_vals.append(rouge_l(sa.get('queryText'), sb.get('queryText')))
        sel_a = norm((sa.get('editableSelectedMemory') or {}).get('value'))
        sel_b = norm((sb.get('editableSelectedMemory') or {}).get('value'))
        support_f1s.append(singleton_f1(sel_a, sel_b))

    return {
        'query_f1':             set_f1(qt_a, qt_b),
        'query_rougeL':         safe_mean(rouge_vals),
        'supporting_memory_f1': safe_mean(support_f1s),
        'matched_query_count':  len(shared),
    }


# ── Session-level helpers ─────────────────────────────────────────────────────

def get_ann(session: dict, task: str):
    """Extract the annotation payload for a given task from a session dict."""
    t = session.get('tasks', {}).get(task)
    return t.get('annotation') if t else None


def get_mems(session: dict, task: str) -> list:
    """Extract the memory list for Task 1 or Task 2 from a session dict.

    Task 1 stores gold memories under 'editableGoldMemories'.
    Task 2 stores updated memories under 'editableUpdatedMemories'.
    """
    ann = get_ann(session, task)
    if not ann:
        return []
    field = 'editableGoldMemories' if task == 'task1' else 'editableUpdatedMemories'
    return list(ann.get(field) or [])


# ── Aggregation and printing ──────────────────────────────────────────────────

def agg_memory(rows: list[dict], label: str) -> None:
    """Aggregate and print corpus-level metrics for Task 1 or Task 2.

    Accumulates per-session intermediate lists (_conf_a, _type_a, …) across
    all rows, then computes corpus-level correlation/kappa statistics.
    Individual session F1 / ROUGE values are averaged with safe_mean.
    """
    f1s    = [r['memory_set_f1']         for r in rows]
    rouges = [r['matched_memory_rougeL'] for r in rows]
    diffs  = [r['memory_count_diff']     for r in rows]

    # Flatten per-session confidence lists into one corpus-level list
    conf_a = sum([r['_conf_a'] for r in rows], [])
    conf_b = sum([r['_conf_b'] for r in rows], [])
    valid_conf = [(a, b) for a, b in zip(conf_a, conf_b)
                  if a is not None and b is not None]
    ca, cb = zip(*valid_conf) if valid_conf else ([], [])

    type_a  = sum([r['_type_a']  for r in rows], [])
    type_b  = sum([r['_type_b']  for r in rows], [])
    label_a = sum([r['_label_a'] for r in rows], [])
    label_b = sum([r['_label_b'] for r in rows], [])
    scope_a = sum([r['_scope_a'] for r in rows], [])
    scope_b = sum([r['_scope_b'] for r in rows], [])

    total_matched = sum(r['matched_count'] for r in rows)

    print(f'\n{"="*60}')
    print(f'  {label}  ({len(rows)} sessions, {total_matched} matched memory pairs)')
    print(f'{"="*60}')
    print(f'  memory_set_f1          : {safe_mean(f1s):.4f}  (std {float(np.std(f1s)):.4f})')
    print(f'  matched_memory_rougeL  : {safe_mean(rouges):.4f}')
    print(f'  memory_count_diff (avg): {safe_mean(diffs):.3f}')
    print(f'  --- confidence ({len(valid_conf)} pairs) ---')
    if ca:
        print(f'  confidence_pearson     : {safe_pearson(list(ca), list(cb)):.4f}')
        print(f'  confidence_spearman    : {safe_spearman(list(ca), list(cb)):.4f}')
        print(f'  confidence_kendall_tau : {safe_kendall(list(ca), list(cb)):.4f}')
        print(f'  confidence_MAE         : {safe_mae(list(ca), list(cb)):.4f}')
    print(f'  --- memory attribute kappa ({total_matched} matched pairs) ---')
    print(f'  type_kappa    (direct/indirect)  : {safe_kappa(type_a, type_b):.4f}')
    print(f'  label_kappa   (taxonomy category): {safe_kappa(label_a, label_b):.4f}')
    print(f'  time_scope_kappa                 : {safe_kappa(scope_a, scope_b):.4f}')


def agg_task3(rows: list[dict]) -> None:
    """Aggregate and print corpus-level metrics for Task 3."""
    pf1 = [r['query_memory_pair_f1']    for r in rows]
    agr = [r['selected_memory_agreement'] for r in rows]
    rl  = [r['selected_memory_rougeL']  for r in rows]
    print(f'\n{"="*60}')
    print(f'  TASK 3  ({len(rows)} sessions)')
    print(f'{"="*60}')
    print(f'  query_memory_pair_f1       : {safe_mean(pf1):.4f}')
    print(f'  selected_memory_agreement  : {safe_mean(agr):.4f}')
    print(f'  selected_memory_rougeL     : {safe_mean(rl):.4f}')


def agg_task4(rows: list[dict]) -> None:
    """Aggregate and print corpus-level metrics for Task 4."""
    qf1  = [r['query_f1']             for r in rows]
    qrl  = [r['query_rougeL']         for r in rows]
    smf1 = [r['supporting_memory_f1'] for r in rows]
    total_matched = sum(r['matched_query_count'] for r in rows)
    print(f'\n{"="*60}')
    print(f'  TASK 4  ({len(rows)} sessions, {total_matched} matched query pairs)')
    print(f'{"="*60}')
    print(f'  query_f1             : {safe_mean(qf1):.4f}  (std {float(np.std(qf1)):.4f})')
    print(f'  query_rougeL         : {safe_mean(qrl):.4f}')
    print(f'  supporting_memory_f1 : {safe_mean(smf1):.4f}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: load all batches, sample sessions, compute and print metrics.

    Workflow:
      1. Load batches 1–4 (paired) and attempt batch 5 (single — skipped if
         annotation_data_2.json is absent).
      2. Keep only sessions that have all 4 tasks in both annotators' files.
      3. Draw a random sample of SAMPLE_SIZE sessions (seed=RANDOM_SEED).
      4. Compute per-session metrics for each task.
      5. Aggregate and print the corpus-level summary.
    """
    # ── Step 1: collect all complete session pairs ────────────────────────
    all_pairs = []  # list of (session_A, session_B, batch_idx)
    batch_info = []

    for batch_idx in range(1, 6):
        s1, s2 = load_batch(batch_idx)
        if not s1 or not s2:
            # batch 5 has no annotation_data_2.json — skip for agreement analysis
            print(f'Batch {batch_idx}: skipped (no paired annotations)')
            continue

        shared   = sorted(set(s1) & set(s2))
        complete = [sid for sid in shared
                    if all(t in s1[sid]['tasks'] and t in s2[sid]['tasks']
                           for t in ['task1', 'task2', 'task3', 'task4'])]

        for sid in complete:
            all_pairs.append((s1[sid], s2[sid], batch_idx))
        batch_info.append({'batch': batch_idx, 'complete': len(complete)})

    total_available = len(all_pairs)
    print(f'\nComplete paired sessions available: {total_available}')
    for bi in batch_info:
        print(f'  Batch {bi["batch"]}: {bi["complete"]} sessions')

    # ── Step 2: random sample ─────────────────────────────────────────────
    rng     = random.Random(RANDOM_SEED)
    sampled = rng.sample(all_pairs, min(SAMPLE_SIZE, total_available))

    from collections import Counter
    sample_dist = Counter(b for _, _, b in sampled)
    print(f'\nSampled {len(sampled)} sessions (seed={RANDOM_SEED})')
    for b, cnt in sorted(sample_dist.items()):
        print(f'  Batch {b}: {cnt} sessions')

    # ── Step 3: compute per-session metrics ───────────────────────────────
    t1_list, t2_list, t3_list, t4_list = [], [], [], []

    for a, b, _ in sampled:
        t1 = memory_metrics(get_mems(a, 'task1'), get_mems(b, 'task1'))
        t2 = memory_metrics(get_mems(a, 'task2'), get_mems(b, 'task2'))

        ann3_a, ann3_b = get_ann(a, 'task3'), get_ann(b, 'task3')
        t3 = task3_metrics(ann3_a, ann3_b) if ann3_a and ann3_b else None

        ann4_a, ann4_b = get_ann(a, 'task4'), get_ann(b, 'task4')
        t4 = task4_metrics(ann4_a, ann4_b) if ann4_a and ann4_b else None

        t1_list.append(t1)
        t2_list.append(t2)
        if t3:
            t3_list.append(t3)
        if t4:
            t4_list.append(t4)

    # ── Step 4: aggregate and print ───────────────────────────────────────
    print('\n' + '#' * 60)
    print(f'  Q1 BENCHMARK CONSTRUCTION EVALUATION')
    print(f'  Sample: {len(sampled)} sessions (seed={RANDOM_SEED})')
    print('#' * 60)

    agg_memory(t1_list, 'TASK 1  (gold memory extraction)')
    agg_memory(t2_list, 'TASK 2  (memory update)')
    agg_task3(t3_list)
    agg_task4(t4_list)

    print('\n' + '#' * 60)
    print('  DONE')
    print('#' * 60)


if __name__ == '__main__':
    main()
