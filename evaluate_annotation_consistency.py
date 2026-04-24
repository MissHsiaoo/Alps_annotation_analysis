"""
evaluate_annotation_consistency.py

Q1 Inter-annotator agreement evaluation for the Alps benchmark.

Loads paired annotation bundles from batches 1–4 (each batch has
annotation_data_1.json and annotation_data_2.json). Batch 5 is skipped
because it has no second-annotator file.

Randomly samples SAMPLE_SIZE complete session pairs (seed=RANDOM_SEED),
then computes agreement metrics for all 4 tasks and prints a summary report.
Results are optionally saved as a structured JSON file (--output-json).

Metrics:
  Tasks 1 & 2 (memory sets)
    memory_set_f1              set-level F1 on selected memories
    matched_memory_rougeL      ROUGE-L on matched-pair texts  [note: see ROUGE-L bias below]
    matched_memory_bertscore   BERTScore F1 on matched pairs  (requires --bertscore on)
    memory_count_diff          |len(A) - len(B)|
    confidence_pearson/spearman/kendall_tau/MAE
    type / label / time_scope kappa (Cohen's κ)

  Task 3 (query-memory linkage)
    query_memory_pair_f1       exact-match F1 on (query, memory) pairs
    selected_memory_agreement  binary: same memory chosen?
    selected_memory_rougeL     ROUGE-L on selected memory texts
    selected_memory_bertscore  BERTScore on selected memory texts (--bertscore on)

  Task 4 (multi-query construction)
    query_f1                   set-level F1 on query texts
    query_rougeL               ROUGE-L on queryId-matched query texts
    supporting_memory_f1       exact-match F1 on supporting memory per query

NOTE — ROUGE-L bias in matched_memory_rougeL
  The same ROUGE-L metric is used to both create semantic matches (threshold
  ≥ 0.5) and to score the matched pairs.  Because only pairs that already
  cleared the threshold are scored, the metric is upward-biased relative to
  the true pairwise similarity distribution.  Exact matches (≈96% of pairs)
  are unaffected.  This is disclosed in the JSON output and in printed output.

Usage:
  python evaluate_annotation_consistency.py
  python evaluate_annotation_consistency.py --output-json results.json
  python evaluate_annotation_consistency.py --bertscore on --output-json results.json
  python evaluate_annotation_consistency.py --sample-size 500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

try:
    from bert_score import score as _bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _bert_score_fn = None
    _BERTSCORE_AVAILABLE = False

# ── Defaults (overridable via CLI) ────────────────────────────────────────────
RANDOM_SEED     = 42
SAMPLE_SIZE     = 1000
ROUGE_THRESHOLD = 0.5
BOOTSTRAP_N     = 1000   # resamples for 95% confidence intervals

BASE  = Path(__file__).parent
ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    All arguments are optional — the defaults reproduce the numbers in
    analysis.md (1000 sessions, seed 42, BERTScore off).
    """
    p = argparse.ArgumentParser(
        description='Q1 inter-annotator agreement for the Alps benchmark.'
    )
    p.add_argument('--sample-size', type=int, default=SAMPLE_SIZE,
                   help=f'Sessions to sample (default {SAMPLE_SIZE})')
    p.add_argument('--seed', type=int, default=RANDOM_SEED,
                   help=f'Random seed (default {RANDOM_SEED})')
    p.add_argument('--bertscore', choices=('on', 'off'), default='off',
                   help='Compute BERTScore (slow on CPU; default off)')
    p.add_argument('--output-json', type=Path, default=None,
                   help='Save structured results to this JSON file')
    p.add_argument('--bootstrap-n', type=int, default=BOOTSTRAP_N,
                   help=f'Bootstrap resamples for 95%% CI (default {BOOTSTRAP_N})')
    return p.parse_args()


# ── Text utilities ────────────────────────────────────────────────────────────

def norm(text: Any) -> str:
    """Lowercase and collapse whitespace for comparison."""
    if text is None:
        return ''
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def safe_float(v: Any) -> float | None:
    """Convert to float; return None on failure (e.g. string or missing field)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rouge_l(a: Any, b: Any) -> float:
    """ROUGE-L F1 on two texts after normalisation.

    Both empty → 1.0 (perfect agreement on nothing).
    One empty  → 0.0.
    """
    na, nb = norm(a), norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return float(ROUGE.score(na, nb)['rougeL'].fmeasure)


# ── Statistics ────────────────────────────────────────────────────────────────

def safe_mean(vals) -> float:
    """Mean ignoring None and NaN.  Returns NaN if nothing valid remains."""
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float('nan')


def safe_std(vals) -> float:
    """Std-dev ignoring None / NaN."""
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.std(v)) if len(v) > 1 else float('nan')


def safe_pearson(xs, ys) -> float:
    """Pearson r, skipping None pairs; NaN if < 2 valid pairs or zero variance."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float('nan')
    xa, ya = zip(*pairs)
    if len(set(xa)) < 2 or len(set(ya)) < 2:
        return float('nan')
    return float(pearsonr(xa, ya).statistic)


def safe_spearman(xs, ys) -> float:
    """Spearman ρ, skipping None pairs.

    Preferred over Pearson for discrete confidence values (0.5/0.7/0.8/0.9).
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float('nan')
    xa, ya = zip(*pairs)
    return float(spearmanr(xa, ya).statistic)


def safe_kendall(xs, ys) -> float:
    """Kendall τ, skipping None pairs (conservative rank-correlation)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return float('nan')
    xa, ya = zip(*pairs)
    return float(kendalltau(xa, ya).statistic)


def safe_mae(xs, ys) -> float:
    """Mean absolute error, skipping None pairs."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if not pairs:
        return float('nan')
    return float(np.mean([abs(x - y) for x, y in pairs]))


def safe_kappa(labels_a, labels_b, weights=None) -> float:
    """Cohen's κ, skipping None pairs.

    weights=None      → nominal (type, label)
    weights='linear'  → ordered categories (memoryDependency)

    Returns NaN if < 2 valid pairs or only one class present.
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


def bootstrap_ci(values: list[float], n: int = BOOTSTRAP_N,
                 lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """Non-parametric bootstrap 95% confidence interval for the mean.

    Draws n resamples with replacement, computes the mean of each,
    and returns the (lo, hi) percentiles of that distribution.
    Returns (nan, nan) if fewer than 2 valid values.
    """
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(RANDOM_SEED)
    means = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n)]
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


# ── Set-level F1 ──────────────────────────────────────────────────────────────

def set_f1(a: list[str], b: list[str]) -> float:
    """Set F1: 2·P·R/(P+R) where P = |A∩B|/|A|, R = |A∩B|/|B|.

    Both empty → 1.0.  One empty → 0.0.
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
    """F1 for single-value 'sets' — returns 1.0 if identical, 0.0 otherwise."""
    return set_f1([a], [b])


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(texts_a: list[str], texts_b: list[str],
                      enabled: bool) -> list[float]:
    """Compute BERTScore F1 for paired text lists.

    Returns a list of NaN if disabled or if the dependency is unavailable.
    Uses bert-base-multilingual-cased to handle mixed Chinese/English content.

    Raises RuntimeError if --bertscore on but the package is not installed.
    """
    if not texts_a:
        return []
    if not enabled:
        return [float('nan')] * len(texts_a)
    if not _BERTSCORE_AVAILABLE:
        raise RuntimeError(
            'BERTScore requested but the bert-score package is not installed.\n'
            'Install it with:  pip install bert-score'
        )
    _, _, f1 = _bert_score_fn(
        texts_a, texts_b,
        model_type='bert-base-multilingual-cased',
        device='cpu',
        verbose=False,
    )
    return [float(x) for x in f1.tolist()]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_batch(batch_idx: int) -> tuple[dict, dict]:
    """Load one batch folder into two session dicts (annotator 1 & 2).

    Each bundle JSON has a top-level 'annotations' list; each entry covers
    one (session × task) pair.  We pivot into:
        {sessionId: {'sessionId': str, 'tasks': {taskName: entry}}}

    Returns ({}, {}) if either file is missing (e.g. batch 5 → no pair).
    """
    def _parse(path: Path) -> dict[str, dict]:
        data = json.loads(path.read_text('utf-8'))
        out: dict[str, dict] = {}
        for ann in data['annotations']:
            sid  = ann['sessionId']
            task = ann['task']
            if sid not in out:
                out[sid] = {'sessionId': sid, 'tasks': {}}
            out[sid]['tasks'][task] = ann
        return out

    p1 = BASE / str(batch_idx) / 'annotation_data_1.json'
    p2 = BASE / str(batch_idx) / 'annotation_data_2.json'
    if not p1.exists() or not p2.exists():
        return {}, {}
    return _parse(p1), _parse(p2)


# ── Memory matching ───────────────────────────────────────────────────────────

def exact_match(mems_a, mems_b):
    """Match memories by exact normalised text (one-to-one, greedy).

    Returns (matched_triples, unmatched_a, unmatched_b).
    Each triple is (mem_a, mem_b, score=1.0).
    """
    bucket: dict[str, list] = defaultdict(list)
    for m in mems_b:
        bucket[norm(m.get('value', ''))].append(m)

    matched, unmatched_a, used_b = [], [], set()
    for m in mems_a:
        key   = norm(m.get('value', ''))
        cands = [x for x in bucket.get(key, []) if id(x) not in used_b]
        if cands:
            used_b.add(id(cands[0]))
            matched.append((m, cands[0], 1.0))
        else:
            unmatched_a.append(m)

    unmatched_b = [m for m in mems_b if id(m) not in used_b]
    return matched, unmatched_a, unmatched_b


def greedy_semantic(ua, ub, threshold: float = ROUGE_THRESHOLD):
    """Match remaining memories by greedy highest-ROUGE-L pairing.

    Handles same-meaning memories with different phrasing.  Only pairs
    scoring ≥ threshold are eligible.  Each memory is used at most once.
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
    """Two-stage matching: exact first, then semantic ROUGE-L on leftovers.

    Returns (all_matched_triples, n_unmatched_a, n_unmatched_b).
    """
    exact, ua, ub = exact_match(mems_a, mems_b)
    semantic      = greedy_semantic(ua, ub)
    return exact + semantic, len(ua) - len(semantic), len(ub) - len(semantic)


# ── Per-task metrics ──────────────────────────────────────────────────────────

def memory_metrics(mems_a, mems_b, use_bertscore: bool = False) -> dict:
    """All Task 1 / Task 2 agreement metrics for one session.

    Private keys (prefix '_') are intermediate lists accumulated across
    sessions before corpus-level correlation / kappa is computed.  They
    are stripped from the per-session JSON output.

    ROUGE-L bias note: matched_memory_rougeL only scores pairs that passed
    the ROUGE-L ≥ 0.5 matching threshold, so its mean is upward-biased
    relative to the full pairwise distribution.  Exact matches (score 1.0)
    dominate (~96%) so the practical effect is small, but it should be
    disclosed when reporting.
    """
    pairs, n_ua, n_ub = match_memories(mems_a, mems_b)
    n    = len(pairs)
    prec = n / len(mems_a) if mems_a else (1.0 if not mems_b else 0.0)
    rec  = n / len(mems_b) if mems_b else (1.0 if not mems_a else 0.0)
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    texts_a = [a.get('value', '') for a, _, _ in pairs]
    texts_b = [b.get('value', '') for _, b, _ in pairs]

    rouge_vals  = [rouge_l(a, b) for a, b in zip(texts_a, texts_b)]
    bert_vals   = compute_bertscore(texts_a, texts_b, enabled=use_bertscore)

    confs_a  = [safe_float(a.get('confidence')) for a, _, _ in pairs]
    confs_b  = [safe_float(b.get('confidence')) for _, b, _ in pairs]
    types_a  = [a.get('type')       for a, _, _ in pairs]
    types_b  = [b.get('type')       for _, b, _ in pairs]
    labels_a = [a.get('label')      for a, _, _ in pairs]
    labels_b = [b.get('label')      for _, b, _ in pairs]
    scopes_a = [a.get('time_scope') for a, _, _ in pairs]
    scopes_b = [b.get('time_scope') for _, b, _ in pairs]

    return {
        # public — included in per-session JSON
        'memory_set_f1':           f1,
        'matched_memory_rougeL':   safe_mean(rouge_vals),
        'matched_memory_bertscore': safe_mean(bert_vals),
        'memory_count_diff':       abs(len(mems_a) - len(mems_b)),
        'matched_count':           n,
        'unmatched_a':             n_ua,
        'unmatched_b':             n_ub,
        # private — accumulation lists for corpus-level stats
        '_conf_a':   confs_a,  '_conf_b':   confs_b,
        '_type_a':   types_a,  '_type_b':   types_b,
        '_label_a':  labels_a, '_label_b':  labels_b,
        '_scope_a':  scopes_a, '_scope_b':  scopes_b,
    }


def task3_metrics(ann_a, ann_b, use_bertscore: bool = False) -> dict:
    """Task 3 agreement metrics for one session.

    Task 3: each annotator selects a query and links it to a memory entry.
    We measure whether they chose the same (query, memory) pair and how
    similar the memory texts are when they differed.
    """
    q_a = norm(ann_a.get('queryText'))
    q_b = norm(ann_b.get('queryText'))
    m_a = norm((ann_a.get('editableSelectedMemory') or {}).get('value'))
    m_b = norm((ann_b.get('editableSelectedMemory') or {}).get('value'))
    pair_a = f'{q_a} || {m_a}' if q_a or m_a else ''
    pair_b = f'{q_b} || {m_b}' if q_b or m_b else ''

    bert_val = compute_bertscore([m_a], [m_b], enabled=use_bertscore)

    return {
        'query_memory_pair_f1':      singleton_f1(pair_a, pair_b),
        'selected_memory_agreement': 1.0 if m_a == m_b else 0.0,
        'selected_memory_rougeL':    rouge_l(m_a, m_b),
        'selected_memory_bertscore': bert_val[0] if bert_val else float('nan'),
    }


def task4_metrics(ann_a, ann_b) -> dict:
    """Task 4 agreement metrics for one session.

    Task 4: annotators construct multiple (query, supporting-memory) pairs.
    Sub-annotations are matched by queryId.  For each matched pair, query
    text similarity and supporting memory agreement are measured.
    """
    subs_a = ann_a.get('subAnnotations') or []
    subs_b = ann_b.get('subAnnotations') or []
    qt_a   = [norm(s.get('queryText')) for s in subs_a]
    qt_b   = [norm(s.get('queryText')) for s in subs_b]
    map_a  = {s['queryId']: s for s in subs_a if s.get('queryId')}
    map_b  = {s['queryId']: s for s in subs_b if s.get('queryId')}
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
    """Return the annotation payload for a task, or None if absent."""
    t = session.get('tasks', {}).get(task)
    return t.get('annotation') if t else None


def get_mems(session: dict, task: str) -> list:
    """Return the memory list for task1 (gold) or task2 (updated)."""
    ann = get_ann(session, task)
    if not ann:
        return []
    field = 'editableGoldMemories' if task == 'task1' else 'editableUpdatedMemories'
    return list(ann.get(field) or [])


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _stat(values: list[float], n_boot: int) -> dict:
    """Build a summary dict {mean, std, ci95_lo, ci95_hi} for a metric."""
    lo, hi = bootstrap_ci(values, n=n_boot)
    return {
        'mean':    round(safe_mean(values), 6),
        'std':     round(safe_std(values),  6),
        'ci95_lo': round(lo, 6),
        'ci95_hi': round(hi, 6),
    }


def agg_memory(rows: list[dict], label: str, n_boot: int) -> dict:
    """Aggregate corpus-level metrics for Task 1 or Task 2.

    Flattens per-session intermediate lists and computes
    correlation / kappa statistics across all matched pairs.
    Returns a dict for the JSON summary and also prints to stdout.
    """
    f1s    = [r['memory_set_f1']         for r in rows]
    rouges = [r['matched_memory_rougeL'] for r in rows]
    berts  = [r['matched_memory_bertscore'] for r in rows]
    diffs  = [r['memory_count_diff']     for r in rows]

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

    # ── print ──
    print(f'\n{"="*60}')
    print(f'  {label}  ({len(rows)} sessions, {total_matched} matched pairs)')
    print(f'{"="*60}')
    print(f'  memory_set_f1            : {safe_mean(f1s):.4f}  '
          f'std={safe_std(f1s):.4f}  '
          f'95%CI=[{bootstrap_ci(f1s, n_boot)[0]:.4f}, {bootstrap_ci(f1s, n_boot)[1]:.4f}]')
    print(f'  matched_memory_rougeL    : {safe_mean(rouges):.4f}'
          '  [upward-biased: semantic matches ≥ 0.5 only]')
    if not all(math.isnan(x) for x in berts):
        print(f'  matched_memory_bertscore : {safe_mean(berts):.4f}')
    print(f'  memory_count_diff (mean) : {safe_mean(diffs):.3f}')
    print(f'  --- confidence ({len(valid_conf)} pairs) ---')
    if ca:
        print(f'  confidence_pearson       : {safe_pearson(list(ca), list(cb)):.4f}')
        print(f'  confidence_spearman      : {safe_spearman(list(ca), list(cb)):.4f}')
        print(f'  confidence_kendall_tau   : {safe_kendall(list(ca), list(cb)):.4f}')
        print(f'  confidence_MAE           : {safe_mae(list(ca), list(cb)):.4f}')
    print(f'  --- attribute kappa ({total_matched} pairs) ---')
    print(f'  type_kappa               : {safe_kappa(type_a,  type_b):.4f}')
    print(f'  label_kappa              : {safe_kappa(label_a, label_b):.4f}')
    print(f'  time_scope_kappa         : {safe_kappa(scope_a, scope_b):.4f}')

    # ── return dict for JSON ──
    result: dict = {
        'n_sessions':    len(rows),
        'matched_pairs': total_matched,
        'memory_set_f1':             _stat(f1s, n_boot),
        'matched_memory_rougeL':     {**_stat(rouges, n_boot),
                                      'note': 'upward-biased: only pairs with ROUGE-L >= 0.5 are matched'},
        'memory_count_diff':         _stat(diffs, n_boot),
    }
    if not all(math.isnan(x) for x in berts):
        result['matched_memory_bertscore'] = _stat(berts, n_boot)
    if ca:
        result['confidence'] = {
            'n_pairs':    len(valid_conf),
            'pearson':    round(safe_pearson(list(ca), list(cb)), 6),
            'spearman':   round(safe_spearman(list(ca), list(cb)), 6),
            'kendall_tau': round(safe_kendall(list(ca), list(cb)), 6),
            'MAE':        round(safe_mae(list(ca), list(cb)), 6),
        }
    result['attribute_kappa'] = {
        'type':       round(safe_kappa(type_a,  type_b), 6),
        'label':      round(safe_kappa(label_a, label_b), 6),
        'time_scope': round(safe_kappa(scope_a, scope_b), 6),
    }
    return result


def agg_task3(rows: list[dict], n_boot: int) -> dict:
    """Aggregate Task 3 corpus-level metrics."""
    pf1  = [r['query_memory_pair_f1']      for r in rows]
    agr  = [r['selected_memory_agreement'] for r in rows]
    rl   = [r['selected_memory_rougeL']    for r in rows]
    bers = [r['selected_memory_bertscore'] for r in rows]

    print(f'\n{"="*60}')
    print(f'  TASK 3  ({len(rows)} sessions)')
    print(f'{"="*60}')
    print(f'  query_memory_pair_f1       : {safe_mean(pf1):.4f}  '
          f'95%CI=[{bootstrap_ci(pf1, n_boot)[0]:.4f}, {bootstrap_ci(pf1, n_boot)[1]:.4f}]')
    print(f'  selected_memory_agreement  : {safe_mean(agr):.4f}')
    print(f'  selected_memory_rougeL     : {safe_mean(rl):.4f}')
    if not all(math.isnan(x) for x in bers):
        print(f'  selected_memory_bertscore  : {safe_mean(bers):.4f}')

    result: dict = {
        'n_sessions': len(rows),
        'query_memory_pair_f1':      _stat(pf1, n_boot),
        'selected_memory_agreement': _stat(agr, n_boot),
        'selected_memory_rougeL':    _stat(rl,  n_boot),
    }
    if not all(math.isnan(x) for x in bers):
        result['selected_memory_bertscore'] = _stat(bers, n_boot)
    return result


def agg_task4(rows: list[dict], n_boot: int) -> dict:
    """Aggregate Task 4 corpus-level metrics."""
    qf1  = [r['query_f1']             for r in rows]
    qrl  = [r['query_rougeL']         for r in rows]
    smf1 = [r['supporting_memory_f1'] for r in rows]
    total_matched = sum(r['matched_query_count'] for r in rows)

    print(f'\n{"="*60}')
    print(f'  TASK 4  ({len(rows)} sessions, {total_matched} matched query pairs)')
    print(f'{"="*60}')
    print(f'  query_f1             : {safe_mean(qf1):.4f}  std={safe_std(qf1):.4f}  '
          f'95%CI=[{bootstrap_ci(qf1, n_boot)[0]:.4f}, {bootstrap_ci(qf1, n_boot)[1]:.4f}]')
    print(f'  query_rougeL         : {safe_mean(qrl):.4f}')
    print(f'  supporting_memory_f1 : {safe_mean(smf1):.4f}')

    return {
        'n_sessions':    len(rows),
        'matched_query_pairs': total_matched,
        'query_f1':             _stat(qf1,  n_boot),
        'query_rougeL':         _stat(qrl,  n_boot),
        'supporting_memory_f1': _stat(smf1, n_boot),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load batches, sample sessions, compute metrics, print + optionally save JSON.

    Workflow:
      1. Load batches 1–4 (paired); batch 5 skipped (no second annotator).
      2. Collect all sessions present in both annotators for each batch.
         Sessions missing individual tasks are NOT dropped globally — each
         task is evaluated only on sessions that have that task in both files.
      3. Sample SAMPLE_SIZE sessions from the full paired pool.
      4. For each task, compute per-session metrics (skip + log if task absent).
      5. Aggregate, print summary, optionally write JSON.
    """
    args        = parse_args()
    use_bert    = args.bertscore == 'on'
    n_boot      = args.bootstrap_n

    if use_bert and not _BERTSCORE_AVAILABLE:
        print('ERROR: --bertscore on requested but bert-score is not installed.',
              file=sys.stderr)
        print('Run:  pip install bert-score', file=sys.stderr)
        sys.exit(1)

    # ── 1. load all batches ───────────────────────────────────────────────
    all_pairs = []  # (session_A, session_B, batch_idx)
    batch_info = []

    for batch_idx in range(1, 6):
        s1, s2 = load_batch(batch_idx)
        if not s1 or not s2:
            print(f'Batch {batch_idx}: skipped (no paired annotations)')
            continue
        shared = sorted(set(s1) & set(s2))
        for sid in shared:
            all_pairs.append((s1[sid], s2[sid], batch_idx))
        batch_info.append({'batch': batch_idx, 'sessions': len(shared)})

    total = len(all_pairs)
    print(f'\nPaired sessions available: {total}')
    for bi in batch_info:
        print(f'  Batch {bi["batch"]}: {bi["sessions"]} sessions')

    # ── 2. random sample ──────────────────────────────────────────────────
    rng     = random.Random(args.seed)
    sampled = rng.sample(all_pairs, min(args.sample_size, total))
    dist    = Counter(b for _, _, b in sampled)
    print(f'\nSampled {len(sampled)} sessions (seed={args.seed})')
    for b, cnt in sorted(dist.items()):
        print(f'  Batch {b}: {cnt} sessions')

    # ── 3. per-session metrics (per-task filtering) ───────────────────────
    t1_rows, t2_rows, t3_rows, t4_rows = [], [], [], []
    per_session_out = []
    skipped = Counter()

    for a, b, batch in sampled:
        sess_out: dict[str, Any] = {
            'sessionId': a['sessionId'],
            'batch':     batch,
        }

        # Task 1
        m1a, m1b = get_mems(a, 'task1'), get_mems(b, 'task1')
        if get_ann(a, 'task1') is not None and get_ann(b, 'task1') is not None:
            r1 = memory_metrics(m1a, m1b, use_bert)
            t1_rows.append(r1)
            sess_out['task1'] = {k: v for k, v in r1.items() if not k.startswith('_')}
        else:
            skipped['task1'] += 1
            sess_out['task1'] = 'skipped'

        # Task 2
        m2a, m2b = get_mems(a, 'task2'), get_mems(b, 'task2')
        if get_ann(a, 'task2') is not None and get_ann(b, 'task2') is not None:
            r2 = memory_metrics(m2a, m2b, use_bert)
            t2_rows.append(r2)
            sess_out['task2'] = {k: v for k, v in r2.items() if not k.startswith('_')}
        else:
            skipped['task2'] += 1
            sess_out['task2'] = 'skipped'

        # Task 3
        a3, b3 = get_ann(a, 'task3'), get_ann(b, 'task3')
        if a3 is not None and b3 is not None:
            r3 = task3_metrics(a3, b3, use_bert)
            t3_rows.append(r3)
            sess_out['task3'] = r3
        else:
            skipped['task3'] += 1
            sess_out['task3'] = 'skipped'

        # Task 4
        a4, b4 = get_ann(a, 'task4'), get_ann(b, 'task4')
        if a4 is not None and b4 is not None:
            r4 = task4_metrics(a4, b4)
            t4_rows.append(r4)
            sess_out['task4'] = r4
        else:
            skipped['task4'] += 1
            sess_out['task4'] = 'skipped'

        per_session_out.append(sess_out)

    if skipped:
        print(f'\nSkipped (task absent in one annotator): {dict(skipped)}')

    # ── 4. aggregate and print ────────────────────────────────────────────
    print('\n' + '#' * 60)
    print(f'  Q1 BENCHMARK CONSTRUCTION EVALUATION')
    print(f'  Sample: {len(sampled)} sessions  seed={args.seed}  bertscore={args.bertscore}')
    print('#' * 60)

    summary = {
        'settings': {
            'sample_size': len(sampled),
            'seed':        args.seed,
            'bertscore':   args.bertscore,
            'rouge_threshold': ROUGE_THRESHOLD,
            'bootstrap_n': n_boot,
        },
        'skipped_per_task': dict(skipped),
        'task1': agg_memory(t1_rows, 'TASK 1  (gold memory extraction)', n_boot),
        'task2': agg_memory(t2_rows, 'TASK 2  (memory update)',          n_boot),
        'task3': agg_task3(t3_rows, n_boot),
        'task4': agg_task4(t4_rows, n_boot),
        'per_session': per_session_out,
    }

    print('\n' + '#' * 60)
    print('  DONE')
    print('#' * 60)

    # ── 5. optional JSON output ───────────────────────────────────────────
    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        print(f'\nResults saved to: {args.output_json}')


if __name__ == '__main__':
    main()
