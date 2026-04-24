# Benchmark Evaluation Report — Q1 & Q2

## Overview

**Goal:** Measure whether the benchmark is stable across human annotators — if two annotators label the same session independently, do they construct similar benchmark data?

**Data:** Random sample of 1000 sessions drawn from 1802 paired sessions across 4 annotator pairs (batches 1–4), seed=42. Batch 5 excluded (single annotator only, no pair for agreement computation).
| **Total** | | **1000** |

**Matching strategy:** Memories matched first by exact normalized text, then by greedy ROUGE-L (threshold 0.5). Task 4 sub-annotations matched by `queryId`.

---

## Task 1 — Gold Memory Extraction

**Sample size:** 1000 sessions, 2178 matched memory pairs

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **memory_set_f1** | **0.752** (std 0.327) | Match memories A↔B via exact text then greedy ROUGE-L; F1 = 2·P·R/(P+R) where P = matched/\|A\|, R = matched/\|B\| | **Memory selection agreement** — did both annotators pick the same set of memories? |
| matched_memory_rougeL | 0.978 | ROUGE-L F1 on the `value` text of each matched pair; averaged across all pairs | **Memory content similarity** — for memories they both selected, how similar is the wording? |
| memory_count_diff | 0.607 | \|len(A) − len(B)\|; averaged across sessions | **Memory quantity gap** — how many more/fewer memories one annotator extracted |
| confidence_pearson | 0.793 | Pearson r on confidence floats of matched pairs | **Confidence score agreement (linear)** — do annotators assign similar numeric confidence? |
| confidence_spearman | **0.869** | Spearman ρ; rank-based | **Confidence score agreement (rank)** — preferred here given discrete values (0.5/0.7/0.8/0.9) |
| confidence_kendall_tau | 0.824 | Kendall τ; concordant vs discordant pairs | **Confidence score agreement (rank, conservative)** |
| confidence_MAE | 0.042 | Mean \|conf_A − conf_B\| across matched pairs | **Confidence score gap** — practical magnitude of disagreement |
| type_kappa | **0.933** | Cohen's κ on `type` (direct/indirect) for matched pairs | **Memory type classification agreement** |
| label_kappa | **0.861** | Cohen's κ on `label` (taxonomy category) for matched pairs | **Memory category classification agreement** |
| time_scope_kappa | 0.721 | Cohen's κ on `time_scope` (long\_term/short\_term/unknown) for matched pairs | **Memory time scope classification agreement** |

---

## Task 2 — Memory Update

**Sample size:** 1000 sessions, 3460 matched memory pairs

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **memory_set_f1** | **0.841** (std 0.155) | Same as Task 1 F1 on `editableUpdatedMemories` | **Update selection agreement** — did both annotators update the same memories? |
| matched_memory_rougeL | 0.975 | ROUGE-L F1 on matched updated-memory `value` pairs | **Updated memory content similarity** |
| memory_count_diff | 1.046 | \|len(A) − len(B)\| on updated memory lists | **Update quantity gap** |
| confidence_pearson | 0.737 | Pearson r on confidence of matched updated-memory pairs | **Post-update confidence agreement (linear)** |
| confidence_spearman | **0.733** | Spearman ρ | **Post-update confidence agreement (rank)** |
| confidence_kendall_tau | 0.651 | Kendall τ | **Post-update confidence agreement (rank, conservative)** |
| confidence_MAE | 0.051 | Mean \|conf_A − conf_B\| | **Post-update confidence gap** |
| type_kappa | **0.904** | Cohen's κ on `type` for matched updated-memory pairs | **Updated memory type classification agreement** |
| label_kappa | **0.841** | Cohen's κ on `label` for matched updated-memory pairs | **Updated memory category classification agreement** |
| time_scope_kappa | 0.731 | Cohen's κ on `time_scope` for matched updated-memory pairs | **Updated memory time scope classification agreement** |

---

## Task 3 — Query-Memory Linkage

**Sample size:** 1000 sessions

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **query_memory_pair_f1** | **0.837** | Exact-match F1 on `(queryText, selectedMemory)` pairs across 1000 sessions | **Query-memory linkage agreement** — did both annotators pair the same query with the same memory? |
| selected_memory_agreement | 0.867 | Fraction of sessions where normalized selected memory is identical | **Memory selection agreement** — did they pick the same memory for the query? |
| selected_memory_rougeL | 0.909 | ROUGE-L F1 on selected memory `value` per session | **Selected memory content similarity** — when they picked differently, how close were the texts? |

---

## Task 4 — Multi-Query Construction

**Sample size:** 1000 sessions, 1097 matched query pairs

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **query_f1** | **0.925** (std 0.260) | Set F1 on normalized query texts per session; averaged | **Query set agreement** — did both annotators construct the same queries? |
| query_rougeL | 0.749 | ROUGE-L F1 on `queryId`-matched query text pairs | **Query content similarity** — for matched queries, how similar is the wording? |
| supporting_memory_f1 | 0.815 | Exact-match F1 on selected memory for each matched query pair | **Supporting memory agreement** — for the same query, did they pick the same memory? |

---

## Key Observations

### 1. Content agreement is strong across all tasks
The headline F1 metrics (Task 1: 0.752, Task 2: 0.841, Task 3: 0.837, Task 4: 0.925) show that annotators consistently select similar memory sets and construct similar query sets. Task 2 and Task 4 show higher agreement than Task 1, suggesting that memory *update* and query *construction* are more constrained than free-form memory *extraction*.

### 2. Attribute classification agreement is even stronger than content agreement
Type kappa (0.90–0.93) and label kappa (0.84–0.86) substantially exceed the content F1 scores. This means annotators don't just extract similar memories — they also classify them the same way. This is a stronger reliability claim than content overlap alone.

### 3. Task 1 has higher variance than Task 2 (std 0.327 vs 0.155)
Memory *extraction* is inherently more subjective — there is no fixed set of memories to choose from. Memory *update* is more constrained by the existing memory list, which explains the lower variance and higher F1.

---

# Q2 LLM-as-Judge Reliability Evaluation — Analysis Report

## Overview

**Goal:** Measure whether the LLM judge agrees with human judgment — if a human reviewer checks the judge's output for each session, how often do they say the judge was correct?

**Data:** `manual_check_data` — 554 Q2 annotations across all 4 tasks. A human reviewer examined the judge's output for each session and labelled it `aligned`, `partially_aligned`, or `not_aligned`.

| Task | Sessions | Models evaluated |
|------|----------|-----------------|
| task1 | 140 | 7 (20 sessions each) |
| task2 | 139 | 7 |
| task3 | 139 | 7 |
| task4 | 136 (232 sub-queries) | mixed |

**Models judged:** claude-sonnet-4-5, deepseek-reasoner, gemini-3-flash-preview, gpt-4.1-mini, gpt-5.2, llama-4-maverick, qwen3-max

**Metric design:** Three levels of human judgment are available:
- **Session-level** `alignmentVerdict` (aligned / partially_aligned / not_aligned) — overall verdict on judge quality per session
- **Item-level** `humanVerdict` (supported / not_supported) — per individual judge decision within a session (task1/2 only)
- **Flag-level** — 4 binary dimensions of judge quality (task3 only)

---

## Task 1 — Gold Memory Extraction Judge (140 sessions)

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **alignment_accuracy** | **0.729** | % sessions where human labelled judge as `aligned` | **Overall judge correctness** — did the judge's verdict match human judgment? |
| weighted_alignment_rate | 0.807 | (aligned×1 + partial×0.5 + not_aligned×0) / total | **Partial-credit accuracy** — gives half-credit for partially correct judge outputs |
| misalignment_rate | 0.114 | % sessions labelled `not_aligned` | **Judge failure rate** — fraction of sessions where judge was clearly wrong |
| item_pair_support_rate | **1.000** | % of pair-review decisions where human said `supported` (177/177) | **Item-level correctness: memory matching** — did human agree with each individual judge match decision? |
| item_missing_support_rate | **0.995** | % of missing-review decisions human supported (197/198) | **Item-level correctness: missed memories** — did human agree the judge correctly flagged missing memories? |
| item_extra_support_rate | **1.000** | % of extra-review decisions human supported (231/231) | **Item-level correctness: hallucinated memories** — did human agree the judge correctly flagged extra memories? |

---

## Task 2 — Memory Update Judge (139 sessions)

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **alignment_accuracy** | **0.849** | % sessions labelled `aligned` | **Overall judge correctness for memory updates** |
| weighted_alignment_rate | 0.903 | (aligned×1 + partial×0.5 + not_aligned×0) / total | **Partial-credit accuracy** |
| misalignment_rate | 0.043 | % sessions labelled `not_aligned` | **Judge failure rate** |
| item_pair_support_rate | **0.997** | % of pair-review decisions supported (386/387) | **Item-level correctness: memory update matching** |
| item_missing_support_rate | **1.000** | % of missing-review decisions supported (190/190) | **Item-level correctness: missed updates** |
| item_extra_support_rate | **1.000** | % of extra-review decisions supported (80/80) | **Item-level correctness: spurious updates** |

---

## Task 3 — Query-Memory Linkage Judge (139 sessions)

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **alignment_accuracy** | **0.964** | % sessions labelled `aligned` (134/139) | **Overall judge correctness for query-memory scoring** |
| weighted_alignment_rate | 0.968 | (aligned×1 + partial×0.5 + not_aligned×0) / total | **Partial-credit accuracy** |
| misalignment_rate | 0.029 | % sessions labelled `not_aligned` (4/139) | **Judge failure rate** |
| usedMemoryCorrect_rate | **0.986** | % sessions where human confirmed judge identified the correct memory (137/139) | **Memory identification accuracy** — did the judge reference the right memory? |
| scoreReasonable_rate | **0.971** | % sessions where human confirmed the judge's score was reasonable (135/139) | **Score validity** — was the judge's numeric score justified? |
| reasonSupportsJudgment_rate | **0.978** | % sessions where judge's reasoning supported its score (136/139) | **Reasoning coherence** — was the judge's explanation consistent with its verdict? |
| scoreConsistentWithUsedMemory_rate | **1.000** | % sessions where score was consistent with the memory used (139/139) | **Internal consistency** — was the score consistent with the judge's own memory citation? |

---

## Task 4 — Multi-Query Judge (136 sessions, 232 sub-queries)

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **query_alignment_accuracy** | **0.828** | % sub-queries labelled `aligned` (192/232) | **Per-query judge correctness** — did the judge correctly evaluate each individual query-response pair? |
| query_weighted_alignment | 0.888 | (aligned×1 + partial×0.5 + not_aligned×0) / total sub-queries | **Per-query partial-credit accuracy** |
| query_misalignment_rate | 0.052 | % sub-queries labelled `not_aligned` (12/232) | **Per-query judge failure rate** |
| session_alignment_accuracy | **0.779** | % sessions where all sub-queries were `aligned` (106/136) | **Session-level judge correctness** — did the judge get every query right within a session? |
| session_partial_rate | 0.162 | % sessions with ≥1 partially_aligned or mix (22/136) | **Session-level partial correctness** |
| session_misalignment_rate | 0.059 | % sessions with ≥1 not_aligned sub-query (8/136) | **Session-level judge failure rate** |

---

## Key Observations

### 1. Judge performs best on Task 3, weakest on Task 1
Alignment accuracy increases from Task 1 (0.729) → Task 4 (0.828) → Task 2 (0.849) → Task 3 (0.964). Task 3 (query-memory linkage scoring) is the most well-defined task for the judge — there is a clear right answer. Task 1 (memory extraction evaluation) is the most open-ended, and the judge struggles more with deciding what counts as a valid memory.

### 2. Item-level agreement is near-perfect even when session-level alignment is not
For tasks 1 and 2, item-level support rates are 0.995–1.000 across all review kinds (pair/missing/extra). This means the judge's individual decisions are almost always correct — the `partially_aligned` and `not_aligned` session-level verdicts arise from the judge **missing** some errors or making a borderline overall call, not from individual decisions being wrong.

### 3. Task 4 session-level accuracy (0.779) is lower than query-level (0.828)
A session fails the session-level check if even one sub-query is not fully aligned. Since each session has ~1.7 sub-queries on average, the probability of all passing is lower than any single one passing, which explains the gap.
