# Q1 Benchmark Construction Evaluation — Analysis Report

## Overview

**Goal:** Measure whether the benchmark is stable across human annotators — if two annotators label the same session independently, do they construct similar benchmark data?

**Data:** 1802 paired sessions across 4 annotator pairs (batches 1–4). Batch 5 excluded (missing second annotator file).

| Batch | Annotator Pair | Sessions |
|-------|---------------|----------|
| 1 | 贾子龙 & 田睿 | 451 |
| 2 | 索子骁 & qyc | 445 |
| 3 | 王鼎皓 & Achiles | 455 |
| 4 | 杨奕麟 & 高杰 | 451 |
| **Total** | | **1802** |

**Matching strategy:** Memories matched first by exact normalized text, then by greedy ROUGE-L (threshold 0.5). Task 4 sub-annotations matched by `queryId`.

---

## Task 1 — Gold Memory Extraction

**Sample size:** 1802 sessions, 4109 matched memory pairs

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **memory_set_f1** | **0.764** (std 0.319) | Match memories A↔B via exact text then greedy ROUGE-L; F1 = 2·P·R/(P+R) where P = matched/\|A\|, R = matched/\|B\| | **Memory selection agreement** — did both annotators pick the same set of memories? |
| matched_memory_rougeL | 0.979 | ROUGE-L F1 on the `value` text of each matched pair; averaged across all pairs | **Memory content similarity** — for memories they both selected, how similar is the wording? |
| memory_count_diff | 0.615 | \|len(A) − len(B)\|; averaged across sessions | **Memory quantity gap** — how many more/fewer memories one annotator extracted |
| confidence_pearson | 0.647 | Pearson r on confidence floats of matched pairs | **Confidence score agreement (linear)** — do annotators assign similar numeric confidence? |
| confidence_spearman | **0.861** | Spearman ρ; rank-based | **Confidence score agreement (rank)** — preferred here given discrete values (0.5/0.7/0.8/0.9) |
| confidence_kendall_tau | 0.818 | Kendall τ; concordant vs discordant pairs | **Confidence score agreement (rank, conservative)** |
| confidence_MAE | 0.044 | Mean \|conf_A − conf_B\| across matched pairs | **Confidence score gap** — practical magnitude of disagreement |
| type_kappa | **0.935** | Cohen's κ on `type` (direct/indirect) for matched pairs | **Memory type classification agreement** |
| label_kappa | **0.867** | Cohen's κ on `label` (taxonomy category) for matched pairs | **Memory category classification agreement** |
| time_scope_kappa | 0.730 | Cohen's κ on `time_scope` (long\_term/short\_term/unknown) for matched pairs | **Memory time scope classification agreement** |

---

## Task 2 — Memory Update

**Sample size:** 1802 sessions, 6412 matched memory pairs

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **memory_set_f1** | **0.843** (std 0.158) | Same as Task 1 F1 on `editableUpdatedMemories` | **Update selection agreement** — did both annotators update the same memories? |
| matched_memory_rougeL | 0.974 | ROUGE-L F1 on matched updated-memory `value` pairs | **Updated memory content similarity** |
| memory_count_diff | 1.054 | \|len(A) − len(B)\| on updated memory lists | **Update quantity gap** |
| confidence_pearson | 0.658 | Pearson r on confidence of matched updated-memory pairs | **Post-update confidence agreement (linear)** |
| confidence_spearman | **0.749** | Spearman ρ | **Post-update confidence agreement (rank)** |
| confidence_kendall_tau | 0.670 | Kendall τ | **Post-update confidence agreement (rank, conservative)** |
| confidence_MAE | 0.050 | Mean \|conf_A − conf_B\| | **Post-update confidence gap** |
| type_kappa | **0.909** | Cohen's κ on `type` for matched updated-memory pairs | **Updated memory type classification agreement** |
| label_kappa | **0.846** | Cohen's κ on `label` for matched updated-memory pairs | **Updated memory category classification agreement** |
| time_scope_kappa | 0.744 | Cohen's κ on `time_scope` for matched updated-memory pairs | **Updated memory time scope classification agreement** |

---

## Task 3 — Query-Memory Linkage

**Sample size:** 1802 sessions

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **query_memory_pair_f1** | **0.838** | Exact-match F1 on `(queryText, selectedMemory)` pairs across 1802 sessions | **Query-memory linkage agreement** — did both annotators pair the same query with the same memory? |
| selected_memory_agreement | 0.868 | Fraction of sessions where normalized selected memory is identical | **Memory selection agreement** — did they pick the same memory for the query? |
| selected_memory_rougeL | 0.914 | ROUGE-L F1 on selected memory `value` per session | **Selected memory content similarity** — when they picked differently, how close were the texts? |

---

## Task 4 — Multi-Query Construction

**Sample size:** 1802 sessions, 1996 matched query pairs

| Metric | Value | How calculated | What it measures |
|--------|-------|----------------|-----------------|
| **query_f1** | **0.930** (std 0.252) | Set F1 on normalized query texts per session; averaged | **Query set agreement** — did both annotators construct the same queries? |
| query_rougeL | 0.748 | ROUGE-L F1 on `queryId`-matched query text pairs | **Query content similarity** — for matched queries, how similar is the wording? |
| supporting_memory_f1 | 0.809 | Exact-match F1 on selected memory for each matched query pair | **Supporting memory agreement** — for the same query, did they pick the same memory? |
| overallVerdict_kappa | 0.185 | Cohen's κ on `overallVerdict` (reasonable/unreasonable/partially\_reasonable) | **Query quality judgment agreement** — note: low due to ~95% class imbalance, not genuine disagreement |
| testsTargetAbility_kappa | 0.160 | Cohen's κ on `testsTargetAbility` (yes/partial/no) | **Ability coverage judgment agreement** — same imbalance caveat (~92% yes) |
| memoryDependency_kappa | 0.212 | Cohen's κ on `memoryDependency` (strong/medium/weak) | **Memory dependency judgment agreement** |
| memoryDependency_weighted | 0.161 | Linear-weighted Cohen's κ on `memoryDependency` | **Memory dependency agreement (adjacent errors penalised less)** |
| abilityPurity_kappa | 0.071 | Cohen's κ on `abilityPurity` (high/medium/low) | **Query purity judgment agreement** — near-zero due to ~97% class imbalance, not genuine disagreement |

---

## Key Observations

### 1. Content agreement is strong across all tasks
The headline F1 metrics (Task 1: 0.764, Task 2: 0.843, Task 3: 0.838, Task 4: 0.930) show that annotators consistently select similar memory sets and construct similar query sets. Task 2 and Task 4 show higher agreement than Task 1, suggesting that memory *update* and query *construction* are more constrained than free-form memory *extraction*.

### 2. Attribute classification agreement is even stronger than content agreement
Type kappa (0.91–0.94) and label kappa (0.85–0.87) substantially exceed the content F1 scores. This means annotators don't just extract similar memories — they also classify them the same way. This is a stronger reliability claim than content overlap alone.

### 3. Use Spearman/Kendall for confidence, not Pearson
Confidence values are discrete (0.5, 0.7, 0.8, 0.9) and skewed toward 0.9. Pearson assumes a linear, normally distributed relationship and underestimates agreement (Task 1: 0.647). Spearman (0.861) and Kendall Tau (0.818) are more appropriate and show strong rank-order agreement. The MAE of 0.044–0.050 confirms the practical gap is small.

### 4. Task 4 quality-flag kappas are low due to class imbalance, not disagreement
Cohen's kappa for overallVerdict (0.185), testsTargetAbility (0.160), and abilityPurity (0.071) appear poor by standard thresholds, but this reflects the kappa paradox: when one class dominates (~95% reasonable, ~97% high purity), kappa heavily discounts agreement as "chance". The raw agreement rates are ~94–97%. These fields should be reported with both the kappa and the raw agreement rate to avoid misinterpretation.

### 5. Task 1 has higher variance than Task 2 (std 0.319 vs 0.158)
Memory *extraction* is inherently more subjective — there is no fixed set of memories to choose from. Memory *update* is more constrained by the existing memory list, which explains the lower variance and higher F1.

### 6. Session-level verdict fields are uninformative for agreement analysis
All session-level flags (Task 1–3: `overallVerdict`, `hasDialogueEvidence`, `overInference`, `faithfulToOriginalMeaning`, `relevanceLevel`, etc.) are uniform across all 1802 sessions from both annotators (100% agree, single class). This reflects the cleaning process: sessions that were flagged as unreasonable were removed before this comparison. These fields cannot be used as agreement metrics on the cleaned dataset.
