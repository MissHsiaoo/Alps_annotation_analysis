# Annotation Comparison Evaluation Pipeline Outline

## Goal

Implement a Python evaluation script to compare two annotation files for the same benchmark split:

- `annotation_A.json`
- `annotation_B.json`

The script is intended to measure **annotation stability** and **benchmark construction reliability**.

The comparison is between **two human annotation versions**, not between model outputs and gold labels.


## Scope

The evaluation is divided into four task-specific modules:

- `task1`: extracted gold memories
- `task2`: updated memories
- `task3`: query-selected memory pair consistency
- `task4`: multi-query consistency + query-level similarity + selected-memory consistency

This pipeline will use the actual schema observed in the dataset, not a generic QA schema.


## Dataset Assumptions

Each sample is a session-level record with task entries under:

- `task1.annotation.editableGoldMemories`
- `task2.annotation.editableUpdatedMemories`
- `task3.annotation.queryText`
- `task3.annotation.editableSelectedMemory`
- `task4.annotation.subAnnotations[]`

Expected task semantics:

- `task1`
  - compare extracted memory sets
- `task2`
  - compare updated memory sets
- `task3`
  - compare `(query, selected_memory)` pairs
- `task4`
  - compare query sets and each query's linked selected memory


## Evaluation Targets

### Task1-2

Primary goal:

- whether the memory annotations are consistent across annotators

Main metrics:

- `memory_set_f1`
- `matched_memory_bertscore`
- `confidence_pearson`

Auxiliary metrics:

- `matched_memory_rougeL`
- `memory_count_diff`

### Task3

Primary goal:

- whether the annotators produce the same query-memory linkage

Main metrics:

- `query_memory_pair_f1`
- `selected_memory_agreement`
- `selected_memory_bertscore`

Auxiliary metrics:

- `selected_memory_rougeL`

### Task4

Primary goal:

- whether the annotators construct the same query set
- whether matched queries point to the same supporting memory

Main metrics:

- `query_f1`
- `query_bertscore`
- `query_rougeL`
- `supporting_memory_f1`


## Text Normalization

All text comparisons will use normalized text.

Normalization steps:

- lowercase
- strip leading/trailing whitespace
- collapse repeated whitespace
- optionally normalize Unicode if needed

For metric computation:

- set-based matching uses normalized text
- BERTScore and ROUGE use original text or lightly cleaned text


## Matching Strategy

### Memory Matching for Task1-2

Function:

- `match_memories(memories_a, memories_b)`

Matching order:

1. exact normalized-value match
2. for unmatched items, compute pairwise semantic similarity
3. greedy matching by highest similarity

Notes:

- default semantic score: BERTScore F1
- optional fallback: ROUGE-L when BERTScore is unavailable
- configurable similarity threshold to avoid low-quality forced matches

Output:

- matched pairs
- unmatched A memories
- unmatched B memories

Possible future improvement:

- Hungarian matching instead of greedy


## Metric Definitions

### Task1-2 Metrics

#### 1. memory_set_f1

Compare the two memory sets after normalization and matching.

Definitions:

- `precision = matched_count / len(memories_A)`
- `recall = matched_count / len(memories_B)`
- `f1 = harmonic_mean(precision, recall)`

This is the main structural stability metric.

#### 2. matched_memory_bertscore

For each matched memory pair:

- compare memory `value`
- compute BERTScore F1

Aggregate:

- average across matched pairs

#### 3. matched_memory_rougeL

For each matched memory pair:

- compute ROUGE-L F1 on memory `value`

Aggregate:

- average across matched pairs

#### 4. confidence_pearson

For matched memory pairs:

- extract `confidence` from A and B
- compute Pearson correlation

Behavior:

- if fewer than 2 valid pairs, return `NaN`

#### 5. memory_count_diff

Simple count difference:

- `abs(len(memories_A) - len(memories_B))`


### Task3 Metrics

Task3 is treated as a single query-memory pair per sample.

#### 1. query_memory_pair_f1

Represent each sample as a pair:

- `(normalized_query_text, normalized_selected_memory_value)`

For dataset-level evaluation:

- compare pair sets between A and B

#### 2. selected_memory_agreement

Per sample:

- exact agreement on normalized selected memory content

Aggregate:

- average exact match rate

#### 3. selected_memory_bertscore

Per sample:

- BERTScore F1 between selected memory values

Aggregate:

- average over valid samples

#### 4. selected_memory_rougeL

Per sample:

- ROUGE-L F1 between selected memory values

Aggregate:

- average over valid samples


### Task4 Metrics

Task4 contains multiple subqueries under `subAnnotations`.

#### 1. query_f1

Compare query sets between A and B.

Default query unit:

- `queryId` if stable

Fallback if query IDs are unstable:

- `(ability, normalized_query_text)`

#### 2. query_bertscore

For matched query pairs:

- compute BERTScore F1 between query texts

Aggregate:

- average across matched queries

#### 3. query_rougeL

For matched query pairs:

- compute ROUGE-L F1 between query texts

Aggregate:

- average across matched queries

#### 4. supporting_memory_f1

For matched query pairs:

- compare selected memory values or memory IDs

Current dataset assumption:

- one selected memory per subquery

So this becomes:

- exact or matched selected-memory agreement aggregated as F1

If future schema contains multiple supporting memories:

- extend to set-based memory F1


## Output Format

The script should return a dictionary like:

```python
{
  "task1": {
    "memory_set_f1": float,
    "matched_memory_bertscore": float,
    "matched_memory_rougeL": float,
    "confidence_pearson": float,
    "memory_count_diff": float
  },
  "task2": {
    "memory_set_f1": float,
    "matched_memory_bertscore": float,
    "matched_memory_rougeL": float,
    "confidence_pearson": float,
    "memory_count_diff": float
  },
  "task3": {
    "query_memory_pair_f1": float,
    "selected_memory_agreement": float,
    "selected_memory_bertscore": float,
    "selected_memory_rougeL": float
  },
  "task4": {
    "query_f1": float,
    "query_bertscore": float,
    "query_rougeL": float,
    "supporting_memory_f1": float
  }
}
```

The script should also print detailed per-sample diagnostics.


## Required Functions

### `normalize_text(text: str) -> str`

Purpose:

- normalize text for set comparison and matching

### `extract_task_payload(sample: dict, task_name: str) -> dict | None`

Purpose:

- extract the task entry from a session-level sample

### `match_memories(memories_a: list[dict], memories_b: list[dict]) -> dict`

Purpose:

- align memory items between A and B

Returns:

- matched pairs
- unmatched items
- matching summary

### `compute_memory_metrics(memories_a: list[dict], memories_b: list[dict]) -> dict`

Used for:

- task1
- task2

### `compute_task3_metrics(sample_a: dict, sample_b: dict) -> dict`

Used for:

- task3

### `compute_task4_metrics(sample_a: dict, sample_b: dict) -> dict`

Used for:

- task4

### `aggregate_metrics(per_sample_results: list[dict]) -> dict`

Purpose:

- macro-average results across samples

### `main()`

Responsibilities:

- load A and B annotation files
- align samples by session ID
- evaluate task1-task4
- print per-sample details
- print final aggregated results


## Alignment Rules

### Sample-Level Alignment

Align samples by:

- `sessionId`

For task4 subqueries:

- align by `queryId` first
- fallback to normalized query text if needed


## Edge Cases

The implementation must handle:

- missing task in one file
- empty memory list
- empty query list
- unmatched memories
- missing confidence
- Pearson with too few valid points
- ROUGE/BERTScore on empty strings
- division by zero in precision/recall/F1

Rules:

- return `NaN` when a metric is undefined
- do not crash on partial samples
- record skipped cases in per-sample logs


## Efficiency Notes

To keep runtime acceptable:

- batch BERTScore calls whenever possible
- avoid repeated computation for the same text pairs
- compute pairwise similarity only for unmatched candidates
- separate exact-match stage from semantic-match stage

Potential optimization:

- cache normalized text
- cache pairwise similarity scores


## Per-Sample Reporting

Each sample report should include:

- `sessionId`
- whether each task exists in A and B
- task1/task2 matched memory count
- task1/task2 unmatched count
- task3 selected memory exact match
- task4 matched query count
- task4 unmatched query count
- all per-task metric values

This report will help inspect disagreement cases manually.


## Recommended Implementation Order

1. implement data loading and sample alignment
2. implement text normalization helpers
3. implement exact-match memory matching
4. add semantic matching with BERTScore
5. implement task1/task2 metrics
6. implement task3 metrics
7. implement task4 metrics
8. implement aggregation
9. add detailed printing and JSON-friendly output


## Open Questions Before Coding

These should be confirmed before the final implementation:

1. whether task4 query alignment should trust `queryId` fully
2. whether memory matching should use greedy or Hungarian by default
3. whether `confidence` is the intended proxy for memory importance
4. whether final reporting should be macro-average or weighted-average
5. whether to output both overall summary and per-task JSON files


## Proposed Default Decisions

If not otherwise specified, the implementation will use:

- sample alignment by `sessionId`
- task4 subquery alignment by `queryId`
- memory matching = exact match first, then greedy BERTScore matching
- `confidence` as the numeric score for Pearson
- macro-average aggregation across samples
- console print + final summary dictionary output
