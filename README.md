# Alps Annotation Analysis

Inter-annotator agreement evaluation for the Alps benchmark (Q1) and
LLM-as-judge reliability analysis (Q2).

## Setup

```bash
conda activate extraction_evermemos
pip install -r requirements.txt
```

To use BERTScore (optional, slow on CPU):
```bash
pip install bert-score
```

## Running Q1 evaluation

```bash
# default: 1000 sessions, seed=42, BERTScore off, print to stdout
python evaluate_annotation_consistency.py

# save machine-readable results
python evaluate_annotation_consistency.py --output-json results.json

# with BERTScore (requires bert-score installed)
python evaluate_annotation_consistency.py --bertscore on --output-json results_bert.json

# custom sample size / seed
python evaluate_annotation_consistency.py --sample-size 500 --seed 0
```

## Contents

| Path | Description |
|------|-------------|
| `1/` – `4/` | Paired annotation bundles (`annotation_data_1.json` + `annotation_data_2.json`) |
| `5/` | Single-annotator batch (no pair; excluded from Q1 agreement analysis) |
| `evaluate_annotation_consistency.py` | Q1 evaluation script |
| `analysis.md` | Q1 + Q2 results report |
| `outline.md` | Evaluation design specification |
| `llm-as-judge/` | Q2 data and analysis inputs |
| `requirements.txt` | Python dependencies |

## Data notes

- **Batches 1–4**: each has two independent annotators — used for Q1 inter-annotator agreement.
- **Batch 5**: single annotator only — excluded from Q1, available for the final merged dataset.
- **Q2**: uses `llm-as-judge/manual_check_data-merged-dataset-*.json` (human reviewer labels on judge outputs); computed separately, not by this script.

## Reproducibility

All Q1 results in `analysis.md` are produced by running the script with default arguments (sample=1000, seed=42, BERTScore off). The numbers are stable across runs with the same seed.
