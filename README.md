## OWID Fossil Share Rebound Tail Risk (Next Year)

Portfolio-ready, Kaggle-style prediction task built from **Our World in Data (OWID)** energy + emissions datasets.

### What you’re predicting
- **Task**: binary classification
- **Predict**: `pred_fossil_share_rebound_next`
- **Target**: `target_fossil_share_rebound_next` — whether fossil dependence rebounds unusually fast over the next year
- **Panelization**: expanded across `segment_id` (multiple fossil/CI views) and `horizon_y`

### Data & evaluation highlights
- **Rows**: train **21,166**, test **3,757**
- **Split**: time-based regime shift (see `dataset_card.md`)
- **Metric**: composite scorer (LogLoss overall + slice LogLoss + (1 − AUPRC)); see `instruction.md`

### Repository contents
- `train.csv`, `test.csv`, `solution.csv`
- `sample_submission.csv`, `perfect_submission.csv`
- `build_dataset.py`, `build_meta.json`
- `score_submission.py`
- `instruction.md`, `golden_workflow.md`, `dataset_card.md`

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

