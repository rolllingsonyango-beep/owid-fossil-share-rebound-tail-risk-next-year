## Data sanity checks

**Rationale:** Multi-entity OWID data has missingness and coverage changes. A sanity pass prevents “mystery score drops” caused by data handling mistakes.

**Dataset and objective nuances:**

- Verify `test.csv` has **no** `target_fossil_share_rebound_next`.
- Check base rates by `event_era`, by `horizon_y`, and by `slice_hard_transition`.
- Treat `entity_id` as categorical; it is anonymized and not ordered.
- `*_bin == -1` means missing, not low.

## Validation strategy

**Rationale:** Split is time-based. Random splits leak slow-moving structure and overstate performance.

**Dataset and objective nuances:**

- Use an era holdout:
  - train on earlier `event_era` values
  - validate on the newest era available in training
- Evaluate separately on:
  - `horizon_y == 1` vs `== 2`
  - `slice_hard_transition == 1` (30% metric weight)

## Preprocessing plan

**Rationale:** Features are already binned. Most preprocessing is correct encoding + avoiding leakage in any additional transforms.

**Dataset and objective nuances:**

- Treat bins as ordinal:
  - trees can use them directly
  - linear/NN models should one-hot or embed them (`-1` as its own category)
- Keep `missing_count` numeric.
- Encode `horizon_y` explicitly; decision boundaries differ by horizon.

## Baseline model

**Rationale:** Rebound events are threshold-y and interaction-heavy (mix × demand × regime × horizon).

**Dataset and objective nuances:**

- Start with a regularized GBDT and add interactions like:
  - `fossil_share_energy_bin × energy_cons_change_pct_bin × event_era`
  - `renewables_cons_change_pct_bin × fossil_cons_change_pct_bin × horizon_y`
- Then calibrate probabilities (LogLoss-heavy metric).

## Iteration plan

**Rationale:** Metric emphasizes calibration; overconfidence in hard-transition entities is common.

**Dataset and objective nuances:**

- Improve LogLoss first:
  - tune regularization
  - post-hoc calibration (temperature scaling / Platt) on era-aware validation
- Improve AUPRC next:
  - increase capacity cautiously
- Focus on hard-transition slice:
  - this slice can dominate LogLoss when miscalibrated

## Error analysis

**Rationale:** Confident wrong predictions dominate LogLoss.

**Dataset and objective nuances:**

- Slice errors by:
  - `event_era`
  - `horizon_y`
  - `fossil_share_energy_bin`, `renewables_share_energy_bin`
  - `energy_cons_change_pct_bin`
  - `slice_hard_transition`
- If one era is consistently miscalibrated, prefer era-conditional calibration over adding complexity.

## Leaderboard safety

**Rationale:** Test emphasizes newer eras; older patterns can fail out-of-sample.

**Dataset and objective nuances:**

- Avoid random CV.
- Avoid over-using `entity_id` memorization; force performance via interactions and calibration.

## Submission checks

**Rationale:** Formatting errors are avoidable.

**Dataset and objective nuances:**

- `submission.csv` must have exactly `row_id` and `pred_fossil_share_rebound_next`.
- Predictions must be in \([0,1]\), with no NaNs/inf, and ids must match `test.csv`.
- Run:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

