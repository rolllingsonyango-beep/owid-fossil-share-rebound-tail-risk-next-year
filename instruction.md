## Objective

Predict whether an anonymized entity (a country) will experience an **unusually large rebound in “recarbonization metrics”** over the next \(h\) years.

Each row represents an **entity-year-segment-horizon** at time \(t\): one country-year paired with:

- `segment_id` in {0,1,2} selecting the metric to forecast
- `horizon_y` in {1,2}

Let \(M_s(t)\) be the segment’s metric value:

- `segment_id = 0`: \(M(t)=\) `fossil_share_energy`
- `segment_id = 1`: \(M(t)=\) `fossil_share_elec`
- `segment_id = 2`: \(M(t)=\) `carbon_intensity_elec`

Define:

\[
\Delta M_h(t)=M(t+h)-M(t)
\]

Your label is:

\[
\texttt{target\_fossil\_share\_rebound\_next}=\mathbb{1}\left[\Delta M_h(t)\ge \tau(\text{entity},\text{segment},h)\right]
\]

Where \(\tau(\text{entity},\text{segment},h)\) is the **entity-specific 85th percentile** of \(\Delta M_h\) computed using **training rows only** (within segment+horizon), with a global fallback for entities with limited training history.

Intuition: this is a tail-risk forecast of **worsening decarbonization metrics** under regime shift, not a “which countries use fossil fuels?” lookup.

## Inputs

- **`train.csv`**: includes `row_id`, feature columns, `horizon_y`, and `target_fossil_share_rebound_next`
- **`test.csv`**: includes `row_id` and the same feature columns and `horizon_y`, but **no target**

Anonymization/leakage mitigations:

- `entity_id` is a stable anonymized identifier.
- Exact years are not provided; only coarse `event_era` is exposed.
- Continuous values are released as **bins learned from training rows only** (missing values are encoded as `-1` bins).

## Output

Create a file named **`submission.csv`** with predicted probabilities for each row in `test.csv`.

## Submission format

Your `submission.csv` must contain exactly these columns:

- `row_id`
- `pred_fossil_share_rebound_next`: probability that `target_fossil_share_rebound_next = 1`

Predictions must be numeric and in **[0, 1]**.

## Metric (lower is better)

We use a deterministic composite metric:

\[
\text{Score} = 0.55\cdot \text{LogLoss}_{\text{all}}
             + 0.30\cdot \text{LogLoss}_{\text{hard-transition slice}}
             + 0.15\cdot (1 - \text{AUPRC}_{\text{all}})
\]

- The **hard-transition slice** emphasizes entities that are fossil-heavy and renewables-light at baseline.
- Slice membership is provided as `slice_hard_transition` in both `train.csv` and `test.csv`.

Slice reproducibility:

- With `n_bins=12`, `slice_hard_transition = 1` when:
  - `fossil_share_energy_bin >= 9` and `renewables_share_energy_bin <= 2`.

Deterministic scoring command:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

## Constraints and leakage notes

- The train/test split is **time-based under the hood** (newer eras emphasized in test), but exact years are hidden.
- For honest offline evaluation, avoid random splits. Prefer:
  - an `event_era` holdout, and
  - group-aware checks by `entity_id`.

