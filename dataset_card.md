## Overview

This competition is built from **Our World in Data (OWID)** public energy datasets. It is reframed as an anonymized tail-event forecasting task: given an entity-year snapshot of energy mix, demand dynamics, and macro context, predict whether the entity will experience an **unusually large rebound** (increase) in a selected **recarbonization metric** over the next \(h\) years (\(h\in\{1,2\}\)).

The task is intentionally challenging because:

- the label is **entity-relative** (85th percentile thresholds computed on training rows only),
- there is strong **non-stationarity** across eras (crises, policy shifts, measurement changes),
- many features have missingness for some entities/years (real coverage),
- the score rewards both **calibration** (LogLoss) and **ranking** (AUPRC), and emphasizes a hard-transition slice.

## Source

- OWID energy repository: `https://github.com/owid/energy-data`
- OWID energy website: `https://ourworldindata.org/energy`

This competition uses a cached copy of `owid-energy-data.csv` already present in this workspace.

## License

CC-BY-4.0

## Features

All continuous signals are converted to **bins learned from training rows only**. Missing values are represented via `-1` bin codes and summarized by `missing_count`.

Identifiers / time context:

- `row_id`: unique row identifier (integer)
- `entity_id`: stable anonymized entity identifier (integer)
- `segment_id`: metric segment identifier (0/1/2)
- `horizon_y`: horizon in years (1 or 2)
- `event_era`: coarse time bucket derived from the hidden year (older → newer)
- `missing_count`: number of missing raw/derived fields used for features
- `slice_hard_transition`: 1 if baseline is fossil-heavy and renewables-light, else 0

Energy / macro context (binned; examples):

- `fossil_share_energy_bin`, `renewables_share_energy_bin`, `low_carbon_share_energy_bin`
- `fossil_share_elec_bin`, `renewables_share_elec_bin`, `low_carbon_share_elec_bin`
- `energy_cons_change_pct_bin`, `fossil_cons_change_pct_bin`, `renewables_cons_change_pct_bin`
- `carbon_intensity_elec_bin`
- `population_bin`, `gdp_bin`, `energy_per_capita_bin`, `energy_per_gdp_bin`, `electricity_demand_per_capita_bin`
- short-history stability proxies: `*_std5_bin` and 1-year changes `*_chg1_bin`

Label (train/solution only):

- `target_fossil_share_rebound_next`: 1 if \(\Delta M_h\) exceeds the entity’s training-era 85th percentile threshold for that segment+horizon, else 0

## Splitting & Leakage

- **Split strategy**: time-based regime shift. Older years are used for training; newer eras are emphasized in test. A deterministic portion of bridge years is mixed into test.
- **Leakage mitigations**:
  - exact years and country identifiers are excluded (only anonymized ids and coarse eras),
  - bin edges are learned from training rows only,
  - the label depends on future fossil share not present in features.

