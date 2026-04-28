from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


COMP_DIR = Path(__file__).resolve().parent
ROOT = COMP_DIR.parent

UPSTREAM_CACHE = (
    ROOT
    / "competition_owid_energy_renewables_accel_nextyear_regime_shift"
    / "_cache"
    / "owid-energy-data.csv"
)

N_BINS = 12
EPS = 1e-12

ID_COLUMN = "row_id"
ENTITY_COLUMN = "entity_id"
HORIZON_COLUMN = "horizon_y"  # 1 or 2 years
SEGMENT_COLUMN = "segment_id"  # 0..K-1
ERA_COLUMN = "event_era"
MISSING_COLUMN = "missing_count"

TARGET_COLUMN = "target_fossil_share_rebound_next"
PRED_COLUMN = "pred_fossil_share_rebound_next"
SLICE_COLUMN = "slice_hard_transition"

MIN_TRAIN_ROWS = 12_000


def _stable_int_hash(text: str, n_bytes: int = 8) -> int:
    h = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(h[:n_bytes], byteorder="big", signed=False)


def _is_iso3(x: object) -> bool:
    if not isinstance(x, str):
        return False
    if len(x) != 3:
        return False
    return x.isalpha() and x.upper() == x


def _event_era(year: int) -> int:
    if year <= 1999:
        return 0
    if year <= 2009:
        return 1
    if year <= 2015:
        return 2
    return 3


def _hash_bucket(text: str, mod: int) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % mod


def _quantile_edges(train_values: np.ndarray, n_bins: int) -> np.ndarray:
    train_values = train_values[np.isfinite(train_values)]
    if train_values.size == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(train_values, qs).astype(float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + EPS
    return edges


def _bin_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=int)
    mask = np.isfinite(values)
    if edges.size == 0:
        out[mask] = 0
        return out
    out[mask] = np.digitize(values[mask], edges, right=False).astype(int)
    return out


def _make_bins(df_all: pd.DataFrame, df_train: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    edges_map: Dict[str, List[float]] = {}
    for c in cols:
        edges = _quantile_edges(df_train[c].to_numpy(dtype=float), N_BINS)
        edges_map[c] = edges.tolist()
        df_all[f"{c}_bin"] = _bin_with_edges(df_all[c].to_numpy(dtype=float), edges)
    return df_all, edges_map


def main() -> None:
    if not UPSTREAM_CACHE.exists():
        raise FileNotFoundError(f"Missing upstream cache file: {UPSTREAM_CACHE}")

    usecols = [
        "iso_code",
        "country",
        "year",
        "population",
        "gdp",
        "energy_per_capita",
        "energy_per_gdp",
        "electricity_demand_per_capita",
        "carbon_intensity_elec",
        "fossil_share_energy",
        "renewables_share_energy",
        "low_carbon_share_energy",
        "energy_cons_change_pct",
        "fossil_cons_change_pct",
        "renewables_cons_change_pct",
        "fossil_share_elec",
        "renewables_share_elec",
        "low_carbon_share_elec",
    ]
    df = pd.read_csv(UPSTREAM_CACHE, usecols=usecols, low_memory=False)
    df["iso_code"] = df["iso_code"].astype(str)
    df = df[df["iso_code"].map(_is_iso3)].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2022)].copy()

    # Numeric conversions.
    num_cols = [
        "population",
        "gdp",
        "energy_per_capita",
        "energy_per_gdp",
        "electricity_demand_per_capita",
        "carbon_intensity_elec",
        "fossil_share_energy",
        "renewables_share_energy",
        "low_carbon_share_energy",
        "energy_cons_change_pct",
        "fossil_cons_change_pct",
        "renewables_cons_change_pct",
        "fossil_share_elec",
        "renewables_share_elec",
        "low_carbon_share_elec",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["iso_code", "year"]).reset_index(drop=True)
    g = df.groupby("iso_code", sort=False)

    # Past-only features.
    df["fossil_share_energy_chg1"] = g["fossil_share_energy"].diff(1)
    df["renewables_share_energy_chg1"] = g["renewables_share_energy"].diff(1)
    df["energy_cons_change_pct_std5"] = g["energy_cons_change_pct"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    df["fossil_share_energy_std5"] = g["fossil_share_energy"].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)

    # Horizons: next 1y and 2y fossil share change.
    panel_rows = []
    taus: Dict[str, Dict[str, float]] = {}

    # Deterministic time split: newer years emphasized in test; deterministic bridge mix.
    df[ERA_COLUMN] = df["year"].map(_event_era).astype(int)
    bridge_bucket = df.apply(lambda r: _hash_bucket(f"{r['iso_code']}_{int(r['year'])}", 10), axis=1)
    is_test = (df["year"] >= 2018) | ((df["year"].isin([2016, 2017])) & (bridge_bucket == 0))
    df["is_test"] = is_test.astype(int)

    segments = [
        (0, "fossil_share_energy"),
        (1, "fossil_share_elec"),
        (2, "carbon_intensity_elec"),
    ]

    for seg_id, seg_col in segments:
        for h in (1, 2):
            sub = df.copy()
            sub[SEGMENT_COLUMN] = int(seg_id)
            sub[HORIZON_COLUMN] = h
            gs = sub.groupby("iso_code", sort=False)
            sub["delta_metric_h"] = gs[seg_col].shift(-h) - sub[seg_col]
            sub = sub[sub["delta_metric_h"].notna()].copy()

            train_sub = sub[sub["is_test"] == 0].copy()
            if train_sub.empty:
                continue

            global_tau = float(np.nanquantile(train_sub["delta_metric_h"].to_numpy(dtype=float), 0.85))
            thr_by: Dict[str, float] = {}
            for iso, gg in train_sub.groupby("iso_code", sort=True):
                v = gg["delta_metric_h"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size >= 20:
                    thr_by[str(iso)] = float(np.quantile(v, 0.85))
            sub["tau_entity"] = sub["iso_code"].map(lambda s: float(thr_by.get(str(s), global_tau))).astype(float)
            sub[TARGET_COLUMN] = (sub["delta_metric_h"].to_numpy(dtype=float) >= sub["tau_entity"].to_numpy(dtype=float)).astype(int)

            taus[f"seg{seg_id}_{seg_col}_h{h}"] = {"global_tau_p85": global_tau, "n_entities_specific": int(len(thr_by))}
            panel_rows.append(sub)

    if not panel_rows:
        raise RuntimeError("No panel rows produced; cannot build dataset.")

    panel = pd.concat(panel_rows, axis=0, ignore_index=True)

    # Anonymize entity id.
    panel[ENTITY_COLUMN] = panel["iso_code"].map(lambda s: _stable_int_hash(f"owid_iso3::{s}", n_bytes=8)).astype(np.int64)

    feature_cols = [
        "population",
        "gdp",
        "energy_per_capita",
        "energy_per_gdp",
        "electricity_demand_per_capita",
        "carbon_intensity_elec",
        "fossil_share_energy",
        "renewables_share_energy",
        "low_carbon_share_energy",
        "energy_cons_change_pct",
        "fossil_cons_change_pct",
        "renewables_cons_change_pct",
        "fossil_share_elec",
        "renewables_share_elec",
        "low_carbon_share_elec",
        "fossil_share_energy_chg1",
        "renewables_share_energy_chg1",
        "energy_cons_change_pct_std5",
        "fossil_share_energy_std5",
    ]

    panel[MISSING_COLUMN] = panel[feature_cols].isna().sum(axis=1).astype(int)

    train_rows = panel[panel["is_test"] == 0].copy()
    if train_rows.empty:
        raise RuntimeError("No training rows after split; cannot build dataset.")

    panel_all, edges_map = _make_bins(panel, train_rows, feature_cols)

    # Slice: "hard transition" baseline (high fossil + low renewables).
    panel_all[SLICE_COLUMN] = ((panel_all["fossil_share_energy_bin"] >= 9) & (panel_all["renewables_share_energy_bin"] <= 2)).astype(int)

    keep = [ENTITY_COLUMN, SEGMENT_COLUMN, HORIZON_COLUMN, ERA_COLUMN, MISSING_COLUMN] + [f"{c}_bin" for c in feature_cols] + [SLICE_COLUMN]
    out = panel_all[keep + [TARGET_COLUMN, "is_test"]].copy()
    out[ID_COLUMN] = np.arange(out.shape[0], dtype=np.int64)

    train = out[out["is_test"] == 0].drop(columns=["is_test"]).copy()
    test = out[out["is_test"] == 1].drop(columns=["is_test", TARGET_COLUMN]).copy()
    solution = out[out["is_test"] == 1][[ID_COLUMN, TARGET_COLUMN, SLICE_COLUMN]].copy()

    train = train.sort_values(ID_COLUMN).reset_index(drop=True)
    test = test.sort_values(ID_COLUMN).reset_index(drop=True)
    solution = solution.sort_values(ID_COLUMN).reset_index(drop=True)

    if len(train) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Train set too small ({len(train)} rows). Need >= {MIN_TRAIN_ROWS} to avoid rejection risk.")

    sample = test[[ID_COLUMN]].copy()
    sample[PRED_COLUMN] = 0.5

    perfect = solution[[ID_COLUMN, TARGET_COLUMN]].copy()
    perfect[PRED_COLUMN] = perfect[TARGET_COLUMN].astype(float)
    perfect = perfect[[ID_COLUMN, PRED_COLUMN]]

    train.to_csv(COMP_DIR / "train.csv", index=False)
    test.to_csv(COMP_DIR / "test.csv", index=False)
    solution.to_csv(COMP_DIR / "solution.csv", index=False)
    sample.to_csv(COMP_DIR / "sample_submission.csv", index=False)
    perfect.to_csv(COMP_DIR / "perfect_submission.csv", index=False)

    meta = {
        "upstream_cache": str(UPSTREAM_CACHE),
        "n_bins": N_BINS,
        "horizons_y": [1, 2],
        "segments": [
            {"segment_id": 0, "metric": "fossil_share_energy"},
            {"segment_id": 1, "metric": "fossil_share_elec"},
            {"segment_id": 2, "metric": "carbon_intensity_elec"},
        ],
        "thresholds_summary": taus,
        "bin_edges": edges_map,
        "slice_definition": {
            "slice_column": SLICE_COLUMN,
            "rule": "slice_hard_transition = 1 if fossil_share_energy_bin >= 9 and renewables_share_energy_bin <= 2 (n_bins=12).",
        },
        "row_counts": {"train": int(len(train)), "test": int(len(test))},
        "positive_rate": {
            "train": float(train[TARGET_COLUMN].mean()) if len(train) else None,
            "test": float(solution[TARGET_COLUMN].mean()) if len(solution) else None,
        },
    }
    (COMP_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    print("Wrote competition files to", COMP_DIR)
    print("train rows:", len(train), "test rows:", len(test))
    print("train positive rate:", float(train[TARGET_COLUMN].mean()) if len(train) else float("nan"))
    print("test positive rate:", float(solution[TARGET_COLUMN].mean()) if len(solution) else float("nan"))


if __name__ == "__main__":
    main()

