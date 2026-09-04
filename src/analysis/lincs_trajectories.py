from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


CONDITION_COLUMNS = ["cell_id", "pert_id", "pert_idose", "pert_type"]


def parse_time_hours(value: object) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match:
        return None
    numeric = float(match.group())
    text = str(value).lower()
    if "min" in text:
        return numeric / 60.0
    if "day" in text or re.search(r"\bd\b", text):
        return numeric * 24.0
    return numeric


def matched_timepoint_metadata(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = (
        processed_dir
        / "lincs_gse70138"
        / "lincs_cancer_landmark_signature_metadata.parquet"
    )
    metadata = pd.read_parquet(path)
    required = set(CONDITION_COLUMNS + ["sig_id", "pert_itime", "matrix_row"])
    if not required.issubset(metadata.columns):
        raise ValueError(f"LINCS signature metadata missing: {sorted(required - set(metadata))}")
    metadata = metadata[metadata["pert_type"].isin(["trt_cp", "trt_xpr"])].copy()
    metadata["time_hours"] = metadata["pert_itime"].map(parse_time_hours)
    metadata = metadata.dropna(subset=CONDITION_COLUMNS + ["time_hours", "matrix_row"])
    metadata["matrix_row"] = metadata["matrix_row"].astype(int)
    time_counts = (
        metadata.groupby(CONDITION_COLUMNS, dropna=False)["time_hours"]
        .nunique()
        .rename("unique_timepoints")
        .reset_index()
    )
    eligible_conditions = time_counts[time_counts["unique_timepoints"].ge(2)]
    eligible = metadata.merge(
        eligible_conditions[CONDITION_COLUMNS + ["unique_timepoints"]],
        on=CONDITION_COLUMNS,
        how="inner",
        validate="many_to_one",
    )
    return eligible, eligible_conditions


def profile_lincs_matched_trajectories(processed_dir: Path, results_dir: Path) -> dict:
    """Profile real matched-condition LINCS time courses before model execution."""
    eligible, conditions = matched_timepoint_metadata(processed_dir)
    time_sequences = (
        eligible.groupby(CONDITION_COLUMNS)["time_hours"]
        .apply(lambda values: "->".join(map(lambda value: f"{value:g}h", sorted(set(values)))))
        .value_counts()
        .to_dict()
    )
    output_dir = results_dir / "validation" / "lincs_future_state"
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible_path = output_dir / "matched_timepoint_signature_metadata.parquet"
    condition_path = output_dir / "matched_timepoint_conditions.csv"
    eligible.to_parquet(eligible_path, index=False)
    conditions.to_csv(condition_path, index=False)
    summary = {
        "status": "created",
        "data_policy": "observed LINCS signatures only; no simulated trajectories",
        "matched_conditions": int(len(conditions)),
        "matched_signatures": int(len(eligible)),
        "cancer_cells": int(eligible["cell_id"].nunique()),
        "perturbations": int(eligible["pert_id"].nunique()),
        "time_points_hours": sorted(map(float, eligible["time_hours"].unique())),
        "time_sequences": time_sequences,
        "eligible_metadata": str(eligible_path),
        "conditions": str(condition_path),
    }
    summary_path = output_dir / "matched_trajectory_profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
