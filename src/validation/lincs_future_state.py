from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.lincs_trajectories import CONDITION_COLUMNS, matched_timepoint_metadata
from src.observability import (
    TransitionEvent,
    append_transition_event,
    build_recursive_observability_report,
)
from src.observability.recursive import trace_completeness


AGENT_NAMES = ("ridge", "random_forest", "nearest_neighbors")
SEED = 20260824


def _dose_numeric(value: object) -> tuple[float, int]:
    import re

    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match or str(value).strip() in {"-666", "-666.0"}:
        return 0.0, 1
    return float(match.group()), 0


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return 0.0
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _aggregate_real_states(
    matrix_path: Path,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    state_vectors: list[np.ndarray] = []
    state_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    with h5py.File(matrix_path, "r") as handle:
        matrix = handle["matrix"]
        for condition, condition_table in metadata.groupby(CONDITION_COLUMNS, sort=True):
            condition_states: list[tuple[float, int, str, float | None, int]] = []
            for time_hours, time_table in condition_table.groupby("time_hours", sort=True):
                rows = sorted(time_table["matrix_row"].astype(int).unique())
                values = matrix[rows, :].astype(np.float64)
                mean_state = values.mean(axis=0)
                replicate_uncertainty = (
                    float(np.mean(np.std(values, axis=0, ddof=1))) if len(rows) > 1 else None
                )
                state_index = len(state_vectors)
                state_vectors.append(mean_state.astype(np.float32))
                evidence = sorted(time_table["sig_id"].astype(str).unique())
                state_rows.append(
                    {
                        **dict(zip(CONDITION_COLUMNS, condition)),
                        "time_hours": float(time_hours),
                        "state_index": state_index,
                        "signature_count": len(rows),
                        "replicate_uncertainty": replicate_uncertainty,
                        "evidence_sig_ids": "|".join(evidence),
                    }
                )
                condition_states.append(
                    (float(time_hours), state_index, "|".join(evidence), replicate_uncertainty, len(rows))
                )
            condition_states.sort()
            for before, after in zip(condition_states, condition_states[1:]):
                dose, dose_missing = _dose_numeric(condition[2])
                transition_rows.append(
                    {
                        **dict(zip(CONDITION_COLUMNS, condition)),
                        "before_hours": before[0],
                        "after_hours": after[0],
                        "delta_hours": after[0] - before[0],
                        "before_state_index": before[1],
                        "after_state_index": after[1],
                        "before_evidence_sig_ids": before[2],
                        "after_evidence_sig_ids": after[2],
                        "before_replicate_uncertainty": before[3],
                        "after_replicate_uncertainty": after[3],
                        "before_signature_count": before[4],
                        "after_signature_count": after[4],
                        "dose_numeric": dose,
                        "dose_missing": dose_missing,
                    }
                )
    return np.vstack(state_vectors), pd.DataFrame(state_rows), pd.DataFrame(transition_rows)


def _frozen_perturbation_split(perturbations: list[str]) -> dict[str, list[str]]:
    identifiers = np.asarray(sorted(set(map(str, perturbations))), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(identifiers)
    test_count = max(1, int(round(0.20 * len(identifiers))))
    calibration_count = max(1, int(round(0.20 * (len(identifiers) - test_count))))
    return {
        "holdout": sorted(map(str, identifiers[:test_count])),
        "calibration": sorted(map(str, identifiers[test_count : test_count + calibration_count])),
        "fit": sorted(map(str, identifiers[test_count + calibration_count :])),
    }


def _features(before_state: np.ndarray, transitions: pd.DataFrame) -> np.ndarray:
    dose = np.log1p(np.abs(transitions["dose_numeric"].to_numpy(dtype=float)))
    timing = np.column_stack(
        [
            np.log1p(transitions["before_hours"].to_numpy(dtype=float)),
            np.log1p(transitions["after_hours"].to_numpy(dtype=float)),
            np.log1p(transitions["delta_hours"].to_numpy(dtype=float)),
            dose,
            transitions["dose_missing"].to_numpy(dtype=float),
        ]
    )
    return np.column_stack([before_state, timing])


def _models(fit_rows: int) -> dict[str, Any]:
    neighbors = max(3, min(25, fit_rows - 1))
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=-1,
        ),
        "nearest_neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=neighbors, weights="distance", p=2),
        ),
    }


def _prediction_disagreement(predictions: np.ndarray, master: np.ndarray) -> np.ndarray:
    return np.mean(np.linalg.norm(predictions - master[None, :, :], axis=2), axis=0) / np.sqrt(
        master.shape[1]
    )


def _counterfactual_influence(
    predictions: np.ndarray,
    weights: np.ndarray,
    master: np.ndarray,
) -> np.ndarray:
    influence = np.zeros((master.shape[0], len(weights)), dtype=float)
    for index in range(len(weights)):
        retained = np.delete(weights, index)
        retained /= retained.sum()
        leave_one = np.tensordot(retained, np.delete(predictions, index, axis=0), axes=(0, 0))
        influence[:, index] = np.linalg.norm(master - leave_one, axis=1) / np.sqrt(master.shape[1])
    return influence


def run_lincs_future_state_audit(
    processed_dir: Path,
    results_dir: Path,
    model_dir: Path,
    *,
    components: int = 16,
) -> dict:
    """Audit real matched-time LINCS forecasts on frozen unseen perturbations."""
    eligible, _ = matched_timepoint_metadata(processed_dir)
    matrix_path = processed_dir / "lincs_gse70138" / "lincs_cancer_landmark_level5.h5"
    states, state_table, transitions = _aggregate_real_states(matrix_path, eligible)
    split = _frozen_perturbation_split(transitions["pert_id"].astype(str).tolist())
    split_name = pd.Series("", index=transitions.index, dtype="string")
    for name, identifiers in split.items():
        split_name.loc[transitions["pert_id"].astype(str).isin(identifiers)] = name
    transitions["split"] = split_name
    masks = {name: transitions["split"].eq(name).to_numpy() for name in split}
    if any(mask.sum() < 20 for mask in masks.values()):
        raise ValueError({name: int(mask.sum()) for name, mask in masks.items()})

    fit_transition_indices = np.flatnonzero(masks["fit"])
    fit_state_indices = np.unique(
        transitions.loc[
            fit_transition_indices, ["before_state_index", "after_state_index"]
        ].to_numpy(dtype=int)
    )
    gene_scaler = StandardScaler().fit(states[fit_state_indices])
    component_count = min(components, len(fit_state_indices) - 1, states.shape[1])
    pca = PCA(n_components=component_count, svd_solver="randomized", random_state=SEED)
    pca.fit(gene_scaler.transform(states[fit_state_indices]))
    raw_components = pca.transform(gene_scaler.transform(states))
    component_scaler = StandardScaler().fit(raw_components[fit_state_indices])
    represented_states = component_scaler.transform(raw_components)

    before = represented_states[transitions["before_state_index"].to_numpy(dtype=int)]
    after = represented_states[transitions["after_state_index"].to_numpy(dtype=int)]
    features = _features(before, transitions)
    models = _models(int(masks["fit"].sum()))
    calibration_predictions = []
    calibration_errors: dict[str, float] = {}
    for name in AGENT_NAMES:
        model = models[name]
        model.fit(features[masks["fit"]], after[masks["fit"]])
        prediction = model.predict(features[masks["calibration"]])
        calibration_predictions.append(prediction)
        calibration_errors[name] = float(np.mean((prediction - after[masks["calibration"]]) ** 2))
    inverse_error = np.asarray(
        [1.0 / (calibration_errors[name] + np.finfo(float).eps) for name in AGENT_NAMES]
    )
    weights = inverse_error / inverse_error.sum()

    calibration_stack = np.stack(calibration_predictions)
    calibration_master = np.tensordot(weights, calibration_stack, axes=(0, 0))
    calibration_disagreement = _prediction_disagreement(calibration_stack, calibration_master)
    calibration_actual_error = np.linalg.norm(
        calibration_master - after[masks["calibration"]], axis=1
    ) / np.sqrt(component_count)
    calibration_change = np.linalg.norm(
        calibration_master - before[masks["calibration"]], axis=1
    ) / np.sqrt(component_count)
    uncertainty_calibrator = LinearRegression(positive=True).fit(
        np.column_stack(
            [
                calibration_disagreement,
                calibration_change,
                transitions.loc[masks["calibration"], "delta_hours"].to_numpy(dtype=float),
            ]
        ),
        calibration_actual_error,
    )

    test_predictions = np.stack(
        [models[name].predict(features[masks["holdout"]]) for name in AGENT_NAMES]
    )
    test_before = before[masks["holdout"]]
    test_after = after[masks["holdout"]]
    master_prediction = np.tensordot(weights, test_predictions, axes=(0, 0))
    agent_errors = np.linalg.norm(test_predictions - test_after[None, :, :], axis=2) / np.sqrt(
        component_count
    )
    individual_agent_instability = agent_errors.mean(axis=0)
    inter_agent_disagreement = _prediction_disagreement(test_predictions, master_prediction)
    master_instability = np.linalg.norm(master_prediction - test_after, axis=1) / np.sqrt(
        component_count
    )
    cancer_future_state_instability = np.linalg.norm(test_after - test_before, axis=1) / np.sqrt(
        component_count
    )
    predicted_change = np.linalg.norm(master_prediction - test_before, axis=1) / np.sqrt(
        component_count
    )
    holdout_transitions = transitions.loc[masks["holdout"]].reset_index(drop=True)
    master_self_uncertainty = np.clip(
        uncertainty_calibrator.predict(
            np.column_stack(
                [
                    inter_agent_disagreement,
                    predicted_change,
                    holdout_transitions["delta_hours"].to_numpy(dtype=float),
                ]
            )
        ),
        0,
        None,
    )
    meta_uncertainty = np.abs(master_self_uncertainty - master_instability)
    influence = _counterfactual_influence(test_predictions, weights, master_prediction)

    actual_scaled = gene_scaler.transform(states)
    reconstructed_scaled = pca.inverse_transform(raw_components)
    state_reconstruction_error = np.linalg.norm(actual_scaled - reconstructed_scaled, axis=1) / np.sqrt(
        states.shape[1]
    )
    holdout_after_indices = holdout_transitions["after_state_index"].to_numpy(dtype=int)

    out_dir = results_dir / "validation" / "lincs_future_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "frozen_holdout_recursive_events.jsonl"
    events_path.unlink(missing_ok=True)
    event_records = []
    future_weights = {name: float(weight) for name, weight in zip(AGENT_NAMES, weights)}
    amplitudes = {name: float(np.sqrt(weight)) for name, weight in future_weights.items()}
    for row_index, row in holdout_transitions.iterrows():
        evidence = (
            str(row["before_evidence_sig_ids"]).split("|")[:8]
            + str(row["after_evidence_sig_ids"]).split("|")[:8]
            + ["NCBI_GEO:GSE70138"]
        )
        for agent_index, agent_name in enumerate(AGENT_NAMES):
            event = TransitionEvent(
                episode_id=f"lincs-holdout-{row_index:05d}",
                time=f"observed-{row['after_hours']:g}h",
                cancer_state_id=(
                    f"{row['cell_id']}|{row['pert_id']}|{row['pert_idose']}|"
                    f"{row['before_hours']:g}-{row['after_hours']:g}h"
                ),
                agent_id=agent_name,
                agent_state_before=test_before[row_index].tolist(),
                agent_state_after=test_predictions[agent_index, row_index].tolist(),
                action="forecast_later_observed_perturbation_state",
                reward=float(1.0 / (1.0 + agent_errors[agent_index, row_index])),
                agent_uncertainty=float(np.sqrt(calibration_errors[agent_name])),
                master_state_before=test_before[row_index].tolist(),
                master_state_after=master_prediction[row_index].tolist(),
                master_uncertainty=float(master_self_uncertainty[row_index]),
                meta_uncertainty=float(meta_uncertainty[row_index]),
                evidence_ids=evidence,
                parent_trace=f"lincs-condition-{row_index:05d}",
                future_states_considered=future_weights,
                quantum_inspired_state=amplitudes,
                gate_applied="frozen_unseen_perturbation_holdout_gate",
                hypothesis_before="forecast_from_observed_earlier_post_perturbation_state",
                hypothesis_after="forecast_audited_against_observed_later_state",
            )
            append_transition_event(events_path, event)
            event_records.append(event.to_record())

    report = build_recursive_observability_report(
        events=event_records,
        cancer_instability=cancer_future_state_instability,
        individual_agent_instability=individual_agent_instability,
        inter_agent_disagreement=inter_agent_disagreement,
        master_instability=master_instability,
        meta_uncertainty=meta_uncertainty,
        master_state=test_after,
        reconstructed_master_state=master_prediction,
        self_observation=master_instability.reshape(-1, 1),
        reconstructed_self_observation=master_self_uncertainty.reshape(-1, 1),
        agent_influence_identifiability=float(np.mean(np.isfinite(influence).all(axis=1))),
        uncertainty_visibility=float(
            np.mean(np.isfinite(np.column_stack([master_self_uncertainty, meta_uncertainty])))
        ),
    )

    audit_columns: dict[str, np.ndarray] = {
        "cancer_future_state_instability": cancer_future_state_instability,
        "individual_agent_instability": individual_agent_instability,
        "inter_agent_disagreement": inter_agent_disagreement,
        "master_state_instability": master_instability,
        "master_self_observation_uncertainty": master_self_uncertainty,
        "meta_uncertainty": meta_uncertainty,
        "after_state_reconstruction_error": state_reconstruction_error[holdout_after_indices],
    }
    for agent_index, name in enumerate(AGENT_NAMES):
        audit_columns[f"{name}_instability"] = agent_errors[agent_index]
        audit_columns[f"{name}_counterfactual_influence"] = influence[:, agent_index]
        for component in range(component_count):
            audit_columns[f"{name}_forecast_{component + 1}"] = test_predictions[
                agent_index, :, component
            ]
    for component in range(component_count):
        audit_columns[f"observed_before_state_{component + 1}"] = test_before[:, component]
        audit_columns[f"observed_after_state_{component + 1}"] = test_after[:, component]
        audit_columns[f"master_forecast_state_{component + 1}"] = master_prediction[:, component]
    audit = pd.concat(
        [holdout_transitions, pd.DataFrame(audit_columns, index=holdout_transitions.index)],
        axis=1,
    )

    # Persist the frozen calibration partition as the reward-learning set for the
    # downstream offline reinforcement Master. The holdout partition above remains
    # untouched until policy evaluation.
    calibration_transitions = transitions.loc[masks["calibration"]].reset_index(drop=True)
    calibration_before = before[masks["calibration"]]
    calibration_after = after[masks["calibration"]]
    calibration_agent_errors = np.linalg.norm(
        calibration_stack - calibration_after[None, :, :], axis=2
    ) / np.sqrt(component_count)
    calibration_cancer_instability = np.linalg.norm(
        calibration_after - calibration_before, axis=1
    ) / np.sqrt(component_count)
    calibration_master_instability = np.linalg.norm(
        calibration_master - calibration_after, axis=1
    ) / np.sqrt(component_count)
    calibration_predicted_change = np.linalg.norm(
        calibration_master - calibration_before, axis=1
    ) / np.sqrt(component_count)
    calibration_self_uncertainty = np.clip(
        uncertainty_calibrator.predict(
            np.column_stack(
                [
                    calibration_disagreement,
                    calibration_predicted_change,
                    calibration_transitions["delta_hours"].to_numpy(dtype=float),
                ]
            )
        ),
        0,
        None,
    )
    calibration_influence = _counterfactual_influence(
        calibration_stack, weights, calibration_master
    )
    calibration_columns: dict[str, np.ndarray] = {
        "cancer_future_state_instability": calibration_cancer_instability,
        "individual_agent_instability": calibration_agent_errors.mean(axis=0),
        "inter_agent_disagreement": calibration_disagreement,
        "master_state_instability": calibration_master_instability,
        "master_self_observation_uncertainty": calibration_self_uncertainty,
        "meta_uncertainty": np.abs(
            calibration_self_uncertainty - calibration_master_instability
        ),
    }
    for agent_index, name in enumerate(AGENT_NAMES):
        calibration_columns[f"{name}_instability"] = calibration_agent_errors[agent_index]
        calibration_columns[f"{name}_reward"] = 1.0 / (
            1.0 + calibration_agent_errors[agent_index]
        )
        calibration_columns[f"{name}_counterfactual_influence"] = calibration_influence[
            :, agent_index
        ]
        for component in range(component_count):
            calibration_columns[f"{name}_forecast_{component + 1}"] = calibration_stack[
                agent_index, :, component
            ]
    for component in range(component_count):
        calibration_columns[f"observed_before_state_{component + 1}"] = calibration_before[
            :, component
        ]
        calibration_columns[f"observed_after_state_{component + 1}"] = calibration_after[
            :, component
        ]
        calibration_columns[f"master_forecast_state_{component + 1}"] = calibration_master[
            :, component
        ]
    calibration_audit = pd.concat(
        [
            calibration_transitions,
            pd.DataFrame(calibration_columns, index=calibration_transitions.index),
        ],
        axis=1,
    )

    audit_path = out_dir / "frozen_holdout_transition_audit.parquet"
    calibration_audit_path = out_dir / "frozen_calibration_transition_audit.parquet"
    state_path = out_dir / "matched_condition_states.parquet"
    audit.to_parquet(audit_path, index=False)
    calibration_audit.to_parquet(calibration_audit_path, index=False)
    state_table.to_parquet(state_path, index=False)
    split_payload = {
        "seed": SEED,
        "split_unit": "pert_id",
        "fit_perturbations": split["fit"],
        "calibration_perturbations": split["calibration"],
        "holdout_perturbations": split["holdout"],
    }
    canonical_split = json.dumps(split_payload, sort_keys=True, separators=(",", ":"))
    split_payload["sha256"] = hashlib.sha256(canonical_split.encode("utf-8")).hexdigest()
    split_path = out_dir / "frozen_perturbation_split.json"
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    model_output_dir = model_dir / "lincs_future_state"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_output_dir / "frozen_holdout_forecasters.joblib"
    joblib.dump(
        {
            "gene_scaler": gene_scaler,
            "pca": pca,
            "component_scaler": component_scaler,
            "agents": models,
            "agent_weights": future_weights,
            "uncertainty_calibrator": uncertainty_calibrator,
            "split_sha256": split_payload["sha256"],
        },
        model_path,
    )

    report_payload = report.to_dict()
    report_payload.update(
        {
            "status": "created",
            "data_policy": "observed LINCS matched-time cancer signatures; no simulation",
            "claim_boundary": (
                "Matched cell-line perturbation forecasting; not patient-level longitudinal "
                "prediction and not a clinical future-state claim."
            ),
            "conditions": int(transitions[CONDITION_COLUMNS].drop_duplicates().shape[0]),
            "transitions": int(len(transitions)),
            "fit_transitions": int(masks["fit"].sum()),
            "calibration_transitions": int(masks["calibration"].sum()),
            "holdout_transitions": int(masks["holdout"].sum()),
            "holdout_perturbations": len(split["holdout"]),
            "components": component_count,
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "agent_calibration_mse": calibration_errors,
            "agent_weights": future_weights,
            "master_holdout_rmse": float(np.sqrt(np.mean((master_prediction - test_after) ** 2))),
            "master_uncertainty_mae": float(np.mean(meta_uncertainty)),
            "master_uncertainty_error_correlation": _safe_correlation(
                master_self_uncertainty, master_instability
            ),
            "trace_completeness_recheck": trace_completeness(event_records),
            "artifacts": {
                "audit": str(audit_path),
                "calibration_audit": str(calibration_audit_path),
                "states": str(state_path),
                "events": str(events_path),
                "split": str(split_path),
                "models": str(model_path),
            },
        }
    )
    report_path = out_dir / "lincs_future_state_recursive_observability_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return {
        "status": "created",
        "holdout_transitions": report_payload["holdout_transitions"],
        "holdout_perturbations": report_payload["holdout_perturbations"],
        "transition_events": len(event_records),
        "trace_completeness": report_payload["trace_completeness_recheck"],
        "master_holdout_rmse": report_payload["master_holdout_rmse"],
        "master_uncertainty_mae": report_payload["master_uncertainty_mae"],
        "rmoi_profile": report_payload["rmoi_profile"],
        "osic": report_payload["osic"],
        "oosc": report_payload["oosc"],
        "observer_depth": report_payload["observer_depth"],
        "report": str(report_path),
        "artifacts": report_payload["artifacts"],
        "claim_boundary": report_payload["claim_boundary"],
    }
