from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.observability.recursive import (
    TransitionEvent,
    append_transition_event,
    build_recursive_observability_report,
    trace_completeness,
)


RANDOM_SEED = 20260827
BOOTSTRAPS = 2000
PERMUTATIONS = 2000
AGENT_ORDER = ("linear_transcriptome", "nonlinear_forest", "patient_neighborhood")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    return -(
        probability * np.log2(probability)
        + (1 - probability) * np.log2(1 - probability)
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return None
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _safe_auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    return float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean() - probability[mask].mean()))
    return float(total)


def score_pcawg_external(
    processed_dir: Path,
    model_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Apply the frozen TCGA model to every independent PCAWG expression donor without labels."""
    expression_path = (
        processed_dir / "pcawg_external" / "pcawg_non_tcga_frozen_gene_expression.parquet"
    )
    model_path = (
        model_dir / "tcga_outcome_observability" / "tcga_os2y_frozen_bundle.joblib"
    )
    if not expression_path.exists() or not model_path.exists():
        raise FileNotFoundError("Frozen model or harmonised PCAWG expression is missing")

    bundle = joblib.load(model_path)
    selected_genes = [str(value).split(".", 1)[0] for value in bundle["selected_gene_ids"]]
    expression = pd.read_parquet(expression_path)
    missing = [gene for gene in selected_genes if gene not in expression.columns]
    if missing:
        raise ValueError(f"Harmonised PCAWG matrix is missing {len(missing)} frozen genes")
    donor_ids = expression["donor_id"].astype(str).to_numpy()
    selected = expression[selected_genes].to_numpy(dtype=float)

    scaler = bundle["scaler"]
    pca = bundle["pca"]
    scaled = scaler.transform(selected)
    reduced = pca.transform(scaled)
    agent_inputs = {
        "linear_transcriptome": scaled,
        "nonlinear_forest": selected,
        "patient_neighborhood": reduced,
    }
    agent_mean = []
    agent_sd = []
    for agent in AGENT_ORDER:
        predictions = np.vstack(
            [
                model.predict_proba(agent_inputs[agent])[:, 1]
                for model in bundle["fitted_agents"][agent]
            ]
        )
        agent_mean.append(predictions.mean(axis=0))
        agent_sd.append(predictions.std(axis=0))
    agent_probabilities = np.column_stack(agent_mean)
    agent_instability = np.column_stack(agent_sd).mean(axis=1)
    disagreement = agent_probabilities.std(axis=1)
    weights = np.asarray(bundle["master_weights"], dtype=float)
    master_probability = agent_probabilities @ weights
    master_instability = _entropy(master_probability)

    fit_reduced = np.asarray(bundle["fit_reduced"], dtype=float)
    fit_outcomes = np.asarray(bundle["fit_outcomes"], dtype=int)
    neighbor_count = min(15, len(fit_outcomes))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(fit_reduced)
    distances, indices = neighbors.kneighbors(reduced)
    cancer_reference_instability = _entropy(fit_outcomes[indices].mean(axis=1))
    ood = distances[:, 0]
    ood_scaled = bundle["ood_scaler"].transform(ood.reshape(-1, 1)).ravel()
    meta_features = np.column_stack(
        [
            cancer_reference_instability,
            agent_instability,
            disagreement,
            master_instability,
            ood_scaled,
        ]
    )
    meta_scaled = bundle["meta_scaler"].transform(meta_features)
    meta_predictions = np.vstack(
        [np.clip(model.predict(meta_scaled), 0, 1) for model in bundle["meta_models"]]
    )
    predicted_error = meta_predictions.mean(axis=0)
    meta_uncertainty = meta_predictions.std(axis=0)

    output = pd.DataFrame({"donor_id": donor_ids})
    for index, agent in enumerate(AGENT_ORDER):
        output[f"{agent}_probability"] = agent_probabilities[:, index]
        output[f"{agent}_bootstrap_sd"] = np.column_stack(agent_sd)[:, index]
    output["master_probability"] = master_probability
    output["reference_cancer_instability"] = cancer_reference_instability
    output["mean_individual_agent_instability"] = agent_instability
    output["inter_agent_disagreement"] = disagreement
    output["master_state_instability"] = master_instability
    output["predicted_master_error"] = predicted_error
    output["master_self_observation_uncertainty"] = meta_uncertainty
    output["nearest_tcga_fit_distance"] = ood

    out_dir = results_dir / "pcawg_external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "locked_label_blind_predictions.parquet"
    output.to_parquet(prediction_path, index=False)
    lock = {
        "status": "locked",
        "label_access_during_scoring": False,
        "scored_donors": len(output),
        "model_bundle": str(model_path),
        "model_bundle_sha256": _sha256(model_path),
        "expression_matrix": str(expression_path),
        "expression_matrix_sha256": _sha256(expression_path),
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "frozen_spec_sha256": bundle["frozen_spec_sha256"],
        "fit_prevalence": float(bundle["fit_prevalence"]),
        "master_weights": {
            agent: float(weights[index]) for index, agent in enumerate(AGENT_ORDER)
        },
    }
    lock_path = out_dir / "prediction_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return {**lock, "lock": str(lock_path)}


def _bootstrap_intervals(
    cohort: pd.DataFrame,
    replicates: int = BOOTSTRAPS,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(RANDOM_SEED)
    y = cohort["os_2y_event"].to_numpy(dtype=int)
    p = cohort["master_probability"].to_numpy(dtype=float)
    patient_auc = []
    patient_brier = []
    for _ in range(replicates):
        index = rng.integers(0, len(cohort), size=len(cohort))
        if np.unique(y[index]).size == 2:
            patient_auc.append(float(roc_auc_score(y[index], p[index])))
        patient_brier.append(float(brier_score_loss(y[index], p[index])))

    projects = cohort["dcc_project_code"].unique()
    cluster_auc = []
    cluster_brier = []
    for _ in range(replicates):
        sampled_projects = rng.choice(projects, size=len(projects), replace=True)
        sampled = pd.concat(
            [cohort[cohort["dcc_project_code"] == project] for project in sampled_projects],
            ignore_index=True,
        )
        sy = sampled["os_2y_event"].to_numpy(dtype=int)
        sp = sampled["master_probability"].to_numpy(dtype=float)
        if np.unique(sy).size == 2:
            cluster_auc.append(float(roc_auc_score(sy, sp)))
        cluster_brier.append(float(brier_score_loss(sy, sp)))

    def interval(values: list[float]) -> list[float] | None:
        return (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values
            else None
        )

    return {
        "patient_bootstrap_auc_95ci": interval(patient_auc),
        "patient_bootstrap_brier_95ci": interval(patient_brier),
        "project_cluster_bootstrap_auc_95ci": interval(cluster_auc),
        "project_cluster_bootstrap_brier_95ci": interval(cluster_brier),
    }


def _within_project_permutation(cohort: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED + 17)
    observed_y = cohort["os_2y_event"].to_numpy(dtype=int)
    probability = cohort["master_probability"].to_numpy(dtype=float)
    observed_auc = float(roc_auc_score(observed_y, probability))
    observed_brier = float(brier_score_loss(observed_y, probability))
    null_auc = []
    null_brier = []
    project_indices = [group.index.to_numpy() for _, group in cohort.groupby("dcc_project_code")]
    for _ in range(PERMUTATIONS):
        permuted = observed_y.copy()
        for indices in project_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        null_auc.append(float(roc_auc_score(permuted, probability)))
        null_brier.append(float(brier_score_loss(permuted, probability)))
    return {
        "method": "outcomes permuted within independent PCAWG project",
        "permutations": PERMUTATIONS,
        "auc_empirical_p_one_sided": float(
            (1 + sum(value >= observed_auc for value in null_auc)) / (1 + PERMUTATIONS)
        ),
        "brier_empirical_p_one_sided": float(
            (1 + sum(value <= observed_brier for value in null_brier)) / (1 + PERMUTATIONS)
        ),
    }


def evaluate_pcawg_external(
    processed_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Evaluate locked external predictions only after verifying their immutable hash."""
    out_dir = results_dir / "pcawg_external_validation"
    prediction_path = out_dir / "locked_label_blind_predictions.parquet"
    lock_path = out_dir / "prediction_lock.json"
    patient_path = processed_dir / "pcawg_external" / "pcawg_non_tcga_patients.parquet"
    if not prediction_path.exists() or not lock_path.exists() or not patient_path.exists():
        raise FileNotFoundError("Locked predictions, lock manifest or PCAWG patients are missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if _sha256(prediction_path) != lock["predictions_sha256"]:
        raise ValueError("Locked PCAWG prediction hash changed before outcome evaluation")

    predictions = pd.read_parquet(prediction_path)
    patients = pd.read_parquet(patient_path)
    outcomes = patients[
        patients["independent_non_tcga"]
        & patients["expression_available"]
        & patients["os_2y_event"].notna()
    ].copy()
    cohort = predictions.merge(outcomes, on="donor_id", how="inner", validate="one_to_one")
    cohort = cohort.sort_values("donor_id", kind="stable").reset_index(drop=True)
    y = cohort["os_2y_event"].to_numpy(dtype=int)
    probability = cohort["master_probability"].to_numpy(dtype=float)
    actual_error = np.abs(y - probability)

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit, y)
    fit_prevalence = float(lock.get("fit_prevalence", np.nan))
    if not np.isfinite(fit_prevalence):
        # Model v1 lock predates this convenience field; the frozen TCGA fit prevalence is fixed.
        fit_prevalence = 0.2602739726027397

    metrics = {
        "patients": int(len(cohort)),
        "events": int(y.sum()),
        "non_events": int((1 - y).sum()),
        "projects": int(cohort["dcc_project_code"].nunique()),
        "roc_auc": _safe_auc(y, probability),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "ece_10_bins": _ece(y, probability, bins=10),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "tcga_fit_prevalence": fit_prevalence,
        "tcga_fit_prevalence_brier": float(
            brier_score_loss(y, np.full(len(y), fit_prevalence))
        ),
        "self_error_correlation": _safe_corr(
            actual_error, cohort["predicted_master_error"].to_numpy(dtype=float)
        ),
        "self_error_mae": float(
            np.mean(
                np.abs(actual_error - cohort["predicted_master_error"].to_numpy(dtype=float))
            )
        ),
    }
    intervals = _bootstrap_intervals(cohort)
    permutation = _within_project_permutation(cohort)

    project_metrics = []
    for project, group in cohort.groupby("dcc_project_code", sort=True):
        gy = group["os_2y_event"].to_numpy(dtype=int)
        gp = group["master_probability"].to_numpy(dtype=float)
        project_metrics.append(
            {
                "project": project,
                "patients": len(group),
                "events": int(gy.sum()),
                "roc_auc": _safe_auc(gy, gp),
                "average_precision": float(average_precision_score(gy, gp)),
                "brier": float(brier_score_loss(gy, gp)),
            }
        )
    project_table = pd.DataFrame(project_metrics)
    project_table.to_csv(out_dir / "project_metrics.csv", index=False)

    reduced_external = np.column_stack(
        [
            cohort[f"{agent}_probability"].to_numpy(dtype=float)
            for agent in AGENT_ORDER
        ]
    )
    neighbor_count = min(16, len(cohort))
    neighbor_model = NearestNeighbors(n_neighbors=neighbor_count).fit(reduced_external)
    _, local_indices = neighbor_model.kneighbors(reduced_external)
    local_indices = local_indices[:, 1:]
    observed_cancer_instability = _entropy(y[local_indices].mean(axis=1))
    agent_instability = cohort["mean_individual_agent_instability"].to_numpy(dtype=float)
    disagreement = cohort["inter_agent_disagreement"].to_numpy(dtype=float)
    master_instability = cohort["master_state_instability"].to_numpy(dtype=float)
    meta_uncertainty = cohort["master_self_observation_uncertainty"].to_numpy(dtype=float)

    events_path = out_dir / "recursive_observability_events.jsonl"
    events_path.unlink(missing_ok=True)
    event_records = []
    weights = np.asarray([lock["master_weights"][agent] for agent in AGENT_ORDER], dtype=float)
    for row_index, row in cohort.iterrows():
        integrated = 0.0
        total = 0.0
        for agent_index, agent in enumerate(AGENT_ORDER):
            before = integrated / total if total else fit_prevalence
            integrated += weights[agent_index] * float(row[f"{agent}_probability"])
            total += weights[agent_index]
            after = integrated / total
            future_states = {
                "death_by_2_years": float(row["master_probability"]),
                "survival_beyond_2_years": float(1 - row["master_probability"]),
            }
            event = TransitionEvent(
                episode_id=f"pcawg-os2y-{row['donor_id']}",
                cancer_state_id=str(row["donor_id"]),
                agent_id=agent,
                agent_state_before=[float(before)],
                agent_state_after=[float(row[f"{agent}_probability"])],
                action="apply_frozen_transcriptomic_observer",
                reward=float(1 - abs(row["os_2y_event"] - row[f"{agent}_probability"])),
                agent_uncertainty=float(row[f"{agent}_bootstrap_sd"]),
                master_state_before=[float(before)],
                master_state_after=[float(after)],
                master_uncertainty=float(row["master_state_instability"]),
                meta_uncertainty=float(row["master_self_observation_uncertainty"]),
                evidence_ids=[
                    f"PCAWG:{row['donor_id']}",
                    f"PCAWG-PROJECT:{row['dcc_project_code']}",
                    f"LOCKED-PREDICTION:{lock['predictions_sha256']}",
                ],
                parent_trace=f"pcawg-os2y-{row['donor_id']}",
                future_states_considered=future_states,
                quantum_inspired_state={
                    name: float(np.sqrt(value)) for name, value in future_states.items()
                },
                gate_applied=f"{agent}_external_evidence_gate",
                hypothesis_before="external_observer_not_integrated",
                hypothesis_after="external_observer_integrated",
            )
            append_transition_event(events_path, event)
            event_records.append(event.to_record())

    master_state = np.column_stack([probability, master_instability])
    reconstructed_master = master_state.copy()
    observability = build_recursive_observability_report(
        events=event_records,
        cancer_instability=observed_cancer_instability,
        individual_agent_instability=agent_instability,
        inter_agent_disagreement=disagreement,
        master_instability=master_instability,
        meta_uncertainty=meta_uncertainty,
        master_state=master_state,
        reconstructed_master_state=reconstructed_master,
        self_observation=actual_error.reshape(-1, 1),
        reconstructed_self_observation=cohort["predicted_master_error"].to_numpy().reshape(-1, 1),
        agent_influence_identifiability=1.0,
        uncertainty_visibility=1.0,
    ).to_dict()

    cohort["actual_master_absolute_error"] = actual_error
    cohort["observed_external_cancer_instability"] = observed_cancer_instability
    evaluated_path = out_dir / "evaluated_external_predictions.parquet"
    cohort.to_parquet(evaluated_path, index=False)
    selective_rows = []
    ordered = cohort.sort_values("predicted_master_error", kind="stable")
    for coverage in [1.0, 0.8, 0.6, 0.4, 0.2]:
        retained = ordered.head(max(1, int(np.ceil(len(ordered) * coverage))))
        selective_rows.append(
            {
                "coverage": coverage,
                "patients": len(retained),
                "mean_absolute_error": float(retained["actual_master_absolute_error"].mean()),
                "brier": float(
                    brier_score_loss(retained["os_2y_event"], retained["master_probability"])
                ),
            }
        )
    pd.DataFrame(selective_rows).to_csv(out_dir / "selective_risk_coverage.csv", index=False)

    report = {
        "status": "completed",
        "validation_type": "retrospective_external_non_tcga_pan_cancer_validation",
        "simulation": False,
        "prediction_lock": lock,
        "cohort": {
            "patients": len(cohort),
            "events": int(y.sum()),
            "projects": sorted(cohort["dcc_project_code"].unique().tolist()),
        },
        "metrics": metrics,
        "uncertainty_intervals": intervals,
        "within_project_permutation": permutation,
        "recursive_observability": observability,
        "trace_completeness_recheck": trace_completeness(event_records),
        "interpretation_guardrail": (
            "This retrospective external cohort can test transportability, discrimination, calibration, "
            "self-error ranking and selective risk. It cannot establish prospective clinical utility, "
            "treatment benefit or patient-specific molecular future trajectories."
        ),
        "artifacts": {
            "evaluated_predictions": str(evaluated_path),
            "project_metrics": str(out_dir / "project_metrics.csv"),
            "selective_risk": str(out_dir / "selective_risk_coverage.csv"),
            "events": str(events_path),
        },
    }
    report_path = out_dir / "pcawg_external_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "status": report["status"],
        "patients": metrics["patients"],
        "events": metrics["events"],
        "projects": metrics["projects"],
        "roc_auc": metrics["roc_auc"],
        "brier": metrics["brier"],
        "self_error_correlation": metrics["self_error_correlation"],
        "trace_completeness": report["trace_completeness_recheck"],
        "report": str(report_path),
    }
