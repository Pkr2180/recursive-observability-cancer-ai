from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.observability.recursive import (
    TransitionEvent,
    append_transition_event,
    build_recursive_observability_report,
    trace_completeness,
)


RANDOM_SEED = 20260824
SPLIT_SALT = "tcga-cdr-os-2y-project-holdout-v1"
ENDPOINT = "os_2y_event"
HORIZON_DAYS = 730
TOP_GENES = 500
PCA_COMPONENTS = 24
BOOTSTRAPS = 24
PERMUTATIONS = 1000
BASELINE_TUMOR_TYPES = {
    "Primary Tumor",
    "Primary Blood Derived Cancer - Peripheral Blood",
}
AGENT_ORDER = ("linear_transcriptome", "nonlinear_forest", "patient_neighborhood")


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


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if upper_inclusive
            else probability < edges[index + 1]
        )
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean() - probability[mask].mean()))
    return float(total)


def _project_split(projects: list[str]) -> dict[str, list[str]]:
    ordered = sorted(
        projects,
        key=lambda project: hashlib.sha256(
            f"{SPLIT_SALT}|{project}".encode("utf-8")
        ).hexdigest(),
    )
    test_count = max(5, int(round(len(ordered) * 0.20)))
    calibration_count = max(4, int(round(len(ordered) * 0.15)))
    return {
        "test": sorted(ordered[:test_count]),
        "calibration": sorted(ordered[test_count : test_count + calibration_count]),
        "fit": sorted(ordered[test_count + calibration_count :]),
    }


def _agent_factory(kind: str, seed: int, fit_size: int) -> Callable[[], Any]:
    if kind == "linear_transcriptome":
        return lambda: LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )
    if kind == "nonlinear_forest":
        return lambda: RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if kind == "patient_neighborhood":
        neighbors = max(3, min(15, int(np.sqrt(fit_size))))
        return lambda: KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    raise ValueError(f"Unknown observer agent: {kind}")


def _bootstrap_agent(
    kind: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_calibration: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    calibration_predictions = []
    test_predictions = []
    models = []
    for bootstrap in range(BOOTSTRAPS):
        rng = np.random.default_rng(RANDOM_SEED + bootstrap * 101 + AGENT_ORDER.index(kind))
        indices = rng.integers(0, len(y_fit), size=len(y_fit))
        # Extremely unlikely for OS, but deterministic retry keeps every estimator binary.
        if np.unique(y_fit[indices]).size < 2:
            indices = np.arange(len(y_fit))
        model = _agent_factory(kind, RANDOM_SEED + bootstrap, len(indices))()
        model.fit(x_fit[indices], y_fit[indices])
        models.append(model)
        calibration_predictions.append(model.predict_proba(x_calibration)[:, 1])
        test_predictions.append(model.predict_proba(x_test)[:, 1])
    return np.vstack(calibration_predictions), np.vstack(test_predictions), models


def _local_outcome_instability(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_count = min(15, len(y_fit))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(x_fit)
    distances, indices = neighbors.kneighbors(x_target)
    local_event_rate = y_fit[indices].mean(axis=1)
    return _entropy(local_event_rate), distances[:, 0]


def _meta_observers(
    x_calibration: np.ndarray,
    error_calibration: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[Any]]:
    scaler = StandardScaler().fit(x_calibration)
    calibration_scaled = scaler.transform(x_calibration)
    test_scaled = scaler.transform(x_test)
    neighbor_count = max(3, min(10, int(np.sqrt(len(error_calibration)))))
    models = [
        Ridge(alpha=2.0),
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        KNeighborsRegressor(n_neighbors=neighbor_count, weights="distance"),
    ]
    predictions = []
    for model in models:
        model.fit(calibration_scaled, error_calibration)
        predictions.append(np.clip(model.predict(test_scaled), 0, 1))
    stacked = np.vstack(predictions)
    return stacked.mean(axis=0), stacked.std(axis=0), scaler, models


def _permutation_audit(
    y: np.ndarray,
    master_probability: np.ndarray,
    cancer_instability: np.ndarray,
    master_instability: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    observed_auc = _safe_auc(y, master_probability)
    observed_brier = float(brier_score_loss(y, master_probability))
    observed_osic = _safe_corr(cancer_instability, master_instability)
    null_auc = []
    null_brier = []
    null_osic = []
    for _ in range(PERMUTATIONS):
        permuted_y = rng.permutation(y)
        if np.unique(permuted_y).size == 2:
            null_auc.append(float(roc_auc_score(permuted_y, master_probability)))
        null_brier.append(float(brier_score_loss(permuted_y, master_probability)))
        permuted_instability = rng.permutation(cancer_instability)
        correlation = _safe_corr(permuted_instability, master_instability)
        if correlation is not None:
            null_osic.append(correlation)
    return {
        "method": "observed_value_permutation_not_synthetic_patient_generation",
        "permutations": PERMUTATIONS,
        "auc_empirical_p_one_sided": (
            float((1 + sum(value >= observed_auc for value in null_auc)) / (1 + len(null_auc)))
            if observed_auc is not None
            else None
        ),
        "brier_empirical_p_one_sided": float(
            (1 + sum(value <= observed_brier for value in null_brier)) / (1 + len(null_brier))
        ),
        "osic_empirical_p_two_sided": (
            float(
                (1 + sum(abs(value) >= abs(observed_osic) for value in null_osic))
                / (1 + len(null_osic))
            )
            if observed_osic is not None
            else None
        ),
    }


def run_tcga_outcome_observability_audit(
    processed_dir: Path,
    results_dir: Path,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Forecast real two-year OS with cancer-project holdout and O0-O6 telemetry."""
    clinical_path = processed_dir / "tcga_cdr" / "tcga_cdr_rnaseq_linked.parquet"
    tpm_path = (
        processed_dir
        / "tcga_gdc_public_rnaseq_index"
        / "gdc_rnaseq_tpm_matrix.parquet"
    )
    gene_meta_path = (
        processed_dir
        / "tcga_gdc_public_rnaseq_index"
        / "gdc_gene_metadata.parquet"
    )
    linked = pd.read_parquet(clinical_path)
    cohort = linked[
        linked["cdr_linked"]
        & linked["sample_type"].isin(BASELINE_TUMOR_TYPES)
        & linked[ENDPOINT].notna()
    ].copy()
    cohort = cohort.sort_values(
        ["case_submitter_id", "sample_type", "sample_id"], kind="stable"
    ).drop_duplicates("case_submitter_id", keep="first")
    cohort[ENDPOINT] = cohort[ENDPOINT].astype(int)
    if len(cohort) < 150 or cohort["project_id"].nunique() < 25:
        raise ValueError("Insufficient linked pan-cancer patients for a project-holdout audit")

    split = _project_split(sorted(cohort["project_id"].unique()))
    split_masks = {
        name: cohort["project_id"].isin(projects).to_numpy()
        for name, projects in split.items()
    }
    if any(mask.sum() == 0 for mask in split_masks.values()):
        raise ValueError("Frozen project split produced an empty partition")

    tpm = pd.read_parquet(tpm_path).set_index("gene_id")
    missing_samples = sorted(set(cohort["sample_id"]) - set(tpm.columns))
    if missing_samples:
        raise ValueError(f"TPM matrix is missing {len(missing_samples)} linked samples")
    expression = np.log2(1 + tpm[cohort["sample_id"].tolist()].T.to_numpy(dtype=np.float32))
    y = cohort[ENDPOINT].to_numpy(dtype=int)

    fit_mask = split_masks["fit"]
    calibration_mask = split_masks["calibration"]
    test_mask = split_masks["test"]
    variances = np.var(expression[fit_mask], axis=0)
    selected_indices = np.argsort(variances, kind="stable")[-min(TOP_GENES, expression.shape[1]) :]
    selected_indices = selected_indices[np.argsort(variances[selected_indices])[::-1]]
    selected_gene_ids = tpm.index.to_numpy()[selected_indices].astype(str)
    selected = expression[:, selected_indices]

    scaler = StandardScaler().fit(selected[fit_mask])
    scaled = scaler.transform(selected)
    component_count = min(PCA_COMPONENTS, fit_mask.sum() - 1, scaled.shape[1])
    pca = PCA(n_components=component_count, random_state=RANDOM_SEED).fit(scaled[fit_mask])
    reduced = pca.transform(scaled)

    agent_inputs = {
        "linear_transcriptome": scaled,
        "nonlinear_forest": selected,
        "patient_neighborhood": reduced,
    }
    calibration_bootstraps: dict[str, np.ndarray] = {}
    test_bootstraps: dict[str, np.ndarray] = {}
    fitted_agents: dict[str, list[Any]] = {}
    for agent in AGENT_ORDER:
        (
            calibration_bootstraps[agent],
            test_bootstraps[agent],
            fitted_agents[agent],
        ) = _bootstrap_agent(
            agent,
            agent_inputs[agent][fit_mask],
            y[fit_mask],
            agent_inputs[agent][calibration_mask],
            agent_inputs[agent][test_mask],
        )

    agent_calibration = np.column_stack(
        [calibration_bootstraps[agent].mean(axis=0) for agent in AGENT_ORDER]
    )
    agent_test = np.column_stack(
        [test_bootstraps[agent].mean(axis=0) for agent in AGENT_ORDER]
    )
    agent_instability_calibration = np.column_stack(
        [calibration_bootstraps[agent].std(axis=0) for agent in AGENT_ORDER]
    ).mean(axis=1)
    agent_instability_test = np.column_stack(
        [test_bootstraps[agent].std(axis=0) for agent in AGENT_ORDER]
    ).mean(axis=1)
    disagreement_calibration = agent_calibration.std(axis=1)
    disagreement_test = agent_test.std(axis=1)

    calibration_brier = np.array(
        [
            brier_score_loss(y[calibration_mask], agent_calibration[:, index])
            for index in range(len(AGENT_ORDER))
        ]
    )
    master_weights = (1 / (calibration_brier + 1e-9))
    master_weights /= master_weights.sum()
    master_calibration = agent_calibration @ master_weights
    master_test = agent_test @ master_weights
    master_instability_calibration = _entropy(master_calibration)
    master_instability_test = _entropy(master_test)

    cancer_instability_calibration, ood_calibration = _local_outcome_instability(
        reduced[fit_mask], y[fit_mask], reduced[calibration_mask]
    )
    cancer_instability_test, ood_test = _local_outcome_instability(
        reduced[fit_mask], y[fit_mask], reduced[test_mask]
    )
    ood_scaler = StandardScaler().fit(ood_calibration.reshape(-1, 1))
    meta_features_calibration = np.column_stack(
        [
            cancer_instability_calibration,
            agent_instability_calibration,
            disagreement_calibration,
            master_instability_calibration,
            ood_scaler.transform(ood_calibration.reshape(-1, 1)).ravel(),
        ]
    )
    meta_features_test = np.column_stack(
        [
            cancer_instability_test,
            agent_instability_test,
            disagreement_test,
            master_instability_test,
            ood_scaler.transform(ood_test.reshape(-1, 1)).ravel(),
        ]
    )
    calibration_error = np.abs(y[calibration_mask] - master_calibration)
    predicted_error, meta_uncertainty_test, meta_scaler, meta_models = _meta_observers(
        meta_features_calibration, calibration_error, meta_features_test
    )
    actual_error = np.abs(y[test_mask] - master_test)

    out_dir = results_dir / "tcga_outcome_observability"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "recursive_observability_events.jsonl"
    events_path.unlink(missing_ok=True)
    test_cohort = cohort.loc[test_mask].reset_index(drop=True)
    event_records = []
    for row_index, patient in test_cohort.iterrows():
        integrated_weight = 0.0
        integrated_probability = 0.0
        for agent_index, agent in enumerate(AGENT_ORDER):
            before = (
                integrated_probability / integrated_weight
                if integrated_weight > 0
                else float(y[fit_mask].mean())
            )
            integrated_probability += master_weights[agent_index] * agent_test[row_index, agent_index]
            integrated_weight += master_weights[agent_index]
            after = integrated_probability / integrated_weight
            future_states = {
                "death_by_2_years": float(master_test[row_index]),
                "survival_beyond_2_years": float(1 - master_test[row_index]),
            }
            event = TransitionEvent(
                episode_id=f"tcga-os2y-{patient['case_submitter_id']}",
                cancer_state_id=str(patient["case_submitter_id"]),
                agent_id=agent,
                agent_state_before=[float(before)],
                agent_state_after=[float(agent_test[row_index, agent_index])],
                action="integrate_baseline_transcriptomic_observer",
                reward=float(master_weights[agent_index]),
                agent_uncertainty=float(
                    test_bootstraps[agent][:, row_index].std()
                ),
                master_state_before=[float(before)],
                master_state_after=[float(after)],
                master_uncertainty=float(master_instability_test[row_index]),
                meta_uncertainty=float(meta_uncertainty_test[row_index]),
                evidence_ids=[
                    "GDC:1b5f413e-a8d1-4d10-92eb-7c4ae739ed81",
                    f"GDC-RNA:{patient['sample_id']}",
                    f"TCGA-CDR:{patient['case_submitter_id']}",
                ],
                parent_trace=f"tcga-os2y-{patient['case_submitter_id']}",
                future_states_considered=future_states,
                quantum_inspired_state={
                    name: float(np.sqrt(probability))
                    for name, probability in future_states.items()
                },
                gate_applied=f"{agent}_observed_evidence_gate",
                hypothesis_before="observer_not_integrated",
                hypothesis_after="observer_integrated",
            )
            append_transition_event(events_path, event)
            event_records.append(event.to_record())

    master_state = np.column_stack([master_test, master_instability_test])
    reconstructed_master = np.column_stack(
        [agent_test @ master_weights, _entropy(agent_test @ master_weights)]
    )
    observability = build_recursive_observability_report(
        events=event_records,
        cancer_instability=cancer_instability_test,
        individual_agent_instability=agent_instability_test,
        inter_agent_disagreement=disagreement_test,
        master_instability=master_instability_test,
        meta_uncertainty=meta_uncertainty_test,
        master_state=master_state,
        reconstructed_master_state=reconstructed_master,
        self_observation=actual_error.reshape(-1, 1),
        reconstructed_self_observation=predicted_error.reshape(-1, 1),
        agent_influence_identifiability=1.0,
        uncertainty_visibility=float(np.mean(np.isfinite(predicted_error))),
    ).to_dict()

    test_y = y[test_mask]
    fit_prevalence = float(y[fit_mask].mean())
    metrics = {
        "patients": int(len(test_y)),
        "events": int(test_y.sum()),
        "non_events": int((1 - test_y).sum()),
        "roc_auc": _safe_auc(test_y, master_test),
        "average_precision": float(average_precision_score(test_y, master_test)),
        "brier": float(brier_score_loss(test_y, master_test)),
        "log_loss": float(log_loss(test_y, master_test, labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(test_y, master_test >= 0.5)),
        "ece_5_bins": _ece(test_y, master_test),
        "fit_prevalence": fit_prevalence,
        "prevalence_baseline_brier": float(
            brier_score_loss(test_y, np.full(len(test_y), fit_prevalence))
        ),
        "mean_predicted_self_error": float(predicted_error.mean()),
        "mean_actual_absolute_error": float(actual_error.mean()),
        "self_error_mae": float(np.mean(np.abs(actual_error - predicted_error))),
        "self_error_correlation": _safe_corr(actual_error, predicted_error),
    }
    permutation = _permutation_audit(
        test_y, master_test, cancer_instability_test, master_instability_test
    )

    prediction_table = test_cohort[
        ["case_submitter_id", "project_id", "sample_id", "sample_type"]
    ].copy()
    prediction_table["observed_death_by_2y"] = test_y
    for index, agent in enumerate(AGENT_ORDER):
        prediction_table[f"{agent}_probability"] = agent_test[:, index]
    prediction_table["master_probability"] = master_test
    prediction_table["cancer_future_state_instability"] = cancer_instability_test
    prediction_table["mean_individual_agent_instability"] = agent_instability_test
    prediction_table["inter_agent_disagreement"] = disagreement_test
    prediction_table["master_state_instability"] = master_instability_test
    prediction_table["predicted_master_error"] = predicted_error
    prediction_table["master_self_observation_uncertainty"] = meta_uncertainty_test
    prediction_table["actual_master_absolute_error"] = actual_error
    predictions_path = out_dir / "heldout_project_predictions.parquet"
    prediction_table.to_parquet(predictions_path, index=False)

    gene_names: dict[str, str] = {}
    if gene_meta_path.exists():
        gene_meta = pd.read_parquet(gene_meta_path).drop_duplicates("gene_id")
        gene_names = dict(
            zip(gene_meta["gene_id"].astype(str), gene_meta["gene_name"].astype(str))
        )
    selected_genes_path = out_dir / "fit_only_selected_genes.csv"
    pd.DataFrame(
        {
            "gene_id": selected_gene_ids,
            "gene_name": [gene_names.get(gene_id, "") for gene_id in selected_gene_ids],
            "fit_variance_rank": np.arange(1, len(selected_gene_ids) + 1),
        }
    ).to_csv(selected_genes_path, index=False)

    cohort_profile = (
        cohort.groupby(["project_id", ENDPOINT], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "non_events", 1: "events"})
        .reset_index()
    )
    for column in ("non_events", "events"):
        if column not in cohort_profile:
            cohort_profile[column] = 0
    cohort_profile["eligible"] = cohort_profile["non_events"] + cohort_profile["events"]
    split_lookup = {project: name for name, projects in split.items() for project in projects}
    cohort_profile["partition"] = cohort_profile["project_id"].map(split_lookup)
    profile_path = out_dir / "endpoint_project_profile.csv"
    cohort_profile.to_csv(profile_path, index=False)

    frozen_spec = {
        "version": "tcga-cdr-os-2y-project-holdout-v1",
        "endpoint": "all-cause death by 730 days",
        "censoring_rule": (
            "event=1 when OS event occurs by day 730; event=0 only when observed through "
            "day 730; earlier event-free censoring excluded"
        ),
        "baseline_sample_types": sorted(BASELINE_TUMOR_TYPES),
        "patient_deduplication": "one stable-sorted baseline tumor RNA file per TCGA case",
        "split_unit": "TCGA cancer project",
        "split_salt": SPLIT_SALT,
        "fit_only_feature_selection": True,
        "top_genes": TOP_GENES,
        "pca_components": component_count,
        "agents": list(AGENT_ORDER),
        "bootstraps": BOOTSTRAPS,
        "bootstrap_note": "resampling observed fit patients; no synthetic patient generation",
        "master_weighting": "inverse calibration-project Brier score",
        "permutations": PERMUTATIONS,
        "simulation": False,
    }
    spec_serialized = json.dumps(frozen_spec, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(spec_serialized.encode("utf-8")).hexdigest()
    split_path = out_dir / "frozen_project_split.json"
    split_path.write_text(
        json.dumps({"spec_sha256": spec_hash, "spec": frozen_spec, "projects": split}, indent=2),
        encoding="utf-8",
    )

    model_artifact = None
    model_sha256 = None
    if model_dir is not None:
        model_out_dir = model_dir / "tcga_outcome_observability"
        model_out_dir.mkdir(parents=True, exist_ok=True)
        model_artifact = model_out_dir / "tcga_os2y_frozen_bundle.joblib"
        bundle = {
            "bundle_version": "tcga-os2y-frozen-bundle-v1",
            "frozen_spec_sha256": spec_hash,
            "endpoint": "all-cause death by 730 days",
            "selected_gene_ids": selected_gene_ids.tolist(),
            "selected_fit_medians": np.nanmedian(selected[fit_mask], axis=0),
            "selected_fit_means": np.nanmean(selected[fit_mask], axis=0),
            "scaler": scaler,
            "pca": pca,
            "fitted_agents": fitted_agents,
            "master_weights": master_weights,
            "master_weight_names": list(AGENT_ORDER),
            "fit_prevalence": fit_prevalence,
            "fit_reduced": reduced[fit_mask],
            "fit_outcomes": y[fit_mask],
            "ood_scaler": ood_scaler,
            "meta_scaler": meta_scaler,
            "meta_models": meta_models,
            "transform_contract": {
                "training_input": "GDC STAR TPM",
                "training_transform": "log2(1 + TPM)",
                "external_gene_mapping": "Ensembl stable ID with version removed",
                "external_missing_gene_rule": "impute TCGA fit median; require >=95% coverage",
                "external_pcawg_transform": (
                    "PCAWG log2(FPKM-UQ + 0.001) inverted to FPKM-UQ, then log2(1 + FPKM-UQ)"
                ),
            },
        }
        joblib.dump(bundle, model_artifact, compress=3)
        digest = hashlib.sha256()
        with model_artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        model_sha256 = digest.hexdigest()

    report = {
        "status": "completed",
        "data_policy": "official_observed_patient_data_only",
        "simulation": False,
        "scientific_scope": (
            "Real baseline tumor RNA to future patient outcome forecasting. This is not a "
            "repeated-molecular-state trajectory experiment."
        ),
        "frozen_spec_sha256": spec_hash,
        "endpoint": "two_year_overall_survival_event",
        "horizon_days": HORIZON_DAYS,
        "cohort": {
            "eligible_patients": int(len(cohort)),
            "projects": int(cohort["project_id"].nunique()),
            "fit_patients": int(fit_mask.sum()),
            "calibration_patients": int(calibration_mask.sum()),
            "test_patients": int(test_mask.sum()),
            "fit_events": int(y[fit_mask].sum()),
            "calibration_events": int(y[calibration_mask].sum()),
            "test_events": int(y[test_mask].sum()),
            "project_split": split,
        },
        "agent_calibration_brier": {
            agent: float(calibration_brier[index])
            for index, agent in enumerate(AGENT_ORDER)
        },
        "master_weights": {
            agent: float(master_weights[index]) for index, agent in enumerate(AGENT_ORDER)
        },
        "test_metrics": metrics,
        "permutation_audit": permutation,
        "recursive_observability": observability,
        "trace_completeness_recheck": trace_completeness(event_records),
        "transition_events": len(event_records),
        "instability_definitions": {
            "cancer_future_state": (
                "Bernoulli entropy of observed two-year outcomes among the 15 nearest fit patients"
            ),
            "individual_agents": (
                "mean within-agent prediction SD over 24 observed-patient bootstrap fits"
            ),
            "inter_agent_disagreement": "SD across the three observer-agent probabilities",
            "master_state": "Bernoulli entropy of the weighted Master probability",
            "master_self_observation_uncertainty": (
                "SD across three calibration-trained models predicting Master absolute error"
            ),
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "events": str(events_path),
            "selected_genes": str(selected_genes_path),
            "project_profile": str(profile_path),
            "frozen_split": str(split_path),
            "frozen_model_bundle": str(model_artifact) if model_artifact else None,
        },
        "frozen_model_bundle_sha256": model_sha256,
        "interpretation_guardrail": (
            "Predictive metrics must be reported whether favorable or unfavorable. Correlation "
            "between observability channels is not evidence that the observer causes cancer outcomes."
        ),
    }
    report_path = out_dir / "tcga_outcome_observability_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "status": report["status"],
        "simulation": False,
        "eligible_patients": report["cohort"]["eligible_patients"],
        "projects": report["cohort"]["projects"],
        "fit_calibration_test": [
            report["cohort"]["fit_patients"],
            report["cohort"]["calibration_patients"],
            report["cohort"]["test_patients"],
        ],
        "test_metrics": metrics,
        "rmoi_profile": observability["rmoi_profile"],
        "osic": observability["osic"],
        "observer_depth": observability["observer_depth"],
        "trace_completeness": report["trace_completeness_recheck"],
        "frozen_spec_sha256": spec_hash,
        "frozen_model_bundle_sha256": model_sha256,
        "report": str(report_path),
        "artifacts": report["artifacts"],
    }
