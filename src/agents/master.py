from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.observability import TransitionEvent, append_transition_event, build_recursive_observability_report
from src.observability.recursive import trace_completeness


AGENT_ORDER = ("expression", "dependency", "pharmacology")
SOURCE_EVIDENCE = {
    "expression": "depmap_expression_top_genes.parquet",
    "dependency": "depmap_crispr_gene_effect_top_genes.parquet",
    "pharmacology": "prism_primary_top_treatments.parquet",
}


def _numeric_matrix(table: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    aligned = table.loc[model_ids].apply(pd.to_numeric, errors="coerce")
    return SimpleImputer(strategy="median").fit_transform(aligned)


def _agent_representation(values: np.ndarray, components: int) -> tuple[np.ndarray, float]:
    standardized = StandardScaler().fit_transform(values)
    component_count = min(components, standardized.shape[0] - 1, standardized.shape[1])
    if component_count < 2:
        raise ValueError("insufficient real observations for agent-state decomposition")
    pca = PCA(n_components=component_count, svd_solver="full")
    state = pca.fit_transform(standardized)
    state = StandardScaler().fit_transform(state)
    return state, float(pca.explained_variance_ratio_.sum())


def _normalized_entropy(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    denominator = absolute.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        absolute,
        denominator,
        out=np.full_like(absolute, 1.0 / absolute.shape[1]),
        where=denominator > 0,
    )
    entropy = -(probabilities * np.log(probabilities + np.finfo(float).eps)).sum(axis=1)
    return entropy / np.log(values.shape[1])


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    low, high = np.nanmin(values[finite]), np.nanmax(values[finite])
    if high == low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -50, 50))
    return exp / exp.sum()


def _pairwise_disagreement(states: dict[str, np.ndarray]) -> np.ndarray:
    pairs = []
    for left_index, left_name in enumerate(AGENT_ORDER):
        for right_name in AGENT_ORDER[left_index + 1 :]:
            pairs.append(
                np.linalg.norm(states[left_name] - states[right_name], axis=1)
                / np.sqrt(states[left_name].shape[1])
            )
    return np.mean(np.vstack(pairs), axis=0)


def _lineages(processed_dir: Path, model_ids: list[str]) -> pd.Series:
    metadata = pd.read_parquet(
        processed_dir / "depmap_figshare" / "depmap_model_metadata.parquet"
    )
    id_column = "ModelID" if "ModelID" in metadata.columns else metadata.columns[0]
    lineage_column = next(
        (
            column
            for column in ("OncotreeLineage", "OncotreePrimaryDisease", "lineage")
            if column in metadata.columns
        ),
        None,
    )
    if lineage_column is None:
        raise ValueError("DepMap metadata has no supported cancer-lineage column")
    lineage = metadata.set_index(id_column)[lineage_column].fillna("unknown").astype(str)
    return lineage.reindex(model_ids).fillna("unknown")


def _oof_reconstruct(observed: np.ndarray, target: np.ndarray) -> np.ndarray:
    folds = KFold(n_splits=5, shuffle=True, random_state=20260824)
    return cross_val_predict(Ridge(alpha=1.0), observed, target, cv=folds)


def _top_observed_futures(
    sensitivity_row: pd.Series,
    count: int = 5,
) -> tuple[dict[str, float], dict[str, float]]:
    observed = pd.to_numeric(sensitivity_row, errors="coerce").dropna().nlargest(count)
    if observed.empty:
        return {"unresolved": 1.0}, {"unresolved": 1.0}
    probabilities = _softmax(observed.to_numpy(dtype=float))
    futures = {str(name): float(value) for name, value in zip(observed.index, probabilities)}
    amplitudes = {name: float(np.sqrt(value)) for name, value in futures.items()}
    return futures, amplitudes


def run_real_master_brain(
    processed_dir: Path,
    results_dir: Path,
    *,
    agent_components: int = 6,
    master_components: int = 8,
) -> dict[str, Any]:
    """Run a first Master Brain exclusively on observed public pan-cancer matrices."""
    depmap_dir = processed_dir / "depmap_figshare"
    prism_dir = processed_dir / "prism_repurposing_24q2"
    expression = pd.read_parquet(depmap_dir / "depmap_expression_top_genes.parquet").set_index(
        "ModelID"
    )
    dependency = -pd.read_parquet(
        depmap_dir / "depmap_crispr_gene_effect_top_genes.parquet"
    ).set_index("ModelID")
    sensitivity = -pd.read_parquet(
        prism_dir / "prism_primary_top_treatments.parquet"
    ).set_index("ModelID").apply(pd.to_numeric, errors="coerce")

    model_ids = sorted(set(expression.index) & set(dependency.index) & set(sensitivity.index))
    if len(model_ids) < 50:
        raise ValueError("fewer than 50 real models share all three modalities")
    lineages = _lineages(processed_dir, model_ids)

    raw_values = {
        "expression": _numeric_matrix(expression, model_ids),
        "dependency": _numeric_matrix(dependency, model_ids),
        "pharmacology": _numeric_matrix(sensitivity, model_ids),
    }
    agent_states: dict[str, np.ndarray] = {}
    explained_variance: dict[str, float] = {}
    for name in AGENT_ORDER:
        agent_states[name], explained_variance[name] = _agent_representation(
            raw_values[name], agent_components
        )

    individual_instability = {
        name: _normalized_entropy(agent_states[name]) for name in AGENT_ORDER
    }
    mean_agent_instability = np.mean(np.vstack(list(individual_instability.values())), axis=0)
    disagreement = _pairwise_disagreement(agent_states)
    disagreement_normalized = _minmax(disagreement)

    # This is explicitly an observed multi-modal proxy, not a known biological truth.
    dependency_dispersion = _minmax(np.nanstd(raw_values["dependency"], axis=1))
    response_dispersion = _minmax(np.nanstd(raw_values["pharmacology"], axis=1))
    cancer_instability_proxy = np.mean(
        np.vstack([dependency_dispersion, response_dispersion]), axis=0
    )

    concatenated_agents = np.concatenate([agent_states[name] for name in AGENT_ORDER], axis=1)
    master_component_count = min(
        master_components, concatenated_agents.shape[0] - 1, concatenated_agents.shape[1]
    )
    master_model = PCA(n_components=master_component_count, svd_solver="full")
    master_state = master_model.fit_transform(concatenated_agents)
    master_state = StandardScaler().fit_transform(master_state)

    agent_width = next(iter(agent_states.values())).shape[1]
    counterfactual_distances = np.zeros((len(model_ids), len(AGENT_ORDER)), dtype=float)
    for agent_index, _ in enumerate(AGENT_ORDER):
        ablated = concatenated_agents.copy()
        start = agent_index * agent_width
        ablated[:, start : start + agent_width] = 0.0
        ablated_master = master_model.transform(ablated)
        ablated_master = (ablated_master - master_model.transform(concatenated_agents).mean(axis=0)) / (
            master_model.transform(concatenated_agents).std(axis=0) + np.finfo(float).eps
        )
        counterfactual_distances[:, agent_index] = np.linalg.norm(
            master_state - ablated_master, axis=1
        )
    influence_total = counterfactual_distances.sum(axis=1, keepdims=True)
    influence = np.divide(
        counterfactual_distances,
        influence_total,
        out=np.full_like(counterfactual_distances, 1 / len(AGENT_ORDER)),
        where=influence_total > 0,
    )
    influence_entropy = -(influence * np.log(influence + np.finfo(float).eps)).sum(axis=1) / np.log(
        len(AGENT_ORDER)
    )
    master_uncertainty = np.mean(
        np.vstack([mean_agent_instability, disagreement_normalized, influence_entropy]), axis=0
    )

    leave_one_uncertainty = []
    for excluded_index, excluded_name in enumerate(AGENT_ORDER):
        retained = [name for name in AGENT_ORDER if name != excluded_name]
        retained_instability = np.mean(
            np.vstack([individual_instability[name] for name in retained]), axis=0
        )
        retained_disagreement = _minmax(
            np.linalg.norm(agent_states[retained[0]] - agent_states[retained[1]], axis=1)
            / np.sqrt(agent_width)
        )
        retained_influence = np.delete(influence, excluded_index, axis=1)
        retained_influence /= retained_influence.sum(axis=1, keepdims=True)
        retained_entropy = -(
            retained_influence * np.log(retained_influence + np.finfo(float).eps)
        ).sum(axis=1) / np.log(len(retained))
        leave_one_uncertainty.append(
            np.mean(
                np.vstack([retained_instability, retained_disagreement, retained_entropy]), axis=0
            )
        )
    meta_uncertainty = np.std(np.vstack(leave_one_uncertainty), axis=0)

    observable_telemetry = np.column_stack(
        [concatenated_agents, mean_agent_instability, disagreement_normalized, influence]
    )
    reconstructed_master = _oof_reconstruct(observable_telemetry, master_state)
    self_observation = np.column_stack(
        [master_uncertainty, meta_uncertainty, disagreement_normalized, influence]
    )
    reconstructed_self = _oof_reconstruct(observable_telemetry, self_observation)

    out_dir = results_dir / "master_brain"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "recursive_observability_events.jsonl"
    events_path.unlink(missing_ok=True)
    event_records = []
    raw_master = master_model.transform(concatenated_agents)
    master_mean = raw_master.mean(axis=0)
    master_std = raw_master.std(axis=0) + np.finfo(float).eps
    for model_index, model_id in enumerate(model_ids):
        integrated = np.zeros(concatenated_agents.shape[1], dtype=float)
        future_states, amplitudes = _top_observed_futures(sensitivity.loc[model_id])
        for agent_index, agent_name in enumerate(AGENT_ORDER):
            before_master = (master_model.transform(integrated.reshape(1, -1))[0] - master_mean) / master_std
            start = agent_index * agent_width
            integrated[start : start + agent_width] = agent_states[agent_name][model_index]
            after_master = (master_model.transform(integrated.reshape(1, -1))[0] - master_mean) / master_std
            event = TransitionEvent(
                episode_id=f"real-master-{model_id}",
                cancer_state_id=model_id,
                agent_id=agent_name,
                agent_state_before=np.zeros(agent_width).tolist(),
                agent_state_after=agent_states[agent_name][model_index].tolist(),
                action="integrate_observed_modality",
                reward=float(influence[model_index, agent_index]),
                agent_uncertainty=float(individual_instability[agent_name][model_index]),
                master_state_before=before_master.tolist(),
                master_state_after=after_master.tolist(),
                master_uncertainty=float(master_uncertainty[model_index]),
                meta_uncertainty=float(meta_uncertainty[model_index]),
                evidence_ids=[SOURCE_EVIDENCE[agent_name], f"ModelID:{model_id}"],
                parent_trace=f"real-master-{model_id}",
                future_states_considered=future_states,
                quantum_inspired_state=amplitudes,
                gate_applied=f"{agent_name}_observed_evidence_gate",
                hypothesis_before="modality_not_integrated",
                hypothesis_after="observed_modality_integrated",
            )
            append_transition_event(events_path, event)
            event_records.append(event.to_record())

    report = build_recursive_observability_report(
        events=event_records,
        cancer_instability=cancer_instability_proxy,
        individual_agent_instability=mean_agent_instability,
        inter_agent_disagreement=disagreement_normalized,
        master_instability=master_uncertainty,
        meta_uncertainty=meta_uncertainty,
        master_state=master_state,
        reconstructed_master_state=reconstructed_master,
        self_observation=self_observation,
        reconstructed_self_observation=reconstructed_self,
        agent_influence_identifiability=float(np.mean(np.isfinite(influence).all(axis=1))),
        uncertainty_visibility=float(np.mean(np.isfinite(self_observation).all(axis=1))),
    )

    agent_table = pd.DataFrame({"ModelID": model_ids, "lineage": lineages.to_numpy()})
    for name in AGENT_ORDER:
        for component in range(agent_states[name].shape[1]):
            agent_table[f"{name}_state_{component + 1}"] = agent_states[name][:, component]
        agent_table[f"{name}_instability"] = individual_instability[name]
    agent_path = out_dir / "real_agent_states.parquet"
    agent_table.to_parquet(agent_path, index=False)

    master_table = pd.DataFrame({"ModelID": model_ids, "lineage": lineages.to_numpy()})
    for component in range(master_state.shape[1]):
        master_table[f"master_state_{component + 1}"] = master_state[:, component]
        master_table[f"reconstructed_master_state_{component + 1}"] = reconstructed_master[:, component]
    self_names = (
        "master_uncertainty",
        "meta_uncertainty",
        "inter_agent_disagreement",
        *[f"{name}_influence" for name in AGENT_ORDER],
    )
    for component, name in enumerate(self_names):
        master_table[f"reconstructed_self_{name}"] = reconstructed_self[:, component]
    master_table["cancer_instability_proxy"] = cancer_instability_proxy
    master_table["mean_individual_agent_instability"] = mean_agent_instability
    master_table["inter_agent_disagreement"] = disagreement_normalized
    master_table["master_uncertainty"] = master_uncertainty
    master_table["meta_uncertainty"] = meta_uncertainty
    for index, name in enumerate(AGENT_ORDER):
        master_table[f"{name}_influence"] = influence[:, index]
    master_path = out_dir / "real_master_states.parquet"
    master_table.to_parquet(master_path, index=False)

    report_path = out_dir / "recursive_observability_report.json"
    report_payload = report.to_dict()
    report_payload.update(
        {
            "data_policy": "observed_public_pan_cancer_data_only",
            "cancer_instability_label": "empirical_multi_modal_instability_proxy",
            "models": len(model_ids),
            "lineages": int(lineages.nunique()),
            "agents": list(AGENT_ORDER),
            "agent_explained_variance": explained_variance,
            "master_explained_variance": float(master_model.explained_variance_ratio_.sum()),
            "transition_events": len(event_records),
            "trace_completeness_recheck": trace_completeness(event_records),
            "artifacts": {
                "agent_states": str(agent_path),
                "master_states": str(master_path),
                "events": str(events_path),
            },
            "interpretation": (
                "Engineering prototype on observed cell-line data. Instability and coupling values "
                "are scientific hypotheses requiring external temporal and biological validation."
            ),
        }
    )
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return {
        "status": "created",
        "models": len(model_ids),
        "lineages": int(lineages.nunique()),
        "transition_events": len(event_records),
        "trace_completeness": report_payload["trace_completeness_recheck"],
        "rmoi_profile": report_payload["rmoi_profile"],
        "osic": report_payload["osic"],
        "oosc": report_payload["oosc"],
        "observer_depth": report_payload["observer_depth"],
        "report": str(report_path),
        "artifacts": report_payload["artifacts"],
        "note": report_payload["interpretation"],
    }
