from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.observability.recursive import (
    OBSERVABILITY_LEVELS,
    REQUIRED_TRANSITION_FIELDS,
    trace_completeness,
    validate_transition_stream,
)


EVALUATION_PRIORITY = [
    "scientific_observability",
    "state_reconstructability",
    "agent_master_attribution",
    "recursive_self_observability",
    "cancer_ai_instability_coupling",
    "learning_and_prediction_performance_secondary",
]

RUN_SPECS = {
    "depmap_prism_master": {
        "report": "master_brain/recursive_observability_report.json",
        "events": "master_brain/recursive_observability_events.jsonl",
    },
    "lincs_future_state": {
        "report": "validation/lincs_future_state/lincs_future_state_recursive_observability_report.json",
        "events": "validation/lincs_future_state/frozen_holdout_recursive_events.jsonl",
    },
    "tcga_patient_outcome": {
        "report": "tcga_outcome_observability/tcga_outcome_observability_report.json",
        "events": "tcga_outcome_observability/recursive_observability_events.jsonl",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON event at {path}:{line_number}: {exc}") from exc
            event["_event_index"] = len(events)
            events.append(event)
    return events


def _observable_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = report.get("recursive_observability")
    payload = nested if isinstance(nested, Mapping) else report
    required = {"rmoi_profile", "instability_channels", "osic", "oosc", "observer_depth"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Recursive observability report is missing: {missing}")
    return payload


def _vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _safe_corr(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    length = min(len(left_array), len(right_array))
    if length < 3:
        return None
    left_array = left_array[:length]
    right_array = right_array[:length]
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if mask.sum() < 3 or np.std(left_array[mask]) == 0 or np.std(right_array[mask]) == 0:
        return None
    return float(np.corrcoef(left_array[mask], right_array[mask])[0, 1])


def _series_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _empirical_rank_ratio(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    final_by_episode: dict[str, np.ndarray] = {}
    for event in events:
        final_by_episode[str(event["episode_id"])] = _vector(event["master_state_after"])
    matrix = np.vstack(list(final_by_episode.values()))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered))
    dimension = int(matrix.shape[1])
    return {
        "empirical_master_state_telemetry_rank": rank,
        "master_state_dimension": dimension,
        "observability_rank_ratio": float(rank / dimension) if dimension else 0.0,
        "method_boundary": (
            "Rank of observed final Master-state telemetry across episodes. This is an empirical "
            "rank diagnostic, not a claim of a complete nonlinear dynamical observability Gramian."
        ),
    }


def _agentic_divergence(events: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    agents = sorted({str(event["agent_id"]) for event in events})
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    episodes: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for event in events:
        episodes[str(event["episode_id"])][str(event["agent_id"])] = _vector(
            event["agent_state_after"]
        )
    for states in episodes.values():
        for left_index, left in enumerate(agents):
            for right in agents[left_index + 1 :]:
                if left not in states or right not in states:
                    continue
                if states[left].shape != states[right].shape:
                    continue
                distance = np.linalg.norm(states[left] - states[right]) / np.sqrt(
                    states[left].size
                )
                pair_values[(left, right)].append(float(distance))
    matrix = pd.DataFrame(np.nan, index=agents, columns=agents, dtype=float)
    for agent in agents:
        matrix.loc[agent, agent] = 0.0
    rows = []
    for (left, right), values in pair_values.items():
        mean = float(np.mean(values))
        matrix.loc[left, right] = mean
        matrix.loc[right, left] = mean
        rows.append({"agent_left": left, "agent_right": right, "mean_distance": mean, "n": len(values)})
    profile = {
        "agents": agents,
        "episodes": len(episodes),
        "observable_agent_pairs": len(pair_values),
        "mean_pairwise_divergence": float(np.mean([row["mean_distance"] for row in rows]))
        if rows
        else None,
    }
    return matrix, profile


def _hypothesis_traceability(events: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, float]:
    rows = []
    traceable = 0
    for event in events:
        complete = all(
            [
                event.get("hypothesis_before"),
                event.get("hypothesis_after"),
                event.get("agent_id"),
                event.get("action"),
                event.get("evidence_ids"),
            ]
        )
        traceable += int(complete)
        rows.append(
            {
                "hypothesis_before": event.get("hypothesis_before"),
                "hypothesis_after": event.get("hypothesis_after"),
                "agent_id": event.get("agent_id"),
                "action": event.get("action"),
                "count": 1,
                "traceable": bool(complete),
            }
        )
    graph = (
        pd.DataFrame(rows)
        .groupby(
            ["hypothesis_before", "hypothesis_after", "agent_id", "action", "traceable"],
            dropna=False,
            as_index=False,
        )["count"]
        .sum()
    )
    return graph, float(traceable / len(events)) if events else 0.0


def _evidence_provenance(events: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    edges = []
    complete = 0
    depths = []
    for event in events:
        episode = str(event["episode_id"])
        hypothesis = str(event["hypothesis_after"])
        agent = str(event["agent_id"])
        evidence = [str(item) for item in event.get("evidence_ids", []) if str(item).strip()]
        is_complete = bool(episode and hypothesis and agent and evidence and event.get("cancer_state_id"))
        complete += int(is_complete)
        depths.append(3 + len(evidence))
        edges.append({"source": episode, "target": hypothesis, "relation": "produced_hypothesis"})
        edges.append({"source": hypothesis, "target": agent, "relation": "attributed_to_agent"})
        for evidence_id in evidence:
            edges.append({"source": agent, "target": evidence_id, "relation": "supported_by_evidence"})
    edge_table = pd.DataFrame(edges).drop_duplicates()
    return edge_table, {
        "evidence_provenance_completeness": float(complete / len(events)) if events else 0.0,
        "mean_evidence_chain_depth": float(np.mean(depths)) if depths else 0.0,
        "unique_provenance_edges": int(len(edge_table)),
    }


def _aligned_amplitude_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    left_values = np.asarray([left.get(key, 0.0) for key in keys], dtype=float)
    right_values = np.asarray([right.get(key, 0.0) for key in keys], dtype=float)
    return float(np.linalg.norm(left_values - right_values))


def _quantum_and_gate_observability(
    events: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    previous_amplitude: dict[str, Mapping[str, float]] = {}
    gate_rows = []
    complete = 0
    normalized = 0
    reorganization = []
    for event in events:
        amplitudes = event.get("quantum_inspired_state", {})
        futures = event.get("future_states_considered", {})
        if amplitudes and futures and set(amplitudes) == set(futures):
            complete += 1
            squared = np.asarray(list(amplitudes.values()), dtype=float) ** 2
            normalized += int(np.isclose(squared.sum(), 1.0, atol=1e-5))
        episode = str(event["episode_id"])
        possibility_delta = None
        if episode in previous_amplitude:
            possibility_delta = _aligned_amplitude_distance(previous_amplitude[episode], amplitudes)
            reorganization.append(possibility_delta)
        previous_amplitude[episode] = amplitudes
        before = _vector(event["master_state_before"])
        after = _vector(event["master_state_after"])
        master_delta = float(np.linalg.norm(after - before) / np.sqrt(after.size))
        gate_rows.append(
            {
                "event_index": event["_event_index"],
                "episode_id": episode,
                "agent_id": event["agent_id"],
                "gate_applied": event["gate_applied"],
                "master_state_delta": master_delta,
                "possibility_amplitude_delta_from_prior_event": possibility_delta,
            }
        )
    total = len(events)
    profile = {
        "quantum_state_observability": float(complete / total) if total else 0.0,
        "normalized_amplitude_rate": float(normalized / total) if total else 0.0,
        "mean_possibility_reorganization": float(np.mean(reorganization)) if reorganization else 0.0,
        "interpretation_boundary": (
            "Amplitudes are a normalized computational representation of declared future-state "
            "weights, not evidence of physical quantum biology."
        ),
    }
    return pd.DataFrame(gate_rows), profile


def _state_dynamics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_episode: dict[str, list[np.ndarray]] = defaultdict(list)
    for event in events:
        by_episode[str(event["episode_id"])].append(_vector(event["master_state_after"]))
    threshold = 0.25
    switches = 0
    comparisons = 0
    persistence_runs = []
    for states in by_episode.values():
        current_run = 1
        for before, after in zip(states, states[1:]):
            delta = float(np.linalg.norm(after - before) / np.sqrt(after.size))
            comparisons += 1
            if delta >= threshold:
                switches += 1
                persistence_runs.append(current_run)
                current_run = 1
            else:
                current_run += 1
        persistence_runs.append(current_run)
    return {
        "state_switch_threshold_normalized_l2": threshold,
        "state_switching_rate_per_integration_step": float(switches / comparisons)
        if comparisons
        else 0.0,
        "mean_state_persistence_integration_steps": float(np.mean(persistence_runs))
        if persistence_runs
        else 0.0,
        "boundary": "Integration-step dynamics; not a biological-time switching rate.",
    }


def _uncertainty_trajectory(events: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_index": event["_event_index"],
                "episode_id": event["episode_id"],
                "time": event["time"],
                "cancer_state_id": event["cancer_state_id"],
                "agent_id": event["agent_id"],
                "agent_uncertainty": event["agent_uncertainty"],
                "master_uncertainty": event["master_uncertainty"],
                "meta_uncertainty": event["meta_uncertainty"],
            }
            for event in events
        ]
    )


def _index_lag_profile(cancer: Sequence[float], master: Sequence[float]) -> list[dict[str, Any]]:
    cancer_array = np.asarray(cancer, dtype=float)
    master_array = np.asarray(master, dtype=float)
    length = min(len(cancer_array), len(master_array))
    cancer_array = cancer_array[:length]
    master_array = master_array[:length]
    profile = []
    for lag in range(-5, 6):
        if lag < 0:
            left, right = cancer_array[-lag:], master_array[: length + lag]
        elif lag > 0:
            left, right = cancer_array[: length - lag], master_array[lag:]
        else:
            left, right = cancer_array, master_array
        profile.append({"observation_index_lag": lag, "correlation": _safe_corr(left, right)})
    return profile


def _multivariate_oosc(channels: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    names = ["cancer_future_state", "individual_agents", "master_state", "master_self_observation_uncertainty"]
    arrays = [np.asarray(channels[name], dtype=float) for name in names]
    length = min(map(len, arrays))
    matrix = np.column_stack([array[:length] for array in arrays])
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    if len(matrix) < 5 or np.any(matrix.std(axis=0) == 0):
        return {"gaussian_total_correlation": None, "observations": int(len(matrix))}
    correlation = np.corrcoef(matrix, rowvar=False)
    determinant = float(np.linalg.det(correlation))
    total_correlation = float(-0.5 * np.log(max(determinant, 1e-12)))
    return {
        "gaussian_total_correlation": total_correlation,
        "correlation_determinant": determinant,
        "observations": int(len(matrix)),
        "boundary": "Descriptive multivariate dependence; not causal direction or transfer entropy.",
    }


def _secondary_performance(report: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("test_metrics"), Mapping):
        return dict(report["test_metrics"])
    keys = (
        "master_holdout_rmse",
        "master_uncertainty_mae",
        "master_uncertainty_error_correlation",
        "models",
        "lineages",
    )
    return {key: report[key] for key in keys if key in report}


def _depth_profile(observable: Mapping[str, Any], strict_validity: float) -> list[dict[str, Any]]:
    rmoi = observable["rmoi_profile"]
    evidence = [
        strict_validity,
        float(rmoi.get("state_reconstructability", 0.0)),
        float(rmoi.get("agent_influence_identifiability", 0.0)),
        float(rmoi.get("state_reconstructability", 0.0)),
        float(rmoi.get("uncertainty_visibility", 0.0)),
        float(rmoi.get("recursive_consistency", 0.0)),
        float(np.isfinite(observable.get("osic", np.nan))),
    ]
    return [
        {"level": level, "name": OBSERVABILITY_LEVELS[level], "observable": value}
        for level, value in enumerate(evidence)
    ]


def _audit_one_run(
    run_name: str,
    report: Mapping[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    observable = _observable_payload(report)
    validation = validate_transition_stream(events)
    if validation["invalid_events"]:
        preview = validation["failures"][:5]
        raise ValueError(f"{run_name} has invalid scientific transitions: {preview}")
    if trace_completeness(events) < 1.0:
        raise ValueError(f"{run_name} trace completeness is below 1.0")

    output_dir.mkdir(parents=True, exist_ok=True)
    rank = _empirical_rank_ratio(events)
    divergence_matrix, divergence = _agentic_divergence(events)
    hypothesis_graph, hypothesis_traceability = _hypothesis_traceability(events)
    evidence_edges, evidence = _evidence_provenance(events)
    gate_trace, quantum = _quantum_and_gate_observability(events)
    state_dynamics = _state_dynamics(events)
    uncertainty = _uncertainty_trajectory(events)

    divergence_path = output_dir / "agentic_divergence_matrix.csv"
    hypothesis_path = output_dir / "hypothesis_transition_graph.csv"
    evidence_path = output_dir / "evidence_provenance_edges.csv"
    gate_path = output_dir / "gate_influence_trace.csv"
    uncertainty_path = output_dir / "uncertainty_trajectory.csv"
    depth_path = output_dir / "observer_depth_profile.csv"
    divergence_matrix.to_csv(divergence_path, index=True)
    hypothesis_graph.to_csv(hypothesis_path, index=False)
    evidence_edges.to_csv(evidence_path, index=False)
    gate_trace.to_csv(gate_path, index=False)
    uncertainty.to_csv(uncertainty_path, index=False)
    depth = _depth_profile(observable, validation["strict_validity"])
    pd.DataFrame(depth).to_csv(depth_path, index=False)

    rmoi = dict(observable["rmoi_profile"])
    channels = observable["instability_channels"]
    msio_profile = {
        "trace_completeness": float(rmoi.get("trace_completeness", 0.0)),
        "state_reconstructability": float(rmoi.get("state_reconstructability", 0.0)),
        "agent_attribution": float(rmoi.get("agent_influence_identifiability", 0.0)),
        "uncertainty_visibility": float(rmoi.get("uncertainty_visibility", 0.0)),
        "transition_visibility": validation["strict_validity"],
        "evidence_provenance_completeness": evidence["evidence_provenance_completeness"],
    }
    saoi_profile = {
        "state_reconstructability": msio_profile["state_reconstructability"],
        "trace_completeness": msio_profile["trace_completeness"],
        "agent_influence_identifiability": msio_profile["agent_attribution"],
        "hypothesis_transition_traceability": hypothesis_traceability,
        "evidence_provenance_completeness": evidence["evidence_provenance_completeness"],
        "quantum_state_observability": quantum["quantum_state_observability"],
    }
    channel_summary = {name: _series_summary(values) for name, values in channels.items()}
    lag = _index_lag_profile(channels["cancer_future_state"], channels["master_state"])
    report_payload = {
        "run": run_name,
        "primary_endpoint": "scientific_recursive_observability",
        "evaluation_priority": EVALUATION_PRIORITY,
        "architecture_name": "GOD-Observability: Global Observability of Distributed Reinforcement States",
        "architecture_guardrail": "A computational architecture name; no consciousness, sentience or divinity claim.",
        "required_transition_schema": list(REQUIRED_TRANSITION_FIELDS),
        "transition_validation": validation,
        "msoi_multidimensional_profile": msio_profile,
        "rmoi_multidimensional_profile": rmoi,
        "saoi_multidimensional_profile": saoi_profile,
        "composite_score_policy": "No post-hoc weighted composite is calculated.",
        "observability_rank": rank,
        "observer_depth": {
            "reported_depth": observable["observer_depth"],
            "levels": depth,
            "saturation": "requires interventions across deliberately varied recursion depths",
        },
        "agentic_divergence": divergence,
        "hypothesis_transition_traceability": hypothesis_traceability,
        "evidence_provenance": evidence,
        "uncertainty_observability": {
            "agent_master_meta_field_coverage": float(
                uncertainty[["agent_uncertainty", "master_uncertainty", "meta_uncertainty"]]
                .notna()
                .all(axis=1)
                .mean()
            ),
            "channel_summaries": channel_summary,
        },
        "state_dynamics": state_dynamics,
        "counterfactual_observability": {
            "agent_influence_identifiability": msio_profile["agent_attribution"],
            "declared_influence_signal_coverage": float(
                np.mean([np.isfinite(float(event["reward"])) for event in events])
            ),
            "boundary": "Identifiability/declared influence; causal intervention depends on each run design.",
        },
        "quantum_inspired_observability": quantum,
        "gate_influence": {
            "events": int(len(gate_trace)),
            "gates": sorted(gate_trace["gate_applied"].unique().tolist()),
            "mean_master_state_delta": float(gate_trace["master_state_delta"].mean()),
        },
        "osic": observable["osic"],
        "oosc_pairwise_profile": observable["oosc"],
        "oosc_multivariate_profile": _multivariate_oosc(channels),
        "index_lag_cancer_master_profile": lag,
        "lag_boundary": "Observation-index diagnostic only unless a run supplies ordered biological time.",
        "learning_and_prediction_performance_secondary": _secondary_performance(report),
        "artifacts": {
            "agentic_divergence_matrix": str(divergence_path),
            "hypothesis_transition_graph": str(hypothesis_path),
            "evidence_provenance_edges": str(evidence_path),
            "gate_influence_trace": str(gate_path),
            "uncertainty_trajectory": str(uncertainty_path),
            "observer_depth_profile": str(depth_path),
        },
    }
    report_path = output_dir / "primary_scientific_observability_report.json"
    report_payload["artifacts"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return report_payload


def _metric_dictionary() -> pd.DataFrame:
    rows = [
        ("TC", "Trace completeness", "Required transition fields present"),
        ("SR", "State reconstructability", "Recoverability of Master state from telemetry"),
        ("ORR", "Observability rank ratio", "Empirical Master-state telemetry rank / state dimension"),
        ("AII", "Agent influence identifiability", "Visibility of agent contribution to Master change"),
        ("ADI", "Agentic divergence", "Pairwise distance between observer-agent states"),
        ("UO", "Uncertainty observability", "Visibility of agent, Master and meta uncertainty"),
        ("HTT", "Hypothesis transition traceability", "Evidence-linked hypothesis lifecycle coverage"),
        ("EPC", "Evidence provenance completeness", "Complete episode-hypothesis-agent-evidence chain"),
        ("OSIC", "Observer-system instability coupling", "Cancer/Master instability dependence"),
        ("OOSC", "Observer-observer-system coupling", "Cancer/agent/Master/meta dependence profile"),
        ("SSR", "State switching rate", "Master changes per integration step"),
        ("SP", "State persistence", "Mean stable Master integration-step run"),
        ("CO", "Counterfactual observability", "Visibility of removal/influence signals"),
        ("QSO", "Quantum-inspired state observability", "Visibility of future-weight amplitudes"),
        ("GIT", "Gate influence trace", "Master/possibility changes associated with declared gates"),
        ("RMOI", "Recursive Master Observability Index", "Six-dimensional profile; not collapsed"),
        ("SAOI", "Scientific AI Observability Index", "Six-dimensional profile; not collapsed"),
    ]
    return pd.DataFrame(rows, columns=["metric", "name", "operational_question"])


def _dashboard_html(run_reports: Mapping[str, Mapping[str, Any]]) -> str:
    cards = []
    for run_name, report in run_reports.items():
        msoi = report["msoi_multidimensional_profile"]
        channel = report["uncertainty_observability"]["channel_summaries"]
        performance = report["learning_and_prediction_performance_secondary"]
        cards.append(
            f"""
            <section class="run">
              <h2>{html.escape(run_name)}</h2>
              <div class="grid">
                <article><h3>A. Cancer future-state field</h3><pre>{html.escape(json.dumps(channel.get('cancer_future_state', {}), indent=2))}</pre></article>
                <article><h3>B. Agentic divergence</h3><pre>{html.escape(json.dumps(report['agentic_divergence'], indent=2))}</pre></article>
                <article><h3>C. Individual-agent instability</h3><pre>{html.escape(json.dumps(channel.get('individual_agents', {}), indent=2))}</pre></article>
                <article><h3>D. Master uncertainty</h3><pre>{html.escape(json.dumps(channel.get('master_state', {}), indent=2))}</pre></article>
                <article><h3>E. Meta-uncertainty</h3><pre>{html.escape(json.dumps(channel.get('master_self_observation_uncertainty', {}), indent=2))}</pre></article>
                <article><h3>F. OSIC / OOSC</h3><pre>{html.escape(json.dumps({'osic': report['osic'], 'oosc': report['oosc_pairwise_profile']}, indent=2))}</pre></article>
                <article><h3>G. Hypothesis lifecycle</h3><p>HTT: {report['hypothesis_transition_traceability']:.3f}</p></article>
                <article><h3>H. Agent → Master influence</h3><pre>{html.escape(json.dumps(report['counterfactual_observability'], indent=2))}</pre></article>
                <article><h3>I. Possibility amplitudes / gates</h3><pre>{html.escape(json.dumps({'qso': report['quantum_inspired_observability'], 'gate': report['gate_influence']}, indent=2))}</pre></article>
                <article><h3>J. Evidence provenance</h3><pre>{html.escape(json.dumps(report['evidence_provenance'], indent=2))}</pre></article>
              </div>
              <h3>Primary MSOI profile</h3><pre>{html.escape(json.dumps(msoi, indent=2))}</pre>
              <details><summary>Secondary performance checks</summary><pre>{html.escape(json.dumps(performance, indent=2))}</pre></details>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Master Scientific Observability Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#07111f;color:#e7eef8}}header{{padding:28px;background:#10213a;border-bottom:1px solid #345}}
main{{padding:24px;max-width:1500px;margin:auto}}.run{{margin-bottom:36px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}}
article{{background:#10213a;border:1px solid #28425f;border-radius:10px;padding:14px}}h1,h2,h3{{color:#8de3ff}}pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}
.guard{{color:#ffd38d}}details{{background:#0c192c;padding:12px;border-radius:8px}}
</style></head><body><header><h1>Master Scientific Observability Dashboard</h1>
<p>Primary endpoint: observability, reconstructability, attribution and falsifiability.</p>
<p class="guard">GOD-Observability means Global Observability of Distributed Reinforcement States. It is not a consciousness, sentience or divinity claim.</p></header>
<main>{''.join(cards)}</main></body></html>"""


def run_primary_observability_audit(results_dir: Path) -> dict[str, Any]:
    """Fail-closed observability-first evaluation over every completed real-data Master run."""
    out_dir = results_dir / "observability_primary"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_reports: dict[str, dict[str, Any]] = {}
    for run_name, paths in RUN_SPECS.items():
        report_path = results_dir / paths["report"]
        events_path = results_dir / paths["events"]
        if not report_path.exists() or not events_path.exists():
            raise FileNotFoundError(f"Required observability inputs missing for {run_name}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        events = _read_jsonl(events_path)
        run_reports[run_name] = _audit_one_run(
            run_name, report, events, out_dir / run_name
        )

    evaluation_rows = []
    for run_name, report in run_reports.items():
        for profile_name in (
            "msoi_multidimensional_profile",
            "rmoi_multidimensional_profile",
            "saoi_multidimensional_profile",
        ):
            for metric, value in report[profile_name].items():
                evaluation_rows.append(
                    {"run": run_name, "profile": profile_name, "metric": metric, "value": value}
                )
    evaluation_path = out_dir / "primary_evaluation_table.csv"
    pd.DataFrame(evaluation_rows).to_csv(evaluation_path, index=False)
    dictionary_path = out_dir / "observability_metric_dictionary.csv"
    _metric_dictionary().to_csv(dictionary_path, index=False)

    dashboard_path = out_dir / "master_observability_dashboard.html"
    dashboard_path.write_text(_dashboard_html(run_reports), encoding="utf-8")
    schema_path = out_dir / "required_transition_schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "required_fields": list(REQUIRED_TRANSITION_FIELDS),
                "failure_policy": "scientific runs fail closed on missing or invalid transition telemetry",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    consolidated = {
        "status": "completed",
        "primary_endpoint": "scientific_recursive_observability",
        "evaluation_priority": EVALUATION_PRIORITY,
        "runs": {
            run_name: {
                "events": report["transition_validation"]["events"],
                "strict_transition_validity": report["transition_validation"]["strict_validity"],
                "msoi_multidimensional_profile": report["msoi_multidimensional_profile"],
                "rmoi_multidimensional_profile": report["rmoi_multidimensional_profile"],
                "saoi_multidimensional_profile": report["saoi_multidimensional_profile"],
                "osic": report["osic"],
                "observer_depth": report["observer_depth"]["reported_depth"],
                "secondary_performance": report["learning_and_prediction_performance_secondary"],
            }
            for run_name, report in run_reports.items()
        },
        "total_events": int(
            sum(report["transition_validation"]["events"] for report in run_reports.values())
        ),
        "composite_score_policy": "RMOI, MSOI and SAOI are reported as profiles; no arbitrary weighted total.",
        "scientific_guardrails": [
            "No synthetic biological samples or trajectories are used.",
            "Observability does not imply predictive accuracy or causal biological control.",
            "Quantum-inspired amplitudes are computational weights, not physical quantum claims.",
            "GOD-Observability is an acronym/metaphor, not a consciousness or divinity claim.",
        ],
        "artifacts": {
            "dashboard": str(dashboard_path),
            "evaluation_table": str(evaluation_path),
            "metric_dictionary": str(dictionary_path),
            "required_transition_schema": str(schema_path),
        },
    }
    consolidated_path = out_dir / "consolidated_primary_observability_report.json"
    consolidated["artifacts"]["report"] = str(consolidated_path)
    consolidated_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    return consolidated


def run_primary_observability_for_run(results_dir: Path, run_name: str) -> dict[str, Any]:
    """Fail the producing scientific action unless its new transition stream is observable."""
    if run_name not in RUN_SPECS:
        raise ValueError(f"Unknown primary observability run: {run_name}")
    paths = RUN_SPECS[run_name]
    report_path = results_dir / paths["report"]
    events_path = results_dir / paths["events"]
    if not report_path.exists() or not events_path.exists():
        raise FileNotFoundError(f"Required observability inputs missing for {run_name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = _read_jsonl(events_path)
    primary = _audit_one_run(
        run_name,
        report,
        events,
        results_dir / "observability_primary" / run_name,
    )
    return {
        "primary_endpoint": primary["primary_endpoint"],
        "events": primary["transition_validation"]["events"],
        "strict_transition_validity": primary["transition_validation"]["strict_validity"],
        "msoi_multidimensional_profile": primary["msoi_multidimensional_profile"],
        "rmoi_multidimensional_profile": primary["rmoi_multidimensional_profile"],
        "saoi_multidimensional_profile": primary["saoi_multidimensional_profile"],
        "report": primary["artifacts"]["report"],
    }
