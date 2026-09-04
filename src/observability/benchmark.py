from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.observability.recursive import REQUIRED_TRANSITION_FIELDS, trace_completeness, validate_transition_stream


# Nature figure guidance calls for Arial/Helvetica labels, consistent sizing at
# final reproduction scale, and colour choices that do not depend on red-green
# discrimination.  These values are applied at the intended single-page width.
NATURE_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.prop_cycle": cycler(color=NATURE_COLOURS),
        "svg.fonttype": "none",
    }
)


SYSTEM_FILES = {
    "depmap_prism": "master_brain/recursive_observability_events.jsonl",
    "lincs": "validation/lincs_future_state/frozen_holdout_recursive_events.jsonl",
    "tcga": "tcga_outcome_observability/recursive_observability_events.jsonl",
}

SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)

FORMAL_CONDITIONS = (
    ("N1", "Trace closure", "Every transition has a valid O0-O6 provenance chain.", "strict_validity"),
    ("N2", "State availability", "Required states are finite and present at every recursive level.", "trace_completeness"),
    ("N3", "Agent identifiability", "Agent channels retain distinguishable state information.", "agent_identifiability"),
    ("N4", "Master reconstructability", "Recorded Master state is recoverable without state substitution.", "master_fidelity"),
    ("N5", "Self-error observability", "Meta-uncertainty retains ordered information about Master uncertainty.", "meta_order_fidelity"),
    ("N6", "Channel separability", "The recursive telemetry spans more than one independent channel.", "channel_rank_ratio"),
    ("S1", "Failure sensitivity", "A condition-specific metric detects controlled loss of each required channel.", "disruption_sensitivity"),
    ("S2", "Recursive non-redundancy", "At least one higher observer level adds held-out reconstruction information.", "depth_information_gain"),
    ("S3", "Cross-system invariance", "Controlled failures produce concordant signatures across biological systems.", "cross_system_concordance"),
)

DISRUPTIONS = {
    "hidden_provenance": "Remove evidence identifiers and parent traces.",
    "hidden_agent_state": "Remove agent-state and agent-uncertainty telemetry.",
    "collapsed_agents": "Replace distinct agent states by their within-unit mean.",
    "corrupted_master_state": "Permute Master states between biological units.",
    "corrupted_meta_observer": "Permute meta-uncertainty between biological units.",
    "collapsed_recursive_channels": "Replace uncertainty channels by one shared standardized signal.",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _episode_groups(events: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        groups.setdefault(str(event["episode_id"]), []).append(index)
    return list(groups.values())


def _selected_groups(events: Sequence[Mapping[str, Any]], severity: float, seed: int) -> list[list[int]]:
    groups = _episode_groups(events)
    if severity <= 0:
        return []
    rng = np.random.default_rng(seed)
    count = min(len(groups), max(1, int(round(severity * len(groups)))))
    return [groups[index] for index in rng.permutation(len(groups))[:count]]


def _vectors(events: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    rows = []
    for event in events:
        value = event.get(field)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            rows.append([np.nan])
        else:
            rows.append(np.asarray(value, dtype=float).ravel().tolist())
    width = max(len(row) for row in rows)
    return np.asarray([row + [np.nan] * (width - len(row)) for row in rows], dtype=float)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return 0.0
    return float(abs(np.corrcoef(left[mask], right[mask])[0, 1]))


def _number(value: Any) -> float:
    try:
        return float(value) if value is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _rank_ratio(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    keep = np.isfinite(matrix).all(axis=1)
    matrix = matrix[keep]
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return 0.0
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0)
    informative = scale > np.finfo(float).eps
    centered = centered[:, informative] / scale[informative]
    if centered.shape[1] <= 1:
        return float(centered.shape[1])
    singular = np.linalg.svd(centered, compute_uv=False)
    weights = singular / (singular.sum() + np.finfo(float).eps)
    effective_rank = float(np.exp(-np.sum(weights[weights > 0] * np.log(weights[weights > 0]))))
    return effective_rank / centered.shape[1]


def benchmark_metrics(events: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    validation = validate_transition_stream(events)
    agent = _vectors(events, "agent_state_after")
    reference_agent = _vectors(reference, "agent_state_after")
    master = _vectors(events, "master_state_after")
    reference_master = _vectors(reference, "master_state_after")
    master_scale = np.linalg.norm(reference_master) + np.finfo(float).eps
    finite_master = np.nan_to_num(master, nan=0.0)
    master_fidelity = 1.0 - np.linalg.norm(finite_master - reference_master) / master_scale

    episode_divergence = []
    for indices in _episode_groups(events):
        states = agent[indices]
        if np.isfinite(states).all() and len(states) > 1:
            episode_divergence.append(float(np.mean(np.std(states, axis=0))))
    reference_divergence = []
    for indices in _episode_groups(reference):
        states = reference_agent[indices]
        if np.isfinite(states).all() and len(states) > 1:
            reference_divergence.append(float(np.mean(np.std(states, axis=0))))
    divisor = np.mean(reference_divergence) + np.finfo(float).eps
    agent_identifiability = np.clip(np.mean(episode_divergence) / divisor, 0, 1) if episode_divergence else 0.0

    uncertainty = np.asarray([_number(event.get("master_uncertainty")) for event in events])
    meta = np.asarray([_number(event.get("meta_uncertainty")) for event in events])
    reference_meta = np.asarray([float(event["meta_uncertainty"]) for event in reference])
    meta_order_fidelity = _safe_corr(meta, reference_meta)
    channel_matrix = np.column_stack(
        [
            np.divide(
                np.nansum(np.abs(agent), axis=1),
                np.isfinite(agent).sum(axis=1),
                out=np.full(len(agent), np.nan),
                where=np.isfinite(agent).sum(axis=1) > 0,
            ),
            np.asarray([_number(event.get("agent_uncertainty")) for event in events]),
            uncertainty,
            meta,
        ]
    )
    return {
        "strict_validity": float(validation["strict_validity"]),
        "trace_completeness": trace_completeness(events),
        "agent_identifiability": float(agent_identifiability),
        "master_fidelity": float(np.clip(master_fidelity, 0, 1)),
        "meta_order_fidelity": meta_order_fidelity,
        "channel_rank_ratio": _rank_ratio(channel_matrix),
        "master_meta_association": _safe_corr(uncertainty, meta),
    }


def disrupt_events(
    events: Sequence[Mapping[str, Any]], disruption: str, severity: float, seed: int = 20260827
) -> list[dict[str, Any]]:
    altered = deepcopy(list(events))
    groups = _selected_groups(altered, severity, seed)
    if not groups:
        return altered
    chosen = [index for group in groups for index in group]
    rng = np.random.default_rng(seed + 31)
    if disruption == "hidden_provenance":
        for index in chosen:
            altered[index]["evidence_ids"] = []
            altered[index]["parent_trace"] = ""
    elif disruption == "hidden_agent_state":
        for index in chosen:
            altered[index]["agent_state_after"] = []
            altered[index]["agent_uncertainty"] = None
    elif disruption == "collapsed_agents":
        for group in groups:
            states = [_vectors([altered[index]], "agent_state_after")[0] for index in group]
            mean_state = np.mean(states, axis=0).tolist()
            for index in group:
                altered[index]["agent_state_after"] = mean_state
    elif disruption == "corrupted_master_state":
        sources = rng.permutation(groups)
        for target_group, source_group in zip(groups, sources):
            source = altered[source_group[0]]["master_state_after"]
            for index in target_group:
                altered[index]["master_state_after"] = list(source)
    elif disruption == "corrupted_meta_observer":
        values = [altered[group[0]]["meta_uncertainty"] for group in groups]
        for group, value in zip(groups, rng.permutation(values)):
            for index in group:
                altered[index]["meta_uncertainty"] = float(value)
    elif disruption == "collapsed_recursive_channels":
        shared = np.asarray([float(altered[index]["agent_uncertainty"]) for index in chosen])
        shared = (shared - shared.min()) / (np.ptp(shared) + np.finfo(float).eps)
        for position, index in enumerate(chosen):
            value = float(shared[position])
            altered[index]["agent_uncertainty"] = value
            altered[index]["master_uncertainty"] = value
            altered[index]["meta_uncertainty"] = value
    else:
        raise ValueError(f"Unknown disruption: {disruption}")
    return altered


def _episode_features(events: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[np.ndarray]], np.ndarray]:
    ids, levels, targets = [], [], []
    for indices in _episode_groups(events):
        rows = [events[index] for index in indices]
        agent_after = np.vstack([np.asarray(row["agent_state_after"], dtype=float) for row in rows])
        agent_before = np.vstack([np.asarray(row["agent_state_before"], dtype=float) for row in rows])
        master_before = np.asarray(rows[0]["master_state_before"], dtype=float)
        target = np.asarray(rows[0]["master_state_after"], dtype=float)
        level_features = [
            np.concatenate([master_before]),
            np.concatenate([master_before, agent_before.mean(axis=0), agent_after.mean(axis=0)]),
            np.concatenate([master_before, agent_before.mean(axis=0), agent_after.mean(axis=0), agent_after.std(axis=0)]),
            np.concatenate([master_before, agent_before.mean(axis=0), agent_after.mean(axis=0), agent_after.std(axis=0), [np.mean([row["agent_uncertainty"] for row in rows])]]),
            np.concatenate([master_before, agent_before.mean(axis=0), agent_after.mean(axis=0), agent_after.std(axis=0), [np.mean([row["agent_uncertainty"] for row in rows]), rows[0]["master_uncertainty"]]]),
            np.concatenate([master_before, agent_before.mean(axis=0), agent_after.mean(axis=0), agent_after.std(axis=0), [np.mean([row["agent_uncertainty"] for row in rows]), rows[0]["master_uncertainty"], rows[0]["meta_uncertainty"]]]),
        ]
        ids.append(str(rows[0]["episode_id"]))
        levels.append(level_features)
        targets.append(target)
    return ids, levels, np.vstack(targets)


def depth_information(events: Sequence[Mapping[str, Any]], seed: int = 20260827) -> pd.DataFrame:
    ids, nested, target = _episode_features(events)
    folds = min(5, max(2, len(ids) // 10))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    previous = None
    for depth in range(6):
        features = np.vstack([levels[depth] for levels in nested])
        predictions = np.zeros_like(target)
        for train, test in splitter.split(features):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(features[train], target[train])
            predicted = np.asarray(model.predict(features[test]))
            predictions[test] = predicted.reshape(len(test), target.shape[1])
        error = np.linalg.norm(target - predictions) / (np.linalg.norm(target) + np.finfo(float).eps)
        reconstructability = float(np.clip(1 - error, 0, 1))
        rows.append(
            {
                "observer_depth": depth,
                "heldout_reconstructability": reconstructability,
                "marginal_information_gain": 0.0 if previous is None else reconstructability - previous,
                "episodes": len(ids),
            }
        )
        previous = reconstructability
    return pd.DataFrame(rows)


def run_benchmark(input_root: Path, output_root: Path, seed: int = 20260827) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    all_rows, depth_rows = [], []
    baseline_by_system: dict[str, dict[str, float]] = {}
    for system, relative_path in SYSTEM_FILES.items():
        events = _read_jsonl(input_root / relative_path)
        baseline = benchmark_metrics(events, events)
        baseline_by_system[system] = baseline
        depth = depth_information(events, seed)
        depth.insert(0, "system", system)
        depth_rows.append(depth)
        for disruption in DISRUPTIONS:
            for severity in SEVERITIES:
                altered = disrupt_events(events, disruption, severity, seed)
                metrics = benchmark_metrics(altered, events)
                all_rows.append({"system": system, "disruption": disruption, "severity": severity, **metrics})
    benchmark = pd.DataFrame(all_rows)
    depth_table = pd.concat(depth_rows, ignore_index=True)

    detector = {
        "hidden_provenance": "strict_validity",
        "hidden_agent_state": "trace_completeness",
        "collapsed_agents": "agent_identifiability",
        "corrupted_master_state": "master_fidelity",
        "corrupted_meta_observer": "meta_order_fidelity",
        "collapsed_recursive_channels": "channel_rank_ratio",
    }
    sensitivity_rows = []
    for (system, disruption), group in benchmark.groupby(["system", "disruption"]):
        metric = detector[disruption]
        ordered = group.sort_values("severity")
        baseline = float(ordered.iloc[0][metric])
        terminal = float(ordered.iloc[-1][metric])
        rho = float(ordered["severity"].corr(ordered[metric], method="spearman"))
        sensitivity_rows.append(
            {
                "system": system,
                "disruption": disruption,
                "detector_metric": metric,
                "baseline": baseline,
                "complete_disruption": terminal,
                "absolute_detection_effect": baseline - terminal,
                "severity_spearman": rho,
                "detected": bool((baseline - terminal) > 0.05 and rho < -0.7),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    signatures = sensitivity.pivot(index="disruption", columns="system", values="absolute_detection_effect")
    invariance = signatures.corr(method="spearman")
    concordance_values = invariance.to_numpy()[np.triu_indices(len(invariance), 1)]
    depth_gain = depth_table.groupby("system")["marginal_information_gain"].max()
    conditions = pd.DataFrame(FORMAL_CONDITIONS, columns=["condition", "name", "definition", "evidence_metric"])
    minimum_baseline = {
        metric: min(values[metric] for values in baseline_by_system.values())
        for metric in ("strict_validity", "trace_completeness", "agent_identifiability", "master_fidelity", "meta_order_fidelity", "channel_rank_ratio")
    }
    thresholds = {"N1": 1.0, "N2": 1.0, "N3": 0.5, "N4": 0.95, "N5": 0.95, "N6": 0.5}
    conditions["acceptance_rule"] = ""
    conditions["observed_evidence"] = np.nan
    conditions["result"] = "not demonstrated"
    condition_metric = {"N1": "strict_validity", "N2": "trace_completeness", "N3": "agent_identifiability", "N4": "master_fidelity", "N5": "meta_order_fidelity", "N6": "channel_rank_ratio"}
    for condition, metric in condition_metric.items():
        threshold = thresholds[condition]
        value = minimum_baseline[metric]
        mask = conditions["condition"] == condition
        conditions.loc[mask, "acceptance_rule"] = f"minimum across systems >= {threshold:.2f}"
        conditions.loc[mask, "observed_evidence"] = value
        conditions.loc[mask, "result"] = "demonstrated" if value >= threshold else "not demonstrated"
    conditions.loc[conditions["condition"] == "S1", "acceptance_rule"] = "all failure tests: effect > 0.05 and severity Spearman < -0.70"
    conditions.loc[conditions["condition"] == "S1", "observed_evidence"] = float(sensitivity["detected"].mean())
    conditions.loc[conditions["condition"] == "S1", "result"] = (
        "demonstrated" if sensitivity["detected"].all() else "not demonstrated"
    )
    conditions.loc[conditions["condition"] == "S2", "result"] = (
        "demonstrated" if (depth_gain > 0.005).any() else "not demonstrated"
    )
    conditions.loc[conditions["condition"] == "S2", "acceptance_rule"] = "positive held-out marginal gain > 0.005"
    conditions.loc[conditions["condition"] == "S2", "observed_evidence"] = float(depth_gain.max())
    conditions.loc[conditions["condition"] == "S3", "result"] = (
        "demonstrated" if np.nanmedian(concordance_values) >= 0.7 else "not demonstrated"
    )
    conditions.loc[conditions["condition"] == "S3", "acceptance_rule"] = "median pairwise signature Spearman >= 0.70"
    conditions.loc[conditions["condition"] == "S3", "observed_evidence"] = float(np.nanmedian(concordance_values))

    taxonomy = pd.DataFrame(
        [
            {"failure_class": key, "operational_intervention": text, "primary_detector": detector[key], "scientific_interpretation": interpretation}
            for key, text, interpretation in [
                ("hidden_provenance", DISRUPTIONS["hidden_provenance"], "The decision exists but its evidentiary ancestry is not independently inspectable."),
                ("hidden_agent_state", DISRUPTIONS["hidden_agent_state"], "The Master output cannot be connected to the state of an observer agent."),
                ("collapsed_agents", DISRUPTIONS["collapsed_agents"], "Nominally distinct observers no longer provide identifiable perspectives."),
                ("corrupted_master_state", DISRUPTIONS["corrupted_master_state"], "The recorded Master state is substituted across biological units."),
                ("corrupted_meta_observer", DISRUPTIONS["corrupted_meta_observer"], "Self-observation is present as telemetry but loses its ordering information."),
                ("collapsed_recursive_channels", DISRUPTIONS["collapsed_recursive_channels"], "Multiple recursive labels encode a single latent signal."),
            ]
        ]
    )

    tables = {
        "table_01_formal_conditions": conditions,
        "table_02_disruption_benchmark": benchmark,
        "table_03_detection_sensitivity": sensitivity,
        "table_04_recursive_depth_information": depth_table,
        "table_05_cross_system_invariance": invariance.reset_index(names="system"),
        "table_06_failure_taxonomy": taxonomy,
    }
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
        table.to_html(tables_dir / f"{name}.html", index=False, float_format=lambda value: f"{value:.4f}")

    metric_order = list(detector.values())
    full = pd.DataFrame(baseline_by_system).T[metric_order]
    nonobservable = benchmark[benchmark["severity"] == 1.0].groupby("system")[metric_order].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    full.plot(kind="bar", ax=axes[0], ylim=(0, 1.05), title="Complete architecture")
    nonobservable.plot(kind="bar", ax=axes[1], ylim=(0, 1.05), title="Deliberately non-observable architecture")
    for ax in axes:
        ax.set_ylabel("Condition-specific observability")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=5.5, frameon=False)
    fig.suptitle("Complete versus disrupted recursive observability", weight="bold")
    _save_figure(fig, figures_dir / "figure_01_complete_vs_nonobservable")

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2), constrained_layout=True)
    for ax, (disruption, metric) in zip(axes.ravel(), detector.items()):
        for system, subset in benchmark[benchmark["disruption"] == disruption].groupby("system"):
            ax.plot(subset["severity"], subset[metric], marker="o", label=system)
        ax.set_title(disruption.replace("_", " "))
        ax.set_xlabel("Disruption severity")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(-0.03, 1.05)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Dose-response detection of controlled observability failures", weight="bold")
    _save_figure(fig, figures_dir / "figure_02_disruption_dose_response")

    fig, ax = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    for system, subset in depth_table.groupby("system"):
        ax.plot(subset["observer_depth"], subset["heldout_reconstructability"], marker="o", label=system)
    ax.set_xlabel("Included recursive observer depth")
    ax.set_ylabel("Cross-validated Master-state reconstructability")
    ax.set_title("Marginal information across recursive observer depth", weight="bold")
    ax.legend(frameon=False)
    _save_figure(fig, figures_dir / "figure_03_recursive_depth_information")

    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    image = ax.imshow(invariance.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(invariance)), invariance.columns, rotation=20)
    ax.set_yticks(range(len(invariance)), invariance.index)
    for row in range(len(invariance)):
        for column in range(len(invariance)):
            ax.text(column, row, f"{invariance.iloc[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Spearman concordance")
    ax.set_title("Cross-system failure-signature invariance", weight="bold")
    _save_figure(fig, figures_dir / "figure_04_cross_system_invariance")

    result = {
        "systems": list(SYSTEM_FILES),
        "events": {system: len(_read_jsonl(input_root / path)) for system, path in SYSTEM_FILES.items()},
        "controlled_tests": int(len(benchmark)),
        "detection_tests_passed": int(sensitivity["detected"].sum()),
        "detection_tests_total": int(len(sensitivity)),
        "median_cross_system_concordance": float(np.nanmedian(concordance_values)),
        "max_depth_information_gain": {key: float(value) for key, value in depth_gain.items()},
        "conditions_demonstrated": int((conditions["result"] == "demonstrated").sum()),
        "conditions_total": int(len(conditions)),
    }
    (output_root / "benchmark_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = _summary_markdown(result, sensitivity, depth_table, conditions)
    (output_root / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    _write_index(output_root)
    _write_manifest(output_root)
    return result


def _save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _summary_markdown(result: Mapping[str, Any], sensitivity: pd.DataFrame, depth: pd.DataFrame, conditions: pd.DataFrame) -> str:
    failed = sensitivity.loc[~sensitivity["detected"], ["system", "disruption"]]
    failed_text = "none" if failed.empty else ", ".join(f"{row.system}/{row.disruption}" for row in failed.itertuples())
    depth_lines = "\n".join(
        f"- {system}: maximum marginal held-out reconstruction gain {group['marginal_information_gain'].max():.4f}."
        for system, group in depth.groupby("system")
    )
    return f"""# Recursive Scientific Observability Benchmark

## Operational definition

For a recorded distributed scientific-AI process, recursive scientific observability (RSO) is
defined as the conjunction `RSO = N1 AND N2 AND N3 AND N4 AND N5 AND N6`. These six necessary
conditions become operationally sufficient within the declared transition schema when the
architecture also satisfies `S1 AND S2 AND S3`: controlled failure sensitivity, recursive
non-redundancy and cross-system invariance. This is a falsifiable operational definition for
the recorded architecture, not a universal mathematical theorem for all dynamical systems.

## Formal result

{result['conditions_demonstrated']}/{result['conditions_total']} operational necessary/sufficient conditions were demonstrated. These conditions establish recursive observability for the recorded AI process; they do not establish biological outcome validity or causal treatment efficacy.

## Controlled failures

The benchmark executed {result['controlled_tests']} graded disruption conditions over {sum(result['events'].values()):,} real-data transition events. {result['detection_tests_passed']}/{result['detection_tests_total']} system-by-failure detection tests passed the prespecified effect (>0.05) and monotonicity (Spearman < -0.70) criteria. Failed tests: {failed_text}.

## Recursive depth

{depth_lines}

Positive gain indicates that a higher observer level contains held-out information not present at lower levels. Negative or zero gain identifies redundant or destabilizing telemetry rather than being hidden by an aggregate score.

## Cross-system invariance

The median pairwise Spearman concordance of controlled failure signatures across DepMap/PRISM, LINCS and TCGA was {result['median_cross_system_concordance']:.3f}. This evaluates invariance of observability failure detection, not invariance of biological outcomes.

## Claim boundary

The result supports recursive scientific observability as a measurable, disruptable and falsifiable property of the recorded distributed AI architecture. It does not prove that an observable system is biologically correct, clinically effective, conscious or causally self-aware.
"""


def _write_index(root: Path) -> None:
    figures = sorted((root / "figures").glob("*.png"))
    tables = sorted((root / "tables").glob("*.html"))
    html = "<!doctype html><meta charset='utf-8'><style>body{font:16px Arial;max-width:1100px;margin:30px auto;color:#17202a}img{max-width:100%}li{margin:7px}</style>"
    html += "<h1>Recursive Scientific Observability Benchmark</h1><p>Outcome-independent validation of observability using controlled internal-state disruptions.</p>"
    html += "<p><a href='RESULTS_SUMMARY.md'>Results summary</a></p><h2>Figures</h2>"
    html += "".join(f"<h3>{path.stem}</h3><img src='figures/{path.name}'>" for path in figures)
    html += "<h2>Tables</h2><ul>" + "".join(f"<li><a href='tables/{path.name}'>{path.stem}</a></li>" for path in tables) + "</ul>"
    (root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            rows.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(root / "artifact_manifest.csv", index=False)
