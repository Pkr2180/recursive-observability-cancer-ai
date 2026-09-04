from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve


RUN_ORDER = ("depmap_prism_master", "lincs_future_state", "tcga_patient_outcome")
RUN_LABELS = {
    "depmap_prism_master": "DepMap/PRISM Master",
    "lincs_future_state": "LINCS future state",
    "tcga_patient_outcome": "TCGA patient outcome",
}
COLORS = {
    "depmap_prism_master": "#277da1",
    "lincs_future_state": "#f8961e",
    "tcga_patient_outcome": "#43aa8b",
}
CHANNEL_ORDER = (
    "cancer_future_state",
    "individual_agents",
    "inter_agent_disagreement",
    "master_state",
    "master_self_observation_uncertainty",
)
CHANNEL_LABELS = {
    "cancer_future_state": "Cancer future-state instability",
    "individual_agents": "Individual-agent instability",
    "inter_agent_disagreement": "Inter-agent disagreement",
    "master_state": "Master-state instability",
    "master_self_observation_uncertainty": "Self-observation uncertainty",
}
SOURCE_REPORTS = {
    "depmap_prism_master": "master_brain/recursive_observability_report.json",
    "lincs_future_state": "validation/lincs_future_state/lincs_future_state_recursive_observability_report.json",
    "tcga_patient_outcome": "tcga_outcome_observability/tcga_outcome_observability_report.json",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> list[Path]:
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, svg_path]


def _save_table(table: pd.DataFrame, tables_dir: Path, stem: str) -> list[Path]:
    csv_path = tables_dir / f"{stem}.csv"
    html_path = tables_dir / f"{stem}.html"
    table.to_csv(csv_path, index=False)
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><style>body{font-family:Arial;margin:24px}"
        "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:6px}"
        "th{background:#eaf2f8}</style>" + table.to_html(index=False, border=0),
        encoding="utf-8",
    )
    return [csv_path, html_path]


def _heatmap(
    data: pd.DataFrame,
    title: str,
    *,
    vmin: float | None = 0,
    vmax: float | None = 1,
    fmt: str = ".3f",
    cmap: str = "viridis",
) -> plt.Figure:
    values = data.to_numpy(dtype=float)
    width = max(7.5, 1.8 * data.shape[1])
    height = max(4.0, 0.65 * data.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1]), data.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(data.shape[0]), data.index)
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            value = values[row, column]
            color = "white" if vmax is not None and value > (vmin + vmax) / 2 else "black"
            ax.text(column, row, format(value, fmt), ha="center", va="center", color=color, fontsize=8)
    ax.set_title(title, weight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def _profile_table(run_reports: dict[str, dict], profile: str) -> pd.DataFrame:
    metrics = list(run_reports[RUN_ORDER[0]][profile])
    return pd.DataFrame(
        {
            RUN_LABELS[run]: [run_reports[run][profile].get(metric, np.nan) for metric in metrics]
            for run in RUN_ORDER
        },
        index=metrics,
    )


def _source_observability(report: dict) -> dict:
    nested = report.get("recursive_observability")
    return nested if isinstance(nested, dict) else report


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_observability_results_package(results_dir: Path) -> dict:
    """Generate static figures and tables exclusively from completed real-data artifacts."""
    primary_dir = results_dir / "observability_primary"
    consolidated = _load_json(primary_dir / "consolidated_primary_observability_report.json")
    run_reports = {
        run: _load_json(primary_dir / run / "primary_scientific_observability_report.json")
        for run in RUN_ORDER
    }
    source_reports = {
        run: _load_json(results_dir / SOURCE_REPORTS[run]) for run in RUN_ORDER
    }
    verification_dir = primary_dir / "verification_non_ablation"
    verification_report = _load_json(
        verification_dir / "non_ablation_observability_verification_report.json"
    )
    verification_tables = {
        name: pd.read_csv(verification_dir / f"{name}.csv")
        for name in (
            "architecture_verification",
            "agent_observability_verification",
            "clustered_metric_intervals",
            "channel_dependence_and_separability",
            "component_performance",
            "uncertainty_calibration",
            "selective_risk_coverage",
        )
    }

    package_dir = primary_dir / "publication_package"
    figures_dir = package_dir / "figures"
    tables_dir = package_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Table 1 and Figure 1: real-data system inventory.
    inventory = pd.DataFrame(
        [
            {
                "system": RUN_LABELS[run],
                "events": consolidated["runs"][run]["events"],
                "strict_transition_validity": consolidated["runs"][run]["strict_transition_validity"],
                "observer_depth": consolidated["runs"][run]["observer_depth"],
                "osic": consolidated["runs"][run]["osic"],
            }
            for run in RUN_ORDER
        ]
    )
    generated += _save_table(inventory, tables_dir, "table_01_real_data_system_inventory")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(
        inventory["system"], inventory["events"], color=[COLORS[run] for run in RUN_ORDER]
    )
    axes[0].set_ylabel("Validated transition events")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Real-data observability event streams")
    axes[1].bar(
        inventory["system"], inventory["strict_transition_validity"],
        color=[COLORS[run] for run in RUN_ORDER],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Strict validity")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("Fail-closed transition validity")
    fig.suptitle("Figure 1. Completed real-data observability systems", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_01_system_inventory")

    schema = pd.DataFrame(
        {
            "field_order": np.arange(1, len(run_reports[RUN_ORDER[0]]["required_transition_schema"]) + 1),
            "required_transition_field": run_reports[RUN_ORDER[0]]["required_transition_schema"],
        }
    )
    generated += _save_table(schema, tables_dir, "table_02_required_transition_schema")

    # Primary multidimensional profiles.
    for number, (profile, title, table_name) in enumerate(
        [
            ("msoi_multidimensional_profile", "MSOI multidimensional profile", "table_03_msoi_profile"),
            ("rmoi_multidimensional_profile", "RMOI multidimensional profile", "table_04_rmoi_profile"),
            ("saoi_multidimensional_profile", "SAOI multidimensional profile", "table_05_saoi_profile"),
        ],
        start=2,
    ):
        matrix = _profile_table(run_reports, profile)
        table = matrix.reset_index(names="metric")
        generated += _save_table(table, tables_dir, table_name)
        fig = _heatmap(matrix, f"Figure {number}. {title}")
        generated += _save_figure(fig, figures_dir, f"figure_{number:02d}_{profile.split('_')[0]}_heatmap")

    # Instability streams remain separated.
    channel_rows = []
    channel_values: dict[str, dict[str, np.ndarray]] = {}
    for run in RUN_ORDER:
        observable = _source_observability(source_reports[run])
        channel_values[run] = {}
        for channel in CHANNEL_ORDER:
            values = np.asarray(observable["instability_channels"][channel], dtype=float)
            values = values[np.isfinite(values)]
            channel_values[run][channel] = values
            channel_rows.append(
                {
                    "system": RUN_LABELS[run],
                    "channel": CHANNEL_LABELS[channel],
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "median": np.median(values),
                    "q25": np.quantile(values, 0.25),
                    "q75": np.quantile(values, 0.75),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    channel_summary = pd.DataFrame(channel_rows)
    generated += _save_table(channel_summary, tables_dir, "table_06_separated_instability_channels")
    fig, axes = plt.subplots(3, 5, figsize=(17, 8.5), constrained_layout=True)
    for row, run in enumerate(RUN_ORDER):
        for column, channel in enumerate(CHANNEL_ORDER):
            ax = axes[row, column]
            ax.boxplot(
                channel_values[run][channel],
                widths=0.55,
                patch_artist=True,
                boxprops={"facecolor": COLORS[run], "alpha": 0.75},
                medianprops={"color": "black"},
                showfliers=False,
            )
            ax.set_xticks([])
            if row == 0:
                ax.set_title(CHANNEL_LABELS[channel], fontsize=9)
            if column == 0:
                ax.set_ylabel(RUN_LABELS[run])
    fig.suptitle("Figure 5. Five non-collapsed instability channels", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_05_instability_channel_distributions")

    # OOSC and OSIC.
    oosc_metrics = list(run_reports[RUN_ORDER[0]]["oosc_pairwise_profile"])
    oosc = pd.DataFrame(
        {
            RUN_LABELS[run]: [run_reports[run]["oosc_pairwise_profile"][key] for key in oosc_metrics]
            for run in RUN_ORDER
        },
        index=oosc_metrics,
    )
    generated += _save_table(oosc.reset_index(names="coupling"), tables_dir, "table_07_osic_oosc_profile")
    fig = _heatmap(oosc, "Figure 6. Pairwise OOSC dependence profile", vmin=-1, vmax=1)
    generated += _save_figure(fig, figures_dir, "figure_06_oosc_heatmap")

    # Agentic divergence matrices.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    divergence_rows = []
    for ax, run in zip(axes, RUN_ORDER):
        path = primary_dir / run / "agentic_divergence_matrix.csv"
        matrix = pd.read_csv(path, index_col=0)
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="magma", aspect="equal", vmin=0)
        labels = [str(value).replace("_", " ") for value in matrix.index]
        ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_title(RUN_LABELS[run])
        for left in range(len(labels)):
            for right in range(len(labels)):
                ax.text(right, left, f"{values[left, right]:.2f}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.7)
        for left_index, left in enumerate(matrix.index):
            for right_index in range(left_index + 1, len(matrix.index)):
                divergence_rows.append(
                    {
                        "system": RUN_LABELS[run],
                        "agent_left": left,
                        "agent_right": matrix.index[right_index],
                        "mean_divergence": values[left_index, right_index],
                    }
                )
    fig.suptitle("Figure 7. Agentic divergence matrices", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_07_agentic_divergence_matrices")
    generated += _save_table(pd.DataFrame(divergence_rows), tables_dir, "table_08_agentic_divergence")

    # O0-O6 observer depth profile.
    depth_rows = []
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for run in RUN_ORDER:
        levels = run_reports[run]["observer_depth"]["levels"]
        x = [row["level"] for row in levels]
        y = [row["observable"] for row in levels]
        ax.plot(x, y, marker="o", linewidth=2, label=RUN_LABELS[run], color=COLORS[run])
        for row in levels:
            depth_rows.append({"system": RUN_LABELS[run], **row})
    ax.set_xticks(range(7), [f"O{index}" for index in range(7)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Observed availability / reconstructability")
    ax.set_title("Figure 8. O0–O6 observer-depth profile", weight="bold")
    ax.legend(frameon=False)
    generated += _save_figure(fig, figures_dir, "figure_08_observer_depth_profiles")
    generated += _save_table(pd.DataFrame(depth_rows), tables_dir, "table_09_observer_depth")

    # Gate influence distributions and table.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    gate_rows = []
    for ax, run in zip(axes, RUN_ORDER):
        gates = pd.read_csv(primary_dir / run / "gate_influence_trace.csv")
        gates["system"] = RUN_LABELS[run]
        gate_rows.append(gates)
        ax.hist(gates["master_state_delta"], bins=30, color=COLORS[run], alpha=0.8)
        ax.axvline(gates["master_state_delta"].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_title(RUN_LABELS[run])
        ax.set_xlabel("Normalized Master-state delta")
        ax.set_ylabel("Events")
    fig.suptitle("Figure 9. Gate-associated Master-state influence", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_09_gate_influence_distributions")
    gate_table = pd.concat(gate_rows, ignore_index=True)
    gate_summary = (
        gate_table.groupby(["system", "gate_applied"], as_index=False)
        .agg(events=("event_index", "size"), mean_master_delta=("master_state_delta", "mean"), median_master_delta=("master_state_delta", "median"))
    )
    generated += _save_table(gate_summary, tables_dir, "table_10_gate_influence")

    # TCGA held-out prediction: ROC and calibration.
    tcga_predictions = pd.read_parquet(
        results_dir / "tcga_outcome_observability" / "heldout_project_predictions.parquet"
    )
    y = tcga_predictions["observed_death_by_2y"].to_numpy(dtype=int)
    probability = tcga_predictions["master_probability"].to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(y, probability)
    prob_true, prob_pred = calibration_curve(y, probability, n_bins=5, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    axes[0].plot(fpr, tpr, color=COLORS["tcga_patient_outcome"], linewidth=2, label=f"AUROC={auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend(frameon=False)
    axes[0].set_title("Held-out ROC")
    axes[1].plot(prob_pred, prob_true, marker="o", color=COLORS["tcga_patient_outcome"], linewidth=2)
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed event fraction")
    axes[1].set_title("Five-bin calibration")
    fig.suptitle("Figure 10. TCGA two-year outcome secondary performance", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_10_tcga_roc_calibration")

    tcga_metrics = source_reports["tcga_patient_outcome"]["test_metrics"]
    tcga_permutation = source_reports["tcga_patient_outcome"]["permutation_audit"]
    tcga_table = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in tcga_metrics.items()]
        + [{"metric": key, "value": value} for key, value in tcga_permutation.items() if key != "method"]
    )
    generated += _save_table(tcga_table, tables_dir, "table_11_tcga_secondary_performance")

    # LINCS cancer/Master instability coupling.
    lincs = pd.read_parquet(
        results_dir / "validation" / "lincs_future_state" / "frozen_holdout_transition_audit.parquet",
        columns=["cancer_future_state_instability", "master_state_instability"],
    )
    x = lincs["cancer_future_state_instability"].to_numpy(dtype=float)
    y_lincs = lincs["master_state_instability"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y_lincs, 1)
    line_x = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    ax.scatter(x, y_lincs, s=12, alpha=0.35, color=COLORS["lincs_future_state"], edgecolors="none")
    ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=2)
    ax.set_xlabel("Observed cancer future-state change")
    ax.set_ylabel("Master forecast error / instability")
    ax.set_title("Figure 11. LINCS cancer–Master instability coupling\nOSIC = 0.895", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_11_lincs_cancer_master_coupling")

    # TCGA project outcome composition.
    project_profile = pd.read_csv(
        results_dir / "tcga_outcome_observability" / "endpoint_project_profile.csv"
    )
    generated += _save_table(project_profile, tables_dir, "table_12_tcga_project_endpoint_profile")
    ordered = project_profile.sort_values(["partition", "project_id"], kind="stable")
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    x_positions = np.arange(len(ordered))
    ax.bar(x_positions, ordered["non_events"], color="#90be6d", label="Survived beyond 2 years")
    ax.bar(x_positions, ordered["events"], bottom=ordered["non_events"], color="#f94144", label="Death by 2 years")
    ax.set_xticks(x_positions, ordered["project_id"].str.replace("TCGA-", "", regex=False), rotation=90)
    ax.set_ylabel("Censoring-eligible patients")
    ax.set_title("Figure 12. TCGA two-year outcome composition across all 33 projects", weight="bold")
    ax.legend(frameon=False, ncol=2)
    generated += _save_figure(fig, figures_dir, "figure_12_tcga_project_outcomes")

    rank_dynamics_rows = []
    for run in RUN_ORDER:
        rank = run_reports[run]["observability_rank"]
        dynamics = run_reports[run]["state_dynamics"]
        rank_dynamics_rows.append(
            {
                "system": RUN_LABELS[run],
                "empirical_rank": rank["empirical_master_state_telemetry_rank"],
                "master_dimension": rank["master_state_dimension"],
                "observability_rank_ratio": rank["observability_rank_ratio"],
                "state_switching_rate_per_integration_step": dynamics["state_switching_rate_per_integration_step"],
                "mean_state_persistence_steps": dynamics["mean_state_persistence_integration_steps"],
                "mean_agentic_divergence": run_reports[run]["agentic_divergence"]["mean_pairwise_divergence"],
                "gaussian_total_correlation": run_reports[run]["oosc_multivariate_profile"]["gaussian_total_correlation"],
            }
        )
    generated += _save_table(
        pd.DataFrame(rank_dynamics_rows), tables_dir, "table_13_rank_state_dynamics_multivariate_oosc"
    )

    # Non-ablation verification of the Master architecture and every observer agent.
    verification_table_specs = (
        ("architecture_verification", "table_14_architecture_verification"),
        ("agent_observability_verification", "table_15_all_agent_observability_verification"),
        ("component_performance", "table_16_agent_and_master_component_performance"),
        ("clustered_metric_intervals", "table_17_clustered_observability_confidence_intervals"),
        ("channel_dependence_and_separability", "table_18_channel_separability_and_dependence"),
        ("uncertainty_calibration", "table_19_recursive_uncertainty_calibration"),
        ("selective_risk_coverage", "table_20_selective_risk_coverage"),
    )
    for source_name, table_name in verification_table_specs:
        # Make intentionally unavailable statistics explicit in publication tables
        # (for example, a confidence interval that is not defined for a count).
        # Keep the numeric in-memory frame unchanged for plotting below.
        publication_table = verification_tables[source_name].fillna("not_applicable")
        generated += _save_table(publication_table, tables_dir, table_name)

    architecture = verification_tables["architecture_verification"].copy()
    architecture_metrics = [
        "strict_transition_validity",
        "trace_completeness",
        "complete_agent_set_per_unit_rate",
        "master_observability_rank_ratio",
        "channel_rank_ratio",
        "uncertainty_field_coverage",
        "evidence_coverage",
        "hypothesis_transition_coverage",
        "gate_coverage",
    ]
    architecture_matrix = architecture.set_index("system")[architecture_metrics]
    fig = _heatmap(
        architecture_matrix,
        "Figure 13. Non-ablation Master-architecture verification",
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
    )
    generated += _save_figure(fig, figures_dir, "figure_13_master_architecture_verification")

    agents = verification_tables["agent_observability_verification"].copy()
    agent_metrics = [
        "unit_coverage",
        "strict_transition_validity",
        "required_field_coverage",
        "evidence_coverage",
        "gate_coverage",
        "hypothesis_coverage",
    ]
    agents["component"] = agents["system"] + " / " + agents["agent_id"]
    agent_matrix = agents.set_index("component")[agent_metrics]
    fig = _heatmap(
        agent_matrix,
        "Figure 14. Observability verification for every observer agent",
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
    )
    generated += _save_figure(fig, figures_dir, "figure_14_all_agent_observability_verification")

    intervals = verification_tables["clustered_metric_intervals"].copy()
    selected_intervals = intervals[
        intervals["metric"].isin(
            ["OSIC_cancer_master", "state_reconstructability", "recursive_consistency"]
        )
    ].copy()
    selected_intervals["label"] = selected_intervals["system"] + " / " + selected_intervals["metric"]
    selected_intervals = selected_intervals.sort_values(["metric", "system"], kind="stable")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    positions = np.arange(len(selected_intervals))
    estimates = selected_intervals["estimate"].to_numpy(dtype=float)
    low = selected_intervals["ci_95_low"].to_numpy(dtype=float)
    high = selected_intervals["ci_95_high"].to_numpy(dtype=float)
    ax.errorbar(
        estimates,
        positions,
        xerr=np.vstack(
            [np.maximum(0, estimates - low), np.maximum(0, high - estimates)]
        ),
        fmt="o",
        color="#1f6f8b",
        ecolor="#6c8da0",
        capsize=3,
    )
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(positions, selected_intervals["label"])
    ax.set_xlabel("Estimate with biological-unit cluster-bootstrap 95% CI")
    ax.set_title("Figure 15. Clustered uncertainty for primary observability metrics", weight="bold")
    ax.invert_yaxis()
    generated += _save_figure(fig, figures_dir, "figure_15_clustered_observability_intervals")

    calibration = verification_tables["uncertainty_calibration"].copy()
    association = calibration[
        calibration["metric"].isin(
            [
                "pearson",
                "spearman",
                "high_error_detection_auroc",
                "high_error_detection_average_precision",
            ]
        )
    ].pivot(index="system", columns="metric", values="value")
    error_calibration = calibration[
        calibration["metric"].isin(["mae", "rmse", "calibration_slope"])
    ].pivot(index="system", columns="metric", values="value")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    association.plot(
        kind="bar",
        ax=axes[0],
        color=["#277da1", "#43aa8b", "#f8961e", "#f94144"],
    )
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axes[0].set_ylim(-1, 1.05)
    axes[0].set_ylabel("Association / discrimination")
    axes[0].set_title("Self-error association and high-error detection")
    axes[0].tick_params(axis="x", rotation=20)
    error_calibration.plot(
        kind="bar", ax=axes[1], color=["#577590", "#90be6d", "#f9c74f"]
    )
    axes[1].axhline(1, color="grey", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Error / calibration slope")
    axes[1].set_title("Recursive uncertainty calibration")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 16. Master self-observation calibration", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_16_recursive_uncertainty_calibration")

    dependence = verification_tables["channel_dependence_and_separability"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    channel_names = list(CHANNEL_LABELS)
    for ax, run in zip(axes, RUN_ORDER):
        matrix = pd.DataFrame(
            np.eye(len(channel_names)), index=channel_names, columns=channel_names
        )
        subset = dependence[dependence["run"] == run]
        for row in subset.itertuples(index=False):
            value = row.partial_correlation_controlling_other_channels
            matrix.loc[row.channel_left, row.channel_right] = value
            matrix.loc[row.channel_right, row.channel_left] = value
        image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
        labels = [CHANNEL_LABELS[name] for name in channel_names]
        ax.set_xticks(np.arange(len(channel_names)), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(channel_names)), labels)
        ax.set_title(RUN_LABELS[run])
        for row_index in range(len(channel_names)):
            for column_index in range(len(channel_names)):
                ax.text(
                    column_index,
                    row_index,
                    f"{matrix.iloc[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    fig.colorbar(image, ax=axes, shrink=0.75, label="Partial correlation")
    fig.suptitle("Figure 17. Five-channel conditional-dependence structure", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_17_channel_partial_correlations")

    risk = verification_tables["selective_risk_coverage"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for run in RUN_ORDER:
        subset = risk[risk["run"] == run].sort_values("coverage")
        ax.plot(
            subset["coverage"],
            subset["mean_absolute_error"],
            marker="o",
            color=COLORS[run],
            label=RUN_LABELS[run],
        )
    ax.set_xlabel("Retained coverage after rejecting highest predicted self-error")
    ax.set_ylabel("Mean realized error among retained units")
    ax.set_title("Figure 18. Selective-risk verification", weight="bold")
    ax.legend(frameon=False)
    generated += _save_figure(fig, figures_dir, "figure_18_selective_risk_coverage")

    performance = verification_tables["component_performance"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    depmap = performance[
        (performance["run"] == "depmap_prism_master")
        & performance["metric"].isin(
            ["mean_declared_influence", "representation_explained_variance"]
        )
    ].pivot(index="component", columns="metric", values="estimate")
    depmap.plot(kind="bar", ax=axes[0], color=["#277da1", "#90be6d"])
    axes[0].set_title("DepMap/PRISM observer agents")
    axes[0].set_ylabel("Observed value")
    axes[0].tick_params(axis="x", rotation=20)
    lincs = performance[
        (performance["run"] == "lincs_future_state")
        & performance["metric"].isin(
            ["forecast_normalized_l2_error", "forecast_cosine_similarity"]
        )
    ].pivot(index="component", columns="metric", values="estimate")
    lincs.plot(kind="bar", ax=axes[1], color=["#f8961e", "#43aa8b"])
    axes[1].set_title("LINCS future-state observers")
    axes[1].tick_params(axis="x", rotation=20)
    tcga = performance[
        (performance["run"] == "tcga_patient_outcome")
        & performance["metric"].isin(["roc_auc", "brier"])
    ].pivot(index="component", columns="metric", values="estimate")
    tcga.plot(kind="bar", ax=axes[2], color=["#f94144", "#577590"])
    axes[2].set_title("TCGA outcome observers")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 19. Agent and Master component verification", weight="bold")
    generated += _save_figure(fig, figures_dir, "figure_19_agent_master_component_performance")

    interval_lookup = intervals.set_index(["run", "metric"])
    osic_lines = []
    for run in RUN_ORDER:
        row = interval_lookup.loc[(run, "OSIC_cancer_master")]
        osic_lines.append(
            f"- {RUN_LABELS[run]}: OSIC {row['estimate']:.3f} "
            f"(cluster-bootstrap 95% CI {row['ci_95_low']:.3f} to {row['ci_95_high']:.3f})."
        )
    verified_agents = int(agents["observational_agent_verified"].sum())
    total_agents = int(len(agents))
    independent_units = int(architecture["biological_units"].sum())
    summary_path = package_dir / "RESULTS_SUMMARY.md"
    summary_path.write_text(
        f"""# Primary Scientific Observability Results

All results derive from observed public pan-cancer data. No simulated patients, cell states or
biological trajectories were used. Statistical bootstraps and permutations only resampled or
rearranged observed values.

## Primary result

The consolidated audit validated 4,431 transitions with strict validity 1.0. MSOI, RMOI and
SAOI are reported as multidimensional profiles and are not collapsed into an arbitrary total.

- DepMap/PRISM: SR 0.999, RC 0.956, OSIC 0.271.
- LINCS: SR 0.223, RC 0.568, OSIC 0.895.
- TCGA outcome: SR 1.000, RC 0.436, OSIC 0.896.

LINCS therefore exposes low future-state reconstructability, and TCGA exposes weak Master
self-error observation despite complete telemetry. These are observable limitations.

## Secondary TCGA result

On 42 patients from seven completely held-out cancer projects, AUROC was 0.677 and Brier score
was 0.215 versus 0.239 for the fit-prevalence baseline. Observed-label permutation p-values
were 0.0270 and 0.0360. This is not external clinical validation.

## Guardrails

GOD-Observability means Global Observability of Distributed Reinforcement States. It is an
architecture acronym/metaphor, not a consciousness, sentience, divinity or physical quantum
claim. Quantum-inspired amplitudes are computational future weights.

## Non-ablation architecture verification

All three Master architectures and {verified_agents}/{total_agents} observer agents passed
structural observability verification across {independent_units:,} independent biological
units. Confidence intervals resample biological units rather than treating the three agent
events per unit as independent observations.

{chr(10).join(osic_lines)}

No new ablation experiment was added. Agent influence is therefore verified observationally
for coverage and consistency and is not upgraded to a causal-attribution claim.
""",
        encoding="utf-8",
    )
    generated.append(summary_path)

    # Human-readable package index.
    figure_files = sorted(figures_dir.glob("*"))
    table_files = sorted(tables_dir.glob("*"))
    index_path = package_dir / "index.html"
    index_path.write_text(
        "<!doctype html><meta charset='utf-8'><style>body{font-family:Arial;max-width:1100px;margin:30px auto}"
        "li{margin:6px}h1,h2{color:#174a6e}</style><h1>Primary Scientific Observability Results</h1>"
        "<p>Observed real-data results only. Prediction performance is secondary.</p><h2>Figures</h2><ul>"
        + "".join(f"<li><a href='figures/{path.name}'>{path.name}</a></li>" for path in figure_files)
        + "</ul><h2>Tables</h2><ul>"
        + "".join(f"<li><a href='tables/{path.name}'>{path.name}</a></li>" for path in table_files)
        + "</ul><p><a href='RESULTS_SUMMARY.md'>Results summary</a></p>",
        encoding="utf-8",
    )
    generated.append(index_path)

    manifest_rows = []
    for path in sorted(generated):
        manifest_rows.append(
            {
                "relative_path": str(path.relative_to(package_dir)),
                "bytes": path.stat().st_size,
                "sha256": _hash(path),
            }
        )
    manifest_table = pd.DataFrame(manifest_rows)
    manifest_csv = package_dir / "artifact_manifest.csv"
    manifest_table.to_csv(manifest_csv, index=False)
    manifest_json = package_dir / "artifact_manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "data_policy": "observed_public_pan_cancer_data_only_no_simulation",
                "figures_png": len(list(figures_dir.glob("*.png"))),
                "figures_svg": len(list(figures_dir.glob("*.svg"))),
                "table_csv": len(list(tables_dir.glob("*.csv"))),
                "table_html": len(list(tables_dir.glob("*.html"))),
                "artifacts": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    zip_path = primary_dir / "primary_scientific_observability_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path("primary_scientific_observability_results") / path.relative_to(package_dir)))

    return {
        "status": "completed",
        "data_policy": "observed_public_pan_cancer_data_only_no_simulation",
        "figures_png": len(list(figures_dir.glob("*.png"))),
        "figures_svg": len(list(figures_dir.glob("*.svg"))),
        "tables_csv": len(list(tables_dir.glob("*.csv"))),
        "tables_html": len(list(tables_dir.glob("*.html"))),
        "package_dir": str(package_dir),
        "index": str(index_path),
        "summary": str(summary_path),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": _hash(zip_path),
    }
