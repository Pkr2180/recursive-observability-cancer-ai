from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from src.observability import (
    TransitionEvent,
    append_transition_event,
    build_recursive_observability_report,
)
from src.observability.recursive import trace_completeness
from src.validation.lincs_future_state import AGENT_NAMES, SEED


EPSILON = np.finfo(float).eps


def _numbered_columns(table: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in table.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[-1]))


def _state_arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before_columns = _numbered_columns(table, "observed_before_state_")
    after_columns = _numbered_columns(table, "observed_after_state_")
    if not before_columns or len(before_columns) != len(after_columns):
        raise ValueError("observed before/after state columns are missing or inconsistent")
    forecasts = []
    for agent in AGENT_NAMES:
        columns = _numbered_columns(table, f"{agent}_forecast_")
        if len(columns) != len(after_columns):
            raise ValueError(f"forecast dimensions are incomplete for {agent}")
        forecasts.append(table[columns].to_numpy(dtype=float))
    return (
        table[before_columns].to_numpy(dtype=float),
        table[after_columns].to_numpy(dtype=float),
        np.stack(forecasts),
    )


def _normalized_error(prediction: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - observed, axis=1) / np.sqrt(observed.shape[1])


def _reward(error: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(error, dtype=float))


def _context(table: pd.DataFrame, before: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    mean_forecast = forecasts.mean(axis=0)
    spread = forecasts.std(axis=0)
    distance_to_mean = np.column_stack(
        [_normalized_error(forecasts[index], mean_forecast) for index in range(len(AGENT_NAMES))]
    )
    predicted_change = np.column_stack(
        [_normalized_error(forecasts[index], before) for index in range(len(AGENT_NAMES))]
    )
    pairwise = np.column_stack(
        [
            _normalized_error(forecasts[left], forecasts[right])
            for left in range(len(AGENT_NAMES))
            for right in range(left + 1, len(AGENT_NAMES))
        ]
    )
    timing = np.column_stack(
        [
            np.log1p(table["before_hours"].to_numpy(dtype=float)),
            np.log1p(table["after_hours"].to_numpy(dtype=float)),
            np.log1p(table["delta_hours"].to_numpy(dtype=float)),
            np.log1p(np.abs(table["dose_numeric"].to_numpy(dtype=float))),
            table["dose_missing"].to_numpy(dtype=float),
        ]
    )
    context = np.column_stack(
        [
            before,
            mean_forecast,
            spread,
            distance_to_mean,
            predicted_change,
            pairwise,
            timing,
        ]
    )
    if not np.isfinite(context).all():
        raise ValueError("non-finite policy context")
    return context


def _reward_model(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=320,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def _softmax_weights(values: np.ndarray, temperature: float) -> np.ndarray:
    centered = (
        np.asarray(values, dtype=float) - np.max(values, axis=1, keepdims=True)
    ) / max(float(temperature), EPSILON)
    centered = np.clip(centered, -60, 60)
    weights = np.exp(centered)
    return weights / weights.sum(axis=1, keepdims=True)


def _policy_forecast(forecasts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("na,and->nd", weights, forecasts)


def _cross_fitted_q(
    context: np.ndarray,
    rewards: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    folds = min(5, len(unique_groups))
    if folds < 2:
        raise ValueError("at least two calibration perturbations are required")
    output = np.full_like(rewards, np.nan, dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit_index, validation_index) in enumerate(
        splitter.split(context, groups=groups)
    ):
        for agent_index in range(rewards.shape[1]):
            model = _reward_model(SEED + 100 * fold + agent_index)
            model.fit(context[fit_index], rewards[fit_index, agent_index])
            output[validation_index, agent_index] = model.predict(context[validation_index])
    if not np.isfinite(output).all():
        raise ValueError("cross-fitted reward predictions are incomplete")
    return np.clip(output, 0, 1)


def _choose_policy(
    q_values: np.ndarray,
    forecasts: np.ndarray,
    observed: np.ndarray,
) -> tuple[str, float, np.ndarray, pd.DataFrame]:
    candidates: list[dict[str, Any]] = []
    hard = np.eye(len(AGENT_NAMES))[np.argmax(q_values, axis=1)]
    hard_error = _normalized_error(_policy_forecast(forecasts, hard), observed)
    candidates.append(
        {
            "mode": "hard_selection",
            "temperature": 0.0,
            "mean_reward": float(_reward(hard_error).mean()),
            "weights": hard,
        }
    )
    for temperature in (0.025, 0.05, 0.1, 0.2, 0.5, 1.0):
        weights = _softmax_weights(q_values, temperature)
        error = _normalized_error(_policy_forecast(forecasts, weights), observed)
        candidates.append(
            {
                "mode": "reward_weighted_mixture",
                "temperature": temperature,
                "mean_reward": float(_reward(error).mean()),
                "weights": weights,
            }
        )
    chosen = max(candidates, key=lambda row: row["mean_reward"])
    tuning = pd.DataFrame(
        [
            {
                "mode": row["mode"],
                "temperature": row["temperature"],
                "cross_fitted_calibration_reward": row["mean_reward"],
            }
            for row in candidates
        ]
    )
    return (
        str(chosen["mode"]),
        float(chosen["temperature"]),
        np.asarray(chosen["weights"], dtype=float),
        tuning,
    )


def _weights_for_policy(q_values: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    if mode == "hard_selection":
        return np.eye(len(AGENT_NAMES))[np.argmax(q_values, axis=1)]
    return _softmax_weights(q_values, temperature)


def _self_context(
    context: np.ndarray,
    q_values: np.ndarray,
    weights: np.ndarray,
    forecasts: np.ndarray,
    policy_forecast: np.ndarray,
) -> np.ndarray:
    policy_disagreement = np.mean(
        np.stack(
            [_normalized_error(forecasts[index], policy_forecast) for index in range(len(AGENT_NAMES))],
            axis=1,
        ),
        axis=1,
    )
    entropy = -np.sum(weights * np.log(weights + EPSILON), axis=1) / np.log(len(AGENT_NAMES))
    return np.column_stack([context, q_values, weights, policy_disagreement, entropy])


def _tree_predictions(model: RandomForestRegressor, features: np.ndarray) -> np.ndarray:
    return np.column_stack([tree.predict(features) for tree in model.estimators_])


def _cluster_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def _bootstrap_comparisons(
    primary_reward: np.ndarray,
    baselines: dict[str, np.ndarray],
    groups: np.ndarray,
    replicates: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 901)
    rows = []
    unique_groups = np.unique(groups)
    for name, baseline in baselines.items():
        difference = np.asarray(primary_reward) - np.asarray(baseline)
        boot = np.empty(replicates, dtype=float)
        for index in range(replicates):
            selected = _cluster_indices(groups, rng)
            boot[index] = float(difference[selected].mean())
        group_difference = np.asarray(
            [difference[groups == group].mean() for group in unique_groups], dtype=float
        )
        null = np.empty(replicates, dtype=float)
        for index in range(replicates):
            signs = rng.choice([-1.0, 1.0], size=len(unique_groups))
            null[index] = float(np.mean(group_difference * signs))
        estimate = float(difference.mean())
        rows.append(
            {
                "comparison": f"reinforcement_master_minus_{name}",
                "mean_reward_difference": estimate,
                "ci_95_low": float(np.quantile(boot, 0.025)),
                "ci_95_high": float(np.quantile(boot, 0.975)),
                "cluster_sign_flip_p": float(
                    (1 + np.sum(np.abs(null) >= abs(estimate))) / (replicates + 1)
                ),
                "cluster_unit": "held-out perturbation identifier",
                "clusters": int(len(unique_groups)),
                "replicates": int(replicates),
            }
        )
    return pd.DataFrame(rows)


def _risk_coverage(actual_error: np.ndarray, predicted_error: np.ndarray) -> pd.DataFrame:
    order = np.argsort(predicted_error)
    rows = []
    for coverage in (1.0, 0.8, 0.6, 0.4, 0.2):
        count = max(1, int(np.ceil(coverage * len(order))))
        retained = order[:count]
        rows.append(
            {
                "coverage": coverage,
                "retained_transitions": count,
                "mean_realized_error": float(actual_error[retained].mean()),
                "rmse_realized_error": float(np.sqrt(np.mean(actual_error[retained] ** 2))),
                "maximum_predicted_self_error": float(predicted_error[retained].max()),
            }
        )
    return pd.DataFrame(rows)


def _cosine(prediction: np.ndarray, observed: np.ndarray) -> float:
    numerator = np.sum(prediction * observed, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(observed, axis=1)
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(values.mean())


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figures(
    output_dir: Path,
    evaluation: pd.DataFrame,
    comparisons: pd.DataFrame,
    weights: np.ndarray,
    predicted_error: np.ndarray,
    actual_error: np.ndarray,
    risk: pd.DataFrame,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ordered = evaluation.sort_values("mean_reward", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.bar(ordered["method"], ordered["mean_reward"], color="#277da1")
    ax.set_ylabel("Observed holdout reward")
    ax.set_title("Reward-driven Master versus frozen observer policies")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, figures_dir, "figure_01_holdout_reward_comparison")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    means = weights.mean(axis=0)
    ax.bar(AGENT_NAMES, means, color=["#277da1", "#f8961e", "#43aa8b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean learned policy weight")
    ax.set_title("Reward-conditioned observer allocation")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, figures_dir, "figure_02_learned_observer_weights")

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    ax.scatter(predicted_error, actual_error, alpha=0.55, s=24, color="#7b2cbf")
    low = float(min(predicted_error.min(), actual_error.min()))
    high = float(max(predicted_error.max(), actual_error.max()))
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
    ax.set_xlabel("Predicted Master self-error")
    ax.set_ylabel("Observed Master error")
    ax.set_title("Recursive self-observation on unseen perturbations")
    _save_figure(fig, figures_dir, "figure_03_master_self_observation")

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.plot(risk["coverage"], risk["mean_realized_error"], marker="o", linewidth=2)
    ax.set_xlabel("Retained coverage after deferring predicted high-error transitions")
    ax.set_ylabel("Mean observed error")
    ax.set_title("Uncertainty-gated selective risk")
    ax.invert_xaxis()
    _save_figure(fig, figures_dir, "figure_04_selective_risk")

    comparison_path = output_dir / "cluster_bootstrap_comparisons.csv"
    if not comparison_path.exists():
        comparisons.to_csv(comparison_path, index=False)


def run_reinforcement_master_validation(
    results_dir: Path,
    model_dir: Path,
    *,
    replicates: int = 2000,
) -> dict[str, Any]:
    """Train and test a reward-driven observer policy on frozen real LINCS transitions."""
    source_dir = results_dir / "validation" / "lincs_future_state"
    calibration_path = source_dir / "frozen_calibration_transition_audit.parquet"
    holdout_path = source_dir / "frozen_holdout_transition_audit.parquet"
    split_path = source_dir / "frozen_perturbation_split.json"
    for path in (calibration_path, holdout_path, split_path):
        if not path.exists():
            raise FileNotFoundError(path)

    calibration = pd.read_parquet(calibration_path)
    holdout = pd.read_parquet(holdout_path)
    if set(calibration["pert_id"].astype(str)) & set(holdout["pert_id"].astype(str)):
        raise ValueError("calibration and holdout perturbations overlap")

    calibration_before, calibration_after, calibration_forecasts = _state_arrays(calibration)
    holdout_before, holdout_after, holdout_forecasts = _state_arrays(holdout)
    calibration_context = _context(calibration, calibration_before, calibration_forecasts)
    holdout_context = _context(holdout, holdout_before, holdout_forecasts)
    calibration_errors = np.column_stack(
        [
            _normalized_error(calibration_forecasts[index], calibration_after)
            for index in range(len(AGENT_NAMES))
        ]
    )
    calibration_rewards = _reward(calibration_errors)
    calibration_groups = calibration["pert_id"].astype(str).to_numpy()
    holdout_groups = holdout["pert_id"].astype(str).to_numpy()

    oof_q = _cross_fitted_q(calibration_context, calibration_rewards, calibration_groups)
    mode, temperature, oof_weights, tuning = _choose_policy(
        oof_q, calibration_forecasts, calibration_after
    )
    oof_forecast = _policy_forecast(calibration_forecasts, oof_weights)
    oof_error = _normalized_error(oof_forecast, calibration_after)
    oof_self_context = _self_context(
        calibration_context, oof_q, oof_weights, calibration_forecasts, oof_forecast
    )

    q_models: dict[str, RandomForestRegressor] = {}
    holdout_q = np.empty((len(holdout), len(AGENT_NAMES)), dtype=float)
    for agent_index, agent in enumerate(AGENT_NAMES):
        model = _reward_model(SEED + agent_index)
        model.fit(calibration_context, calibration_rewards[:, agent_index])
        q_models[agent] = model
        holdout_q[:, agent_index] = model.predict(holdout_context)
    holdout_q = np.clip(holdout_q, 0, 1)
    holdout_weights = _weights_for_policy(holdout_q, mode, temperature)
    reinforcement_forecast = _policy_forecast(holdout_forecasts, holdout_weights)
    reinforcement_error = _normalized_error(reinforcement_forecast, holdout_after)
    reinforcement_reward = _reward(reinforcement_error)

    self_model = RandomForestRegressor(
        n_estimators=480,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=SEED + 700,
        n_jobs=-1,
    )
    self_model.fit(oof_self_context, oof_error)
    holdout_self_context = _self_context(
        holdout_context,
        holdout_q,
        holdout_weights,
        holdout_forecasts,
        reinforcement_forecast,
    )
    self_tree_predictions = _tree_predictions(self_model, holdout_self_context)
    predicted_self_error = np.clip(self_tree_predictions.mean(axis=1), 0, None)
    meta_uncertainty = self_tree_predictions.std(axis=1)

    fixed_master_columns = _numbered_columns(holdout, "master_forecast_state_")
    fixed_master = holdout[fixed_master_columns].to_numpy(dtype=float)
    equal_forecast = holdout_forecasts.mean(axis=0)
    fixed_agent_index = int(np.argmax(calibration_rewards.mean(axis=0)))
    best_fixed_forecast = holdout_forecasts[fixed_agent_index]
    hard_weights = np.eye(len(AGENT_NAMES))[np.argmax(holdout_q, axis=1)]
    hard_forecast = _policy_forecast(holdout_forecasts, hard_weights)

    method_forecasts: dict[str, np.ndarray] = {
        "reinforcement_master": reinforcement_forecast,
        "hard_reward_policy": hard_forecast,
        "frozen_master": fixed_master,
        "equal_ensemble": equal_forecast,
        f"best_fixed_agent_{AGENT_NAMES[fixed_agent_index]}": best_fixed_forecast,
    }
    for agent_index, agent in enumerate(AGENT_NAMES):
        method_forecasts[f"agent_{agent}"] = holdout_forecasts[agent_index]

    evaluation_rows = []
    method_rewards: dict[str, np.ndarray] = {}
    for method, prediction in method_forecasts.items():
        error = _normalized_error(prediction, holdout_after)
        rewards = _reward(error)
        method_rewards[method] = rewards
        evaluation_rows.append(
            {
                "method": method,
                "mean_normalized_l2_error": float(error.mean()),
                "rmse_normalized_l2_error": float(np.sqrt(np.mean(error**2))),
                "mean_reward": float(rewards.mean()),
                "forecast_cosine_similarity": _cosine(prediction, holdout_after),
                "holdout_transitions": int(len(holdout)),
                "holdout_perturbations": int(len(np.unique(holdout_groups))),
            }
        )
    evaluation = pd.DataFrame(evaluation_rows)

    comparison_baselines = {
        name: rewards
        for name, rewards in method_rewards.items()
        if name != "reinforcement_master"
    }
    comparisons = _bootstrap_comparisons(
        reinforcement_reward, comparison_baselines, holdout_groups, replicates
    )
    risk = _risk_coverage(reinforcement_error, predicted_self_error)

    policy_entropy = -np.sum(
        holdout_weights * np.log(holdout_weights + EPSILON), axis=1
    ) / np.log(len(AGENT_NAMES))
    policy_disagreement = np.mean(
        np.stack(
            [
                _normalized_error(holdout_forecasts[index], reinforcement_forecast)
                for index in range(len(AGENT_NAMES))
            ],
            axis=1,
        ),
        axis=1,
    )
    disagreement_scale = max(
        float(
            np.quantile(
                np.mean(
                    np.stack(
                        [
                            _normalized_error(calibration_forecasts[index], oof_forecast)
                            for index in range(len(AGENT_NAMES))
                        ],
                        axis=1,
                    ),
                    axis=1,
                ),
                0.95,
            )
        ),
        EPSILON,
    )
    master_uncertainty = 0.5 * policy_entropy + 0.5 * np.clip(
        policy_disagreement / disagreement_scale, 0, 1
    )
    defer_threshold = float(np.quantile(self_model.predict(oof_self_context), 0.8))

    output_dir = results_dir / "reinforcement_master_brain"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "reinforcement_master_events.jsonl"
    events_path.unlink(missing_ok=True)
    event_records = []
    for row_index, row in holdout.reset_index(drop=True).iterrows():
        evidence = (
            str(row["before_evidence_sig_ids"]).split("|")[:8]
            + str(row["after_evidence_sig_ids"]).split("|")[:8]
            + ["NCBI_GEO:GSE70138"]
        )
        future_weights = {
            agent: float(holdout_weights[row_index, agent_index])
            for agent_index, agent in enumerate(AGENT_NAMES)
        }
        amplitudes = {agent: float(np.sqrt(weight)) for agent, weight in future_weights.items()}
        gate = (
            "defer_high_predicted_self_error"
            if predicted_self_error[row_index] > defer_threshold
            else "act_reward_conditioned_policy"
        )
        cancer_state_id = (
            f"{row['cell_id']}|{row['pert_id']}|{row['pert_idose']}|"
            f"{row['before_hours']:g}-{row['after_hours']:g}h"
        )
        for agent_index, agent in enumerate(AGENT_NAMES):
            event = TransitionEvent(
                episode_id=f"reinforcement-master-{row_index:05d}",
                time=f"observed-{row['after_hours']:g}h",
                cancer_state_id=cancer_state_id,
                agent_id=agent,
                agent_state_before=holdout_before[row_index].tolist(),
                agent_state_after=holdout_forecasts[agent_index, row_index].tolist(),
                action=f"{mode}_observer_policy",
                reward=float(
                    _reward(
                        _normalized_error(
                            holdout_forecasts[agent_index, row_index : row_index + 1],
                            holdout_after[row_index : row_index + 1],
                        )
                    )[0]
                ),
                agent_uncertainty=float(1.0 - holdout_q[row_index, agent_index]),
                master_state_before=fixed_master[row_index].tolist(),
                master_state_after=reinforcement_forecast[row_index].tolist(),
                master_uncertainty=float(master_uncertainty[row_index]),
                meta_uncertainty=float(meta_uncertainty[row_index]),
                evidence_ids=evidence,
                parent_trace=f"reinforcement-master-{row_index:05d}",
                future_states_considered=future_weights,
                quantum_inspired_state=amplitudes,
                gate_applied=gate,
                hypothesis_before="frozen_observer_weighting",
                hypothesis_after="reward_conditioned_observer_policy",
            )
            append_transition_event(events_path, event)
            event_records.append(event.to_record())

    agent_errors = np.column_stack(
        [
            _normalized_error(holdout_forecasts[index], holdout_after)
            for index in range(len(AGENT_NAMES))
        ]
    )
    cancer_instability = holdout["cancer_future_state_instability"].to_numpy(dtype=float)
    observability = build_recursive_observability_report(
        events=event_records,
        cancer_instability=cancer_instability,
        individual_agent_instability=agent_errors.mean(axis=1),
        inter_agent_disagreement=policy_disagreement,
        master_instability=reinforcement_error,
        meta_uncertainty=meta_uncertainty,
        master_state=holdout_after,
        reconstructed_master_state=reinforcement_forecast,
        self_observation=reinforcement_error.reshape(-1, 1),
        reconstructed_self_observation=predicted_self_error.reshape(-1, 1),
        agent_influence_identifiability=float(np.mean(np.isfinite(holdout_weights).all(axis=1))),
        uncertainty_visibility=float(
            np.mean(
                np.isfinite(
                    np.column_stack(
                        [master_uncertainty, predicted_self_error, meta_uncertainty]
                    )
                ).all(axis=1)
            )
        ),
    )

    predictions = holdout[
        [
            "cell_id",
            "pert_id",
            "pert_idose",
            "before_hours",
            "after_hours",
            "delta_hours",
            "before_evidence_sig_ids",
            "after_evidence_sig_ids",
        ]
    ].copy()
    for agent_index, agent in enumerate(AGENT_NAMES):
        predictions[f"predicted_reward_{agent}"] = holdout_q[:, agent_index]
        predictions[f"policy_weight_{agent}"] = holdout_weights[:, agent_index]
        predictions[f"observed_reward_{agent}"] = _reward(agent_errors[:, agent_index])
    predictions["reinforcement_master_error"] = reinforcement_error
    predictions["reinforcement_master_reward"] = reinforcement_reward
    predictions["predicted_master_self_error"] = predicted_self_error
    predictions["self_observation_meta_uncertainty"] = meta_uncertainty
    predictions["master_policy_uncertainty"] = master_uncertainty
    predictions["gate_applied"] = np.where(
        predicted_self_error > defer_threshold,
        "defer_high_predicted_self_error",
        "act_reward_conditioned_policy",
    )
    for component in range(holdout_after.shape[1]):
        predictions[f"observed_after_state_{component + 1}"] = holdout_after[:, component]
        predictions[f"reinforcement_forecast_state_{component + 1}"] = reinforcement_forecast[
            :, component
        ]

    weight_summary = pd.DataFrame(
        {
            "agent": AGENT_NAMES,
            "mean_policy_weight": holdout_weights.mean(axis=0),
            "median_policy_weight": np.median(holdout_weights, axis=0),
            "selection_rate": np.mean(
                np.argmax(holdout_weights, axis=1)[:, None] == np.arange(len(AGENT_NAMES)), axis=0
            ),
            "mean_predicted_reward": holdout_q.mean(axis=0),
            "mean_observed_reward": np.mean(_reward(agent_errors), axis=0),
        }
    )

    evaluation.to_csv(output_dir / "holdout_method_evaluation.csv", index=False)
    comparisons.to_csv(output_dir / "cluster_bootstrap_comparisons.csv", index=False)
    risk.to_csv(output_dir / "selective_risk_coverage.csv", index=False)
    weight_summary.to_csv(output_dir / "learned_policy_weights.csv", index=False)
    tuning.to_csv(output_dir / "calibration_policy_tuning.csv", index=False)
    predictions.to_parquet(output_dir / "holdout_policy_predictions.parquet", index=False)

    model_output_dir = model_dir / "reinforcement_master_brain"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_output_dir / "reward_conditioned_observer_policy.joblib"
    joblib.dump(
        {
            "q_models": q_models,
            "self_observer": self_model,
            "agent_order": AGENT_NAMES,
            "policy_mode": mode,
            "temperature": temperature,
            "defer_threshold": defer_threshold,
            "data_policy": "observed LINCS transitions only; no biological simulation",
        },
        model_path,
    )

    _figures(
        output_dir,
        evaluation,
        comparisons,
        holdout_weights,
        predicted_self_error,
        reinforcement_error,
        risk,
    )

    baseline_reward = float(method_rewards["frozen_master"].mean())
    result = {
        "status": "completed",
        "architecture": "recursive_observable_offline_reinforcement_master",
        "learning_problem": "reward-conditioned observer selection and reweighting",
        "data_policy": "real observed LINCS cancer transitions only; no simulated biological trajectories",
        "calibration_transitions": int(len(calibration)),
        "calibration_perturbations": int(len(np.unique(calibration_groups))),
        "holdout_transitions": int(len(holdout)),
        "holdout_perturbations": int(len(np.unique(holdout_groups))),
        "policy_mode": mode,
        "policy_temperature": temperature,
        "best_fixed_agent": AGENT_NAMES[fixed_agent_index],
        "reinforcement_master_mean_reward": float(reinforcement_reward.mean()),
        "frozen_master_mean_reward": baseline_reward,
        "reward_improvement_over_frozen_master": float(
            reinforcement_reward.mean() - baseline_reward
        ),
        "reinforcement_master_mean_error": float(reinforcement_error.mean()),
        "self_error_pearson": float(
            np.corrcoef(predicted_self_error, reinforcement_error)[0, 1]
        ),
        "trace_completeness": trace_completeness(event_records),
        "strict_event_count": int(len(event_records)),
        "recursive_observability": observability.to_dict(),
        "claim_boundary": (
            "Offline observer-policy reinforcement on observed cancer-cell perturbation transitions; "
            "not a clinical treatment policy and not evidence from simulated biological rollouts."
        ),
        "artifacts": {
            "events": str(events_path),
            "evaluation": str(output_dir / "holdout_method_evaluation.csv"),
            "comparisons": str(output_dir / "cluster_bootstrap_comparisons.csv"),
            "risk_coverage": str(output_dir / "selective_risk_coverage.csv"),
            "policy_weights": str(output_dir / "learned_policy_weights.csv"),
            "predictions": str(output_dir / "holdout_policy_predictions.parquet"),
            "models": str(model_path),
            "figures": str(output_dir / "figures"),
        },
    }
    report_path = output_dir / "reinforcement_master_validation_report.json"
    result["artifacts"]["report"] = str(report_path)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
