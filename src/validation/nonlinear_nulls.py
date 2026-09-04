from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score


CHANNEL_PAIRS = (
    ("cancer_future_state_instability", "individual_agent_instability"),
    ("cancer_future_state_instability", "inter_agent_disagreement"),
    ("cancer_future_state_instability", "master_state_instability"),
    ("cancer_future_state_instability", "meta_uncertainty"),
    ("individual_agent_instability", "inter_agent_disagreement"),
    ("individual_agent_instability", "master_state_instability"),
    ("inter_agent_disagreement", "master_state_instability"),
    ("master_state_instability", "master_self_observation_uncertainty"),
    ("master_state_instability", "meta_uncertainty"),
)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(spearmanr(left, right).statistic)


def _quantile_codes(values: np.ndarray, bins: int = 10) -> np.ndarray:
    unique = np.unique(values).size
    count = max(2, min(bins, unique))
    return pd.qcut(
        pd.Series(values).rank(method="first"),
        q=count,
        labels=False,
        duplicates="drop",
    ).to_numpy(dtype=int)


def _nmi(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        normalized_mutual_info_score(
            _quantile_codes(left),
            _quantile_codes(right),
            average_method="arithmetic",
        )
    )


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def run_lincs_nonlinear_null_audit(
    results_dir: Path,
    *,
    permutations: int = 1000,
    seed: int = 20260824,
) -> dict:
    """Test separated LINCS observability channels against observed-value nulls."""
    out_dir = results_dir / "validation" / "lincs_future_state"
    audit_path = out_dir / "frozen_holdout_transition_audit.parquet"
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    specification = {
        "analysis_status": "exploratory_but_frozen_before_null_execution",
        "seed": seed,
        "permutations": permutations,
        "alpha": 0.05,
        "multiple_testing": "Benjamini-Hochberg across all channel/metric tests",
        "metrics": ["pearson", "spearman", "quantile_normalized_mutual_information"],
        "pairs": [list(pair) for pair in CHANNEL_PAIRS],
        "null": "permute right-hand observed channel; preserve marginal distributions",
        "data_policy": "no simulated biological samples",
    }
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    specification["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    specification_path = out_dir / "frozen_nonlinear_null_specification.json"
    specification_path.write_text(json.dumps(specification, indent=2), encoding="utf-8")

    audit = pd.read_parquet(audit_path)
    rng = np.random.default_rng(seed)
    metric_functions = {
        "pearson": _pearson,
        "spearman": _spearman,
        "quantile_nmi": _nmi,
    }
    rows = []
    for left_name, right_name in CHANNEL_PAIRS:
        pair = audit[[left_name, right_name]].apply(pd.to_numeric, errors="coerce").dropna()
        left = pair[left_name].to_numpy(dtype=float)
        right = pair[right_name].to_numpy(dtype=float)
        for metric_name, function in metric_functions.items():
            observed = function(left, right)
            null = np.empty(permutations, dtype=float)
            for index in range(permutations):
                null[index] = function(left, rng.permutation(right))
            if metric_name in {"pearson", "spearman"}:
                exceedances = int(np.sum(np.abs(null) >= abs(observed)))
            else:
                exceedances = int(np.sum(null >= observed))
            rows.append(
                {
                    "left_channel": left_name,
                    "right_channel": right_name,
                    "metric": metric_name,
                    "observations": int(len(pair)),
                    "observed": observed,
                    "null_mean": float(np.mean(null)),
                    "null_std": float(np.std(null)),
                    "null_95th_percentile": float(np.quantile(null, 0.95)),
                    "null_99th_percentile": float(np.quantile(null, 0.99)),
                    "empirical_p": float((exceedances + 1) / (permutations + 1)),
                    "permutations": permutations,
                    "seed": seed,
                }
            )
    results = pd.DataFrame(rows)
    results["fdr_bh"] = _bh_adjust(results["empirical_p"].to_numpy(dtype=float))
    results["fdr_significant_0_05"] = results["fdr_bh"].le(0.05)
    results_path = out_dir / "nonlinear_observability_null_tests.csv"
    results.to_csv(results_path, index=False)

    cancer_master = results[
        results["left_channel"].eq("cancer_future_state_instability")
        & results["right_channel"].eq("master_state_instability")
    ]
    summary = {
        "status": "created",
        "analysis_status": specification["analysis_status"],
        "data_policy": specification["data_policy"],
        "tests": int(len(results)),
        "fdr_significant_tests": int(results["fdr_significant_0_05"].sum()),
        "cancer_master_tests": cancer_master[
            ["metric", "observed", "empirical_p", "fdr_bh", "fdr_significant_0_05"]
        ].to_dict(orient="records"),
        "specification_sha256": specification["sha256"],
        "specification": str(specification_path),
        "results": str(results_path),
        "interpretation": (
            "A significant cancer-Master coupling result indicates structured association "
            "between observed biological change and forecast error. It does not establish "
            "causality or clinical predictive validity."
        ),
    }
    summary_path = out_dir / "nonlinear_observability_null_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
