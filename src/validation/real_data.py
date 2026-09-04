from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LINEAGE_COLUMNS = (
    "OncotreeLineage",
    "OncotreePrimaryDisease",
    "lineage",
    "primary_disease",
)


def _model_metadata(processed_dir: Path) -> tuple[pd.DataFrame, str]:
    path = processed_dir / "depmap_figshare" / "depmap_model_metadata.parquet"
    model = pd.read_parquet(path)
    id_column = "ModelID" if "ModelID" in model.columns else model.columns[0]
    lineage_column = next((column for column in LINEAGE_COLUMNS if column in model.columns), "")
    if not lineage_column:
        raise ValueError(f"No lineage column found in DepMap metadata: {list(model.columns)}")
    model = model[[id_column, lineage_column]].rename(
        columns={id_column: "ModelID", lineage_column: "lineage"}
    )
    model["ModelID"] = model["ModelID"].astype(str)
    model["lineage"] = model["lineage"].fillna("unknown").astype(str)
    return model.drop_duplicates("ModelID"), lineage_column


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / std


def _collapse_stats(expression: pd.Series, dependency: pd.Series) -> dict[str, float | int]:
    x = _zscore(expression)
    y = _zscore(dependency)
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return {"models_tested": int(mask.sum()), "correlation": np.nan, "collapse_score": 0}
    correlation = float(np.corrcoef(x[mask], y[mask])[0, 1])
    high_expression_low_dependency = int(((x > 1) & (y < -1)).sum())
    low_expression_high_dependency = int(((x < -1) & (y > 1)).sum())
    concordant_high = int(((x > 1) & (y > 1)).sum())
    return {
        "models_tested": int(mask.sum()),
        "correlation": correlation,
        "collapse_score": high_expression_low_dependency
        + low_expression_high_dependency
        - concordant_high,
    }


def _candidate_genes(results_dir: Path, available: Iterable[str], top_n: int) -> list[str]:
    path = results_dir / "rule_collapse" / "depmap_expression_dependency_collapse_top.csv"
    available_set = set(available)
    if path.exists():
        genes = pd.read_csv(path)["gene"].astype(str).tolist()
        return [gene for gene in genes if gene in available_set][:top_n]
    return sorted(available_set)[:top_n]


def _lineage_expression_dependency(
    expression: pd.DataFrame,
    dependency: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list[str],
    min_models: int,
) -> pd.DataFrame:
    rows = []
    model_lineage = metadata.set_index("ModelID")["lineage"]
    for lineage, lineage_ids in model_lineage.groupby(model_lineage).groups.items():
        ids = sorted(set(lineage_ids) & set(expression.index) & set(dependency.index))
        if len(ids) < min_models or lineage == "unknown":
            continue
        for gene in genes:
            stats = _collapse_stats(expression.loc[ids, gene], dependency.loc[ids, gene])
            rows.append({"lineage": lineage, "gene": gene, **stats})
    return pd.DataFrame(rows)


def _permutation_nulls(
    expression: pd.DataFrame,
    dependency: pd.DataFrame,
    genes: list[str],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for gene in genes:
        x = pd.to_numeric(expression[gene], errors="coerce")
        y = pd.to_numeric(dependency[gene], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        observed = _collapse_stats(x, y)
        null_correlations = np.empty(permutations, dtype=float)
        null_scores = np.empty(permutations, dtype=float)
        y_values = y.to_numpy(copy=True)
        for index in range(permutations):
            permuted = pd.Series(rng.permutation(y_values))
            stats = _collapse_stats(x, permuted)
            null_correlations[index] = float(stats["correlation"])
            null_scores[index] = float(stats["collapse_score"])
        correlation_p = (1 + np.sum(np.abs(null_correlations) >= abs(observed["correlation"]))) / (
            permutations + 1
        )
        score_p = (1 + np.sum(null_scores >= observed["collapse_score"])) / (permutations + 1)
        rows.append(
            {
                "gene": gene,
                "models_tested": observed["models_tested"],
                "observed_correlation": observed["correlation"],
                "observed_collapse_score": observed["collapse_score"],
                "null_correlation_mean": float(np.mean(null_correlations)),
                "null_collapse_score_mean": float(np.mean(null_scores)),
                "correlation_empirical_p": float(correlation_p),
                "collapse_score_empirical_p": float(score_p),
                "permutations": permutations,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["collapse_score_empirical_p", "correlation_empirical_p", "observed_collapse_score"],
        ascending=[True, True, False],
    )


def _curate_drug_targets(processed_dir: Path, results_dir: Path) -> pd.DataFrame:
    results_path = (
        results_dir / "rule_collapse" / "depmap_prism_dependency_pharmacology_inversion_all.csv"
    )
    compounds_path = (
        processed_dir / "prism_repurposing_24q2" / "prism_compound_metadata.parquet"
    )
    crispr_path = processed_dir / "depmap_figshare" / "depmap_crispr_gene_effect_top_genes.parquet"
    results = pd.read_csv(results_path)
    compounds = pd.read_parquet(compounds_path)
    crispr_genes = set(pd.read_parquet(crispr_path).columns) - {"ModelID"}
    compound_ids = set(compounds["IDs"].astype(str)) if "IDs" in compounds.columns else set()
    curated = results.copy()
    curated["target_symbol_valid"] = curated["target"].astype(str).str.match(
        re.compile(r"^[A-Z0-9][A-Z0-9-]{1,14}$")
    )
    curated["target_in_crispr_panel"] = curated["target"].isin(crispr_genes)
    curated["treatment_in_compound_manifest"] = curated["treatment_id"].astype(str).isin(compound_ids)
    curated["moa_documented"] = curated["moa"].fillna("").astype(str).str.strip().ne("")
    required = [
        "target_symbol_valid",
        "target_in_crispr_panel",
        "treatment_in_compound_manifest",
        "moa_documented",
    ]
    curated["curation_status"] = np.where(curated[required].all(axis=1), "metadata_supported", "review")
    return curated


def _lineage_pharmacology(
    processed_dir: Path,
    curated: pd.DataFrame,
    metadata: pd.DataFrame,
    top_n: int,
    min_models: int,
) -> pd.DataFrame:
    crispr = pd.read_parquet(
        processed_dir / "depmap_figshare" / "depmap_crispr_gene_effect_top_genes.parquet"
    ).set_index("ModelID")
    dependency = -crispr
    response = pd.read_parquet(
        processed_dir / "prism_repurposing_24q2" / "prism_primary_top_treatments.parquet"
    ).set_index("ModelID")
    sensitivity = -response.apply(pd.to_numeric, errors="coerce")
    model_lineage = metadata.set_index("ModelID")["lineage"]
    rows = []
    for pair in curated.head(top_n).itertuples(index=False):
        if pair.target not in dependency.columns or pair.treatment_id not in sensitivity.columns:
            continue
        common = set(dependency.index) & set(sensitivity.index) & set(model_lineage.index)
        pair_lineages = model_lineage.loc[sorted(common)]
        for lineage, ids in pair_lineages.groupby(pair_lineages).groups.items():
            ids = list(ids)
            if len(ids) < min_models or lineage == "unknown":
                continue
            dep = dependency.loc[ids, pair.target]
            sens = sensitivity.loc[ids, pair.treatment_id]
            mask = dep.notna() & sens.notna()
            if mask.sum() < min_models or dep[mask].std() == 0 or sens[mask].std() == 0:
                continue
            rows.append(
                {
                    "lineage": lineage,
                    "target": pair.target,
                    "treatment_id": pair.treatment_id,
                    "drug_name": pair.drug_name,
                    "models_tested": int(mask.sum()),
                    "dependency_sensitivity_correlation": float(
                        np.corrcoef(dep[mask], sens[mask])[0, 1]
                    ),
                    "curation_status": pair.curation_status,
                }
            )
    return pd.DataFrame(rows)


def run_real_data_validation(
    processed_dir: Path,
    results_dir: Path,
    *,
    top_genes: int = 50,
    top_drug_pairs: int = 100,
    permutations: int = 200,
    min_lineage_models: int = 20,
    seed: int = 20260824,
) -> dict:
    """Validate real pan-cancer signals without generating synthetic samples."""
    depmap_dir = processed_dir / "depmap_figshare"
    expression = pd.read_parquet(depmap_dir / "depmap_expression_top_genes.parquet").set_index(
        "ModelID"
    )
    dependency = -pd.read_parquet(
        depmap_dir / "depmap_crispr_gene_effect_top_genes.parquet"
    ).set_index("ModelID")
    common_models = sorted(set(expression.index) & set(dependency.index))
    common_genes = sorted(set(expression.columns) & set(dependency.columns))
    expression = expression.loc[common_models, common_genes]
    dependency = dependency.loc[common_models, common_genes]
    metadata, source_lineage_column = _model_metadata(processed_dir)
    genes = _candidate_genes(results_dir, common_genes, top_genes)

    lineage = _lineage_expression_dependency(
        expression, dependency, metadata, genes, min_lineage_models
    )
    nulls = _permutation_nulls(expression, dependency, genes, permutations, seed)
    curated = _curate_drug_targets(processed_dir, results_dir)
    lineage_drugs = _lineage_pharmacology(
        processed_dir, curated, metadata, top_drug_pairs, min_lineage_models
    )

    out_dir = results_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "lineage_expression_dependency": out_dir / "lineage_expression_dependency.csv",
        "expression_dependency_nulls": out_dir / "expression_dependency_permutation_nulls.csv",
        "curated_drug_targets": out_dir / "curated_drug_target_pairs.csv",
        "lineage_dependency_pharmacology": out_dir / "lineage_dependency_pharmacology.csv",
    }
    lineage.to_csv(outputs["lineage_expression_dependency"], index=False)
    nulls.to_csv(outputs["expression_dependency_nulls"], index=False)
    curated.to_csv(outputs["curated_drug_targets"], index=False)
    lineage_drugs.to_csv(outputs["lineage_dependency_pharmacology"], index=False)

    summary = {
        "status": "created",
        "data_policy": "observed_public_pan_cancer_data_only",
        "null_policy": "permutations_rearrange_observed_values; no synthetic samples",
        "source_lineage_column": source_lineage_column,
        "common_models": len(common_models),
        "candidate_genes": len(genes),
        "lineage_expression_dependency_rows": len(lineage),
        "lineages": int(lineage["lineage"].nunique()) if not lineage.empty else 0,
        "permutation_tests": len(nulls),
        "permutations_per_gene": permutations,
        "drug_target_pairs": len(curated),
        "metadata_supported_drug_target_pairs": int(
            (curated["curation_status"] == "metadata_supported").sum()
        ),
        "lineage_dependency_pharmacology_rows": len(lineage_drugs),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "interpretation": (
            "Validation and stratification outputs are hypothesis tests on observed data; "
            "they are not clinical validation or causal proof."
        ),
    }
    summary_path = out_dir / "real_data_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
