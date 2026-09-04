from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _drug_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def _finite_float(value: object) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _direction_concordant(first: float, second: float) -> bool | None:
    if not np.isfinite(first) or not np.isfinite(second) or first == 0 or second == 0:
        return None
    return bool(np.sign(first) == np.sign(second))


def _target_annotation_support(annotations: pd.Series, target: str) -> bool:
    target_key = str(target).upper().strip()
    for annotation in annotations.dropna().astype(str):
        tokens = set(re.findall(r"[A-Z0-9-]+", annotation.upper()))
        if target_key in tokens:
            return True
    return False


def _association(
    observations: pd.DataFrame,
    dependency: pd.Series,
    min_models: int,
) -> dict[str, float | int] | None:
    response = (
        observations.groupby("ModelID", as_index=True)[["LN_IC50", "AUC"]]
        .mean(numeric_only=True)
        .rename(columns={"LN_IC50": "ln_ic50", "AUC": "auc"})
    )
    joined = response.join(dependency.rename("dependency"), how="inner").dropna()
    if len(joined) < min_models:
        return None
    if joined["dependency"].std() == 0:
        return None
    ln_sensitivity = -joined["ln_ic50"]
    auc_sensitivity = -joined["auc"]
    if ln_sensitivity.std() == 0 or auc_sensitivity.std() == 0:
        return None
    return {
        "models_tested": int(len(joined)),
        "gdsc_dependency_ln_ic50_pearson": float(joined["dependency"].corr(ln_sensitivity)),
        "gdsc_dependency_ln_ic50_spearman": float(
            joined["dependency"].corr(ln_sensitivity, method="spearman")
        ),
        "gdsc_dependency_auc_pearson": float(joined["dependency"].corr(auc_sensitivity)),
        "gdsc_dependency_auc_spearman": float(
            joined["dependency"].corr(auc_sensitivity, method="spearman")
        ),
    }


def _release_summary(table: pd.DataFrame) -> dict:
    output: dict[str, dict] = {}
    for release, subset in table.groupby("gdsc_scope", sort=True):
        concordant = subset["direction_concordant_ln_ic50"].dropna().astype(bool)
        effect_mask = subset[
            ["prism_dependency_sensitivity_correlation", "gdsc_dependency_ln_ic50_pearson"]
        ].notna().all(axis=1)
        effect_correlation = None
        if effect_mask.sum() >= 3:
            effect_correlation = _finite_float(
                subset.loc[effect_mask, "prism_dependency_sensitivity_correlation"].corr(
                    subset.loc[effect_mask, "gdsc_dependency_ln_ic50_pearson"]
                )
            )
        output[str(release)] = {
            "evaluated_records": int(len(subset)),
            "unique_prism_pairs": int(
                subset[["treatment_id", "target"]].drop_duplicates().shape[0]
            ),
            "median_models_tested": _finite_float(subset["models_tested"].median()),
            "direction_concordant_records": int(concordant.sum()),
            "direction_evaluable_records": int(len(concordant)),
            "direction_concordance_rate": _finite_float(concordant.mean()),
            "gdsc_target_annotation_supported_records": int(
                subset["gdsc_target_annotation_support"].sum()
            ),
            "prism_gdsc_effect_size_correlation": effect_correlation,
        }
    return output


def run_gdsc_external_replication(
    processed_dir: Path,
    results_dir: Path,
    *,
    min_models: int = 20,
) -> dict:
    """Replicate real PRISM/CRISPR associations in official GDSC observations."""
    gdsc_path = processed_dir / "gdsc_fitted" / "gdsc_fitted_combined.parquet"
    depmap_dir = processed_dir / "depmap_figshare"
    crispr_path = depmap_dir / "depmap_crispr_gene_effect_top_genes.parquet"
    metadata_path = depmap_dir / "depmap_model_metadata.parquet"
    prism_path = (
        results_dir / "rule_collapse" / "depmap_prism_dependency_pharmacology_inversion_all.csv"
    )
    missing = [
        str(path)
        for path in (gdsc_path, crispr_path, metadata_path, prism_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing GDSC replication inputs: {missing}")

    gdsc = pd.read_parquet(gdsc_path)
    crispr = pd.read_parquet(crispr_path).set_index("ModelID")
    crispr.index = crispr.index.astype(str)
    dependency = -crispr.apply(pd.to_numeric, errors="coerce")
    metadata = pd.read_parquet(metadata_path)
    required_metadata = {"ModelID", "SangerModelID"}
    if not required_metadata.issubset(metadata.columns):
        raise ValueError(f"DepMap metadata missing columns: {sorted(required_metadata - set(metadata))}")

    model_map = metadata[["ModelID", "SangerModelID"]].dropna().copy()
    model_map["ModelID"] = model_map["ModelID"].astype(str)
    model_map["SANGER_MODEL_ID"] = model_map["SangerModelID"].astype(str).str.strip()
    model_map = model_map[model_map["SANGER_MODEL_ID"].ne("")].drop_duplicates(
        "SANGER_MODEL_ID", keep=False
    )[["ModelID", "SANGER_MODEL_ID"]]

    required_gdsc = {
        "GDSC_RELEASE",
        "SANGER_MODEL_ID",
        "DRUG_NAME",
        "PUTATIVE_TARGET",
        "LN_IC50",
        "AUC",
    }
    if not required_gdsc.issubset(gdsc.columns):
        raise ValueError(f"GDSC data missing columns: {sorted(required_gdsc - set(gdsc))}")
    gdsc["SANGER_MODEL_ID"] = gdsc["SANGER_MODEL_ID"].astype(str).str.strip()
    gdsc["DRUG_KEY"] = gdsc["DRUG_NAME"].map(_drug_key)
    gdsc["LN_IC50"] = pd.to_numeric(gdsc["LN_IC50"], errors="coerce")
    gdsc["AUC"] = pd.to_numeric(gdsc["AUC"], errors="coerce")
    gdsc = gdsc.merge(model_map, on="SANGER_MODEL_ID", how="inner", validate="many_to_one")
    gdsc = gdsc[gdsc["ModelID"].isin(dependency.index)].copy()

    prism = pd.read_csv(prism_path, low_memory=False)
    required_prism = {
        "target",
        "treatment_id",
        "drug_name",
        "dependency_sensitivity_correlation",
    }
    if not required_prism.issubset(prism.columns):
        raise ValueError(f"PRISM results missing columns: {sorted(required_prism - set(prism))}")
    prism["DRUG_KEY"] = prism["drug_name"].map(_drug_key)
    prism["target"] = prism["target"].astype(str).str.strip()
    prism = prism.drop_duplicates(["treatment_id", "target"])
    gdsc_drugs = set(gdsc["DRUG_KEY"]) - {""}
    eligible = prism[
        prism["DRUG_KEY"].isin(gdsc_drugs) & prism["target"].isin(dependency.columns)
    ].copy()

    drug_groups = {key: group for key, group in gdsc.groupby("DRUG_KEY", sort=False)}
    rows: list[dict] = []
    for pair in eligible.itertuples(index=False):
        drug_observations = drug_groups[pair.DRUG_KEY]
        target_supported = _target_annotation_support(
            drug_observations["PUTATIVE_TARGET"], pair.target
        )
        scopes = [(str(name), group) for name, group in drug_observations.groupby("GDSC_RELEASE")]
        scopes.append(("POOLED", drug_observations))
        prism_correlation = float(pair.dependency_sensitivity_correlation)
        for scope, observations in scopes:
            association = _association(observations, dependency[pair.target], min_models)
            if association is None:
                continue
            ln_concordant = _direction_concordant(
                prism_correlation, association["gdsc_dependency_ln_ic50_pearson"]
            )
            auc_concordant = _direction_concordant(
                prism_correlation, association["gdsc_dependency_auc_pearson"]
            )
            rows.append(
                {
                    "gdsc_scope": scope,
                    "target": pair.target,
                    "treatment_id": str(pair.treatment_id),
                    "drug_name_prism": pair.drug_name,
                    "drug_name_gdsc": str(observations["DRUG_NAME"].iloc[0]),
                    "prism_models_tested": int(pair.models_tested),
                    "prism_dependency_sensitivity_correlation": prism_correlation,
                    **association,
                    "direction_concordant_ln_ic50": ln_concordant,
                    "direction_concordant_auc": auc_concordant,
                    "gdsc_target_annotation_support": target_supported,
                }
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            ["gdsc_scope", "direction_concordant_ln_ic50", "models_tested"],
            ascending=[True, False, False],
        )

    out_dir = results_dir / "validation" / "gdsc_external_replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "prism_crispr_gdsc_replication.csv"
    results.to_csv(results_path, index=False)
    summary = {
        "status": "created",
        "data_policy": "observed public data only; no simulated biological samples",
        "gdsc_source": "Wellcome Sanger Institute Cell Model Passports GDSC1/GDSC2",
        "gdsc_rows_total": int(pd.read_parquet(gdsc_path, columns=["GDSC_RELEASE"]).shape[0]),
        "gdsc_rows_mapped_to_depmap": int(len(gdsc)),
        "gdsc_models_mapped_to_depmap": int(gdsc["ModelID"].nunique()),
        "gdsc_drugs_after_mapping": int(gdsc["DRUG_KEY"].nunique()),
        "prism_pairs_total": int(len(prism)),
        "prism_pairs_with_matching_gdsc_drug_and_crispr_target": int(len(eligible)),
        "minimum_models_per_test": int(min_models),
        "evaluated_records": int(len(results)),
        "evaluated_unique_prism_pairs": int(
            results[["treatment_id", "target"]].drop_duplicates().shape[0]
        )
        if not results.empty
        else 0,
        "by_scope": _release_summary(results) if not results.empty else {},
        "output": str(results_path),
        "interpretation": (
            "Cross-dataset effect-direction comparison using real DepMap CRISPR, PRISM and "
            "official GDSC fitted-response observations. Direction concordance is external "
            "replication evidence, not causal or clinical validation."
        ),
    }
    summary_path = out_dir / "gdsc_external_replication_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
