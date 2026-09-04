from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _zscore(table: pd.DataFrame) -> pd.DataFrame:
    values = table.apply(pd.to_numeric, errors="coerce")
    return (values - values.mean(axis=0)) / values.std(axis=0).replace(0, np.nan)


def mine_expression_dependency_collapse(
    processed_dir: Path,
    results_dir: Path,
    top_n: int = 200,
) -> dict:
    """Find first-pass expression-dependency mismatch signals in DepMap."""
    source_dir = processed_dir / "depmap_figshare"
    expression_path = source_dir / "depmap_expression_top_genes.parquet"
    crispr_path = source_dir / "depmap_crispr_gene_effect_top_genes.parquet"
    if not expression_path.exists() or not crispr_path.exists():
        raise FileNotFoundError("DepMap harmonised expression/CRISPR matrices are missing.")

    expression = pd.read_parquet(expression_path).set_index("ModelID")
    crispr_effect = pd.read_parquet(crispr_path).set_index("ModelID")

    common_models = sorted(set(expression.index) & set(crispr_effect.index))
    common_genes = sorted(set(expression.columns) & set(crispr_effect.columns))
    expression = expression.loc[common_models, common_genes]
    dependency = -crispr_effect.loc[common_models, common_genes]

    expr_z = _zscore(expression)
    dep_z = _zscore(dependency)

    rows = []
    for gene in common_genes:
        x = expr_z[gene]
        y = dep_z[gene]
        mask = x.notna() & y.notna()
        if mask.sum() < 20:
            continue
        corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        high_expr_low_dep = int(((x > 1.0) & (y < -1.0)).sum())
        low_expr_high_dep = int(((x < -1.0) & (y > 1.0)).sum())
        concordant_high = int(((x > 1.0) & (y > 1.0)).sum())
        collapse_score = high_expr_low_dep + low_expr_high_dep - concordant_high
        rows.append(
            {
                "gene": gene,
                "models_tested": int(mask.sum()),
                "expression_dependency_correlation": corr,
                "high_expression_low_dependency_models": high_expr_low_dep,
                "low_expression_high_dependency_models": low_expr_high_dep,
                "concordant_high_expression_dependency_models": concordant_high,
                "expression_dependency_collapse_score": int(collapse_score),
            }
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            [
                "expression_dependency_collapse_score",
                "expression_dependency_correlation",
            ],
            ascending=[False, True],
        )

    out_dir = results_dir / "rule_collapse"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "depmap_expression_dependency_collapse_all.csv"
    top_path = out_dir / "depmap_expression_dependency_collapse_top.csv"
    results.to_csv(all_path, index=False)
    results.head(top_n).to_csv(top_path, index=False)

    summary = {
        "status": "created",
        "genes_tested": int(len(results)),
        "models": int(len(common_models)),
        "top_n": int(top_n),
        "all_results": str(all_path),
        "top_results": str(top_path),
        "top_preview": results.head(10).to_dict(orient="records"),
        "interpretation": (
            "First-pass expression-dependency mismatch only. "
            "This is a hypothesis generator, not validation."
        ),
    }
    (out_dir / "depmap_expression_dependency_collapse_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _split_targets(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).replace("|", ";").replace(",", ";")
    targets = []
    for part in text.split(";"):
        target = part.strip()
        if not target or target.lower() in {"nan", "none", "unknown"}:
            continue
        targets.append(target)
    return targets


def mine_dependency_pharmacology_inversion(
    processed_dir: Path,
    results_dir: Path,
    top_n: int = 200,
) -> dict:
    """Mine first-pass CRISPR dependency versus PRISM drug-response inversions."""
    depmap_dir = processed_dir / "depmap_figshare"
    prism_dir = processed_dir / "prism_repurposing_24q2"
    crispr_path = depmap_dir / "depmap_crispr_gene_effect_top_genes.parquet"
    response_path = prism_dir / "prism_primary_top_treatments.parquet"
    compound_path = prism_dir / "prism_compound_metadata.parquet"
    if not crispr_path.exists() or not response_path.exists() or not compound_path.exists():
        raise FileNotFoundError("Missing CRISPR, PRISM response, or PRISM compound metadata.")

    crispr_effect = pd.read_parquet(crispr_path).set_index("ModelID")
    dependency = -crispr_effect
    response = pd.read_parquet(response_path).set_index("ModelID")
    sensitivity = -response.apply(pd.to_numeric, errors="coerce")
    compounds = pd.read_parquet(compound_path)

    if "IDs" not in compounds.columns or "repurposing_target" not in compounds.columns:
        raise ValueError("PRISM compound metadata missing IDs/repurposing_target columns.")

    compound_rows = []
    for _, row in compounds.iterrows():
        row_dict = row.to_dict()
        treatment_id = str(row_dict.get("IDs"))
        drug_name = row_dict.get("Drug_Name", row_dict.get("Drug.Name", ""))
        moa = row_dict.get("MOA", "")
        for target in _split_targets(row_dict.get("repurposing_target")):
            compound_rows.append(
                {
                    "treatment_id": treatment_id,
                    "drug_name": drug_name,
                    "target": target,
                    "moa": moa,
                }
            )
    compound_map = pd.DataFrame(compound_rows)
    if compound_map.empty:
        return {"status": "skipped", "reason": "no target annotations in PRISM metadata"}

    common_models = sorted(set(dependency.index) & set(sensitivity.index))
    dependency = dependency.loc[common_models]
    sensitivity = sensitivity.loc[common_models]

    rows = []
    for item in compound_map.itertuples(index=False):
        treatment_id = item.treatment_id
        target = item.target
        if treatment_id not in sensitivity.columns or target not in dependency.columns:
            continue
        dep = dependency[target]
        sens = sensitivity[treatment_id]
        mask = dep.notna() & sens.notna()
        if mask.sum() < 20:
            continue
        if dep[mask].std() == 0 or sens[mask].std() == 0:
            continue
        dep_z = (dep[mask] - dep[mask].mean()) / dep[mask].std()
        sens_z = (sens[mask] - sens[mask].mean()) / sens[mask].std()
        corr = float(np.corrcoef(dep_z, sens_z)[0, 1])
        high_dep_low_sens = int(((dep_z > 1.0) & (sens_z < -1.0)).sum())
        low_dep_high_sens = int(((dep_z < -1.0) & (sens_z > 1.0)).sum())
        concordant_high = int(((dep_z > 1.0) & (sens_z > 1.0)).sum())
        inversion_score = high_dep_low_sens + low_dep_high_sens - concordant_high
        rows.append(
            {
                "target": target,
                "treatment_id": treatment_id,
                "drug_name": item.drug_name,
                "moa": item.moa,
                "models_tested": int(mask.sum()),
                "dependency_sensitivity_correlation": corr,
                "high_dependency_low_sensitivity_models": high_dep_low_sens,
                "low_dependency_high_sensitivity_models": low_dep_high_sens,
                "concordant_high_dependency_sensitivity_models": concordant_high,
                "dependency_pharmacology_inversion_score": int(inversion_score),
            }
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            [
                "dependency_pharmacology_inversion_score",
                "dependency_sensitivity_correlation",
            ],
            ascending=[False, True],
        )

    out_dir = results_dir / "rule_collapse"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "depmap_prism_dependency_pharmacology_inversion_all.csv"
    top_path = out_dir / "depmap_prism_dependency_pharmacology_inversion_top.csv"
    results.to_csv(all_path, index=False)
    results.head(top_n).to_csv(top_path, index=False)

    summary = {
        "status": "created",
        "pairs_tested": int(len(results)),
        "models": int(len(common_models)),
        "top_n": int(top_n),
        "all_results": str(all_path),
        "top_results": str(top_path),
        "top_preview": results.head(10).to_dict(orient="records"),
        "interpretation": (
            "First-pass dependency-pharmacology inversion candidates only. "
            "Requires drug-target curation and external validation."
        ),
    }
    (out_dir / "depmap_prism_dependency_pharmacology_inversion_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
