from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _quantise_numeric_column(series: pd.Series, bins: int = 5) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.notna().sum() < 10 or clean.nunique(dropna=True) < 2:
        return pd.Series(["Q_MISSING_OR_CONSTANT"] * len(series), index=series.index)
    try:
        labels = [f"Q{idx}" for idx in range(bins)]
        return pd.qcut(clean, q=bins, labels=labels, duplicates="drop").astype("string")
    except ValueError:
        edges = np.linspace(float(clean.min()), float(clean.max()), bins + 1)
        return pd.cut(clean, bins=edges, include_lowest=True).astype("string")


def quantise_processed_tables(processed_dir: Path, results_dir: Path) -> dict:
    """Create first-pass quantised state-token summaries from processed profiles."""
    token_dir = results_dir / "state_tokens"
    token_dir.mkdir(parents=True, exist_ok=True)

    manifests = list(processed_dir.rglob("profile.json"))
    state_rows = []
    for manifest in manifests:
        profile = json.loads(manifest.read_text(encoding="utf-8"))
        for table in profile.get("tables", []):
            if table.get("status") != "profiled":
                continue
            state_rows.append(
                {
                    "source_key": profile.get("source_key"),
                    "table_path": table.get("path"),
                    "rows": table.get("rows"),
                    "columns": table.get("columns"),
                    "state_token": f"PAN_STATE_{len(state_rows) + 1:05d}",
                    "resolution": "profile_level",
                }
            )

    state_table = pd.DataFrame(state_rows)
    out_path = token_dir / "profile_level_state_tokens.parquet"
    state_table.to_parquet(out_path, index=False)

    expression_result = _quantise_gdc_expression(processed_dir, token_dir)
    depmap_result = _quantise_depmap_core(processed_dir, token_dir)
    prism_result = _quantise_prism_core(processed_dir, token_dir)
    summary = {
        "token_count": int(len(state_table)),
        "token_table": str(out_path),
        "expression_quantisation": expression_result,
        "depmap_quantisation": depmap_result,
        "prism_quantisation": prism_result,
        "note": "Profile-level tokens are a bootstrap layer; omic-level tokenisation follows after full dataset harmonisation.",
    }
    (token_dir / "quantisation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _quantise_gdc_expression(processed_dir: Path, token_dir: Path) -> dict:
    matrices = list(processed_dir.rglob("gdc_rnaseq_tpm_matrix.parquet"))
    if not matrices:
        return {"status": "skipped", "reason": "no harmonised GDC TPM matrices found"}

    token_tables = []
    for matrix_path in matrices:
        matrix = pd.read_parquet(matrix_path)
        if "gene_id" not in matrix.columns or matrix.shape[1] < 3:
            continue
        sample_cols = [col for col in matrix.columns if col != "gene_id"]
        numeric = matrix[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        log_tpm = np.log1p(numeric)
        variances = log_tpm.var(axis=1).sort_values(ascending=False)
        top_indices = variances.head(500).index
        subset = matrix.loc[top_indices, ["gene_id"]].copy()
        subset = pd.concat([subset, log_tpm.loc[top_indices]], axis=1)

        long_rows = []
        for sample_col in sample_cols:
            quantised = _quantise_numeric_column(subset[sample_col], bins=5)
            for gene_id, token in zip(subset["gene_id"], quantised, strict=False):
                long_rows.append(
                    {
                        "source_matrix": str(matrix_path),
                        "sample_id": sample_col,
                        "gene_id": gene_id,
                        "value_space": "log1p_tpm",
                        "state_token": f"RNA_{token}",
                    }
                )
        token_tables.append(pd.DataFrame(long_rows))

    if not token_tables:
        return {"status": "skipped", "reason": "no quantisable expression matrices found"}

    expression_tokens = pd.concat(token_tables, ignore_index=True)
    out_path = token_dir / "gdc_expression_state_tokens.parquet"
    expression_tokens.to_parquet(out_path, index=False)
    return {
        "status": "created",
        "token_rows": int(len(expression_tokens)),
        "unique_genes": int(expression_tokens["gene_id"].nunique()),
        "unique_samples": int(expression_tokens["sample_id"].nunique()),
        "path": str(out_path),
    }


def _quantise_depmap_matrix(matrix_path: Path, token_dir: Path, prefix: str) -> dict:
    matrix = pd.read_parquet(matrix_path)
    if "ModelID" not in matrix.columns or matrix.shape[1] < 3:
        return {"status": "skipped", "reason": f"not a DepMap matrix: {matrix_path}"}

    genes = [column for column in matrix.columns if column != "ModelID"]
    quantised = pd.DataFrame({"ModelID": matrix["ModelID"].astype(str)})
    for gene in genes:
        quantised[gene] = _quantise_numeric_column(matrix[gene], bins=5).map(
            lambda value: f"{prefix}_{value}"
        )

    long = quantised.melt(
        id_vars=["ModelID"],
        var_name="gene",
        value_name="state_token",
    )
    long = long.rename(columns={"ModelID": "model_id"})
    out_path = token_dir / f"depmap_{prefix.lower()}_state_tokens.parquet"
    long.to_parquet(out_path, index=False)
    return {
        "status": "created",
        "matrix": str(matrix_path),
        "path": str(out_path),
        "token_rows": int(len(long)),
        "unique_models": int(long["model_id"].nunique()),
        "unique_genes": int(long["gene"].nunique()),
    }


def _quantise_depmap_core(processed_dir: Path, token_dir: Path) -> dict:
    source_dir = processed_dir / "depmap_figshare"
    expression_path = source_dir / "depmap_expression_top_genes.parquet"
    crispr_path = source_dir / "depmap_crispr_gene_effect_top_genes.parquet"

    results = {}
    if expression_path.exists():
        results["expression"] = _quantise_depmap_matrix(expression_path, token_dir, "DM_EXPR")
    else:
        results["expression"] = {
            "status": "skipped",
            "reason": "missing depmap_expression_top_genes.parquet",
        }

    if crispr_path.exists():
        results["crispr_gene_effect"] = _quantise_depmap_matrix(
            crispr_path, token_dir, "DM_CRISPR_EFFECT"
        )
    else:
        results["crispr_gene_effect"] = {
            "status": "skipped",
            "reason": "missing depmap_crispr_gene_effect_top_genes.parquet",
        }
    return results


def _quantise_prism_core(processed_dir: Path, token_dir: Path) -> dict:
    matrix_path = processed_dir / "prism_repurposing_24q2" / "prism_primary_top_treatments.parquet"
    if not matrix_path.exists():
        return {"status": "skipped", "reason": "missing prism_primary_top_treatments.parquet"}

    matrix = pd.read_parquet(matrix_path)
    if "ModelID" not in matrix.columns or matrix.shape[1] < 3:
        return {"status": "skipped", "reason": f"not a PRISM matrix: {matrix_path}"}

    treatments = [column for column in matrix.columns if column != "ModelID"]
    quantised = pd.DataFrame({"ModelID": matrix["ModelID"].astype(str)})
    for treatment in treatments:
        quantised[treatment] = _quantise_numeric_column(matrix[treatment], bins=5).map(
            lambda value: f"PRISM_LFC_{value}"
        )

    long = quantised.melt(
        id_vars=["ModelID"],
        var_name="treatment_id",
        value_name="state_token",
    ).rename(columns={"ModelID": "model_id"})
    out_path = token_dir / "prism_drug_response_state_tokens.parquet"
    long.to_parquet(out_path, index=False)
    return {
        "status": "created",
        "matrix": str(matrix_path),
        "path": str(out_path),
        "token_rows": int(len(long)),
        "unique_models": int(long["model_id"].nunique()),
        "unique_treatments": int(long["treatment_id"].nunique()),
    }
