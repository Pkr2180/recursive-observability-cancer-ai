from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.common.paths import safe_dataset_dir


MODEL_FILE = "Model.csv"
EXPRESSION_FILE = "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
CRISPR_EFFECT_FILE = "CRISPRGeneEffect.csv"


def _read_depmap_matrix(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    first_col = table.columns[0]
    if first_col != "ModelID":
        table = table.rename(columns={first_col: "ModelID"})
    table["ModelID"] = table["ModelID"].astype(str)
    return table


def _clean_gene_name(column: str) -> str:
    return re.sub(r"\s+\(\d+\)$", "", str(column)).strip()


def _rename_gene_columns(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rename = {}
    seen = set()
    for column in table.columns:
        if column == "ModelID":
            continue
        clean = _clean_gene_name(column)
        if not clean or clean in seen:
            clean = str(column)
        rename[column] = clean
        seen.add(clean)
    return table.rename(columns=rename), rename


def _top_variable_columns(table: pd.DataFrame, candidate_columns: list[str], top_n: int) -> list[str]:
    values = table[candidate_columns].apply(pd.to_numeric, errors="coerce")
    variances = values.var(axis=0).sort_values(ascending=False)
    return list(variances.head(top_n).index)


def harmonise_depmap_core(
    raw_dir: Path,
    processed_dir: Path,
    top_n: int = 1000,
) -> dict:
    """Align DepMap model metadata, expression and CRISPR gene-effect matrices."""
    source_raw = safe_dataset_dir(raw_dir, "depmap_figshare") / "figshare_files"
    source_processed = safe_dataset_dir(processed_dir, "depmap_figshare")

    model_path = source_raw / MODEL_FILE
    expression_path = source_raw / EXPRESSION_FILE
    crispr_path = source_raw / CRISPR_EFFECT_FILE
    missing = [
        str(path)
        for path in [model_path, expression_path, crispr_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing DepMap files: {missing}")

    model = pd.read_csv(model_path, low_memory=False)
    expression, expression_gene_map = _rename_gene_columns(_read_depmap_matrix(expression_path))
    crispr, crispr_gene_map = _rename_gene_columns(_read_depmap_matrix(crispr_path))

    common_models = sorted(set(expression["ModelID"]) & set(crispr["ModelID"]))
    expression = expression[expression["ModelID"].isin(common_models)].set_index("ModelID")
    crispr = crispr[crispr["ModelID"].isin(common_models)].set_index("ModelID")

    common_genes = sorted(set(expression.columns) & set(crispr.columns))
    if not common_genes:
        raise ValueError("No overlapping gene columns after DepMap harmonisation.")

    expression_top = set(_top_variable_columns(expression, common_genes, top_n))
    crispr_top = set(_top_variable_columns(crispr, common_genes, top_n))
    selected_genes = sorted((expression_top | crispr_top) & set(common_genes))

    expression_out = expression[selected_genes].reset_index()
    crispr_out = crispr[selected_genes].reset_index()

    model_id_column = "ModelID" if "ModelID" in model.columns else model.columns[0]
    model_out = model[model[model_id_column].astype(str).isin(common_models)].copy()

    expression_out_path = source_processed / "depmap_expression_top_genes.parquet"
    crispr_out_path = source_processed / "depmap_crispr_gene_effect_top_genes.parquet"
    model_out_path = source_processed / "depmap_model_metadata.parquet"
    gene_map_path = source_processed / "depmap_gene_column_maps.json"

    expression_out.to_parquet(expression_out_path, index=False)
    crispr_out.to_parquet(crispr_out_path, index=False)
    model_out.to_parquet(model_out_path, index=False)
    gene_map_path.write_text(
        json.dumps(
            {
                "expression": expression_gene_map,
                "crispr_gene_effect": crispr_gene_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "common_models": len(common_models),
        "common_genes": len(common_genes),
        "selected_genes": len(selected_genes),
        "top_n_requested": top_n,
        "expression": str(expression_out_path),
        "crispr_gene_effect": str(crispr_out_path),
        "model_metadata": str(model_out_path),
        "gene_column_maps": str(gene_map_path),
    }
    (source_processed / "depmap_harmonisation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary

