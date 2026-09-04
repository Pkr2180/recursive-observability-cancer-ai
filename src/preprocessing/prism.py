from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.paths import safe_dataset_dir


PRISM_MATRIX_FILE = "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv"
PRISM_CELL_META_FILE = "Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv"
PRISM_COMPOUND_FILE = "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv"
PRISM_TREATMENT_META_FILE = "Repurposing_Public_24Q2_Treatment_Meta_Data.csv"


def _standardize_id_column(table: pd.DataFrame, preferred_name: str) -> pd.DataFrame:
    candidates = ["ModelID", "DepMap_ID", "depmap_id", "Broad_ID", "broad_id", "cell_line"]
    for candidate in candidates:
        if candidate in table.columns:
            return table.rename(columns={candidate: preferred_name})
    return table.rename(columns={table.columns[0]: preferred_name})


def harmonise_prism_primary(
    raw_dir: Path,
    processed_dir: Path,
    top_n: int = 1000,
) -> dict:
    """Harmonise PRISM Repurposing 24Q2 primary response matrix."""
    source_raw = safe_dataset_dir(raw_dir, "depmap_figshare") / "figshare_files"
    source_processed = safe_dataset_dir(processed_dir, "prism_repurposing_24q2")

    matrix_path = source_raw / PRISM_MATRIX_FILE
    cell_meta_path = source_raw / PRISM_CELL_META_FILE
    compound_path = source_raw / PRISM_COMPOUND_FILE
    treatment_meta_path = source_raw / PRISM_TREATMENT_META_FILE
    missing = [
        str(path)
        for path in [matrix_path, cell_meta_path, compound_path, treatment_meta_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing PRISM files: {missing}")

    raw_matrix = pd.read_csv(matrix_path, low_memory=False)
    first_col = raw_matrix.columns[0]
    first_values = raw_matrix[first_col].astype(str).head(50)
    treatment_like_rows = (
        first_values.str.startswith("BRD").mean() > 0.5
        or "treatment" in first_col.lower()
        or "compound" in first_col.lower()
    )

    if treatment_like_rows:
        matrix = raw_matrix.set_index(first_col).T.reset_index()
        matrix = matrix.rename(columns={"index": "ModelID"})
        orientation = "transposed_treatment_rows_to_model_rows"
    else:
        matrix = _standardize_id_column(raw_matrix, "ModelID")
        orientation = "model_rows"

    numeric_columns = [column for column in matrix.columns if column != "ModelID"]
    numeric = matrix[numeric_columns].apply(pd.to_numeric, errors="coerce")
    top_columns = list(numeric.var(axis=0).sort_values(ascending=False).head(top_n).index)
    response_out = pd.concat([matrix[["ModelID"]], numeric[top_columns]], axis=1)

    cell_meta = pd.read_csv(cell_meta_path, low_memory=False)
    compound = pd.read_csv(compound_path, low_memory=False)
    treatment_meta = pd.read_csv(treatment_meta_path, low_memory=False)

    response_path = source_processed / "prism_primary_top_treatments.parquet"
    cell_meta_out = source_processed / "prism_cell_line_metadata.parquet"
    compound_out = source_processed / "prism_compound_metadata.parquet"
    treatment_meta_out = source_processed / "prism_treatment_metadata.parquet"

    response_out.to_parquet(response_path, index=False)
    cell_meta.to_parquet(cell_meta_out, index=False)
    compound.to_parquet(compound_out, index=False)
    treatment_meta.to_parquet(treatment_meta_out, index=False)

    summary = {
        "models": int(response_out.shape[0]),
        "selected_treatments": int(len(top_columns)),
        "top_n_requested": int(top_n),
        "response_matrix": str(response_path),
        "cell_metadata": str(cell_meta_out),
        "compound_metadata": str(compound_out),
        "treatment_metadata": str(treatment_meta_out),
        "response_id_column": "ModelID",
        "orientation": orientation,
    }
    (source_processed / "prism_harmonisation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
