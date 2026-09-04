from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.paths import safe_dataset_dir


def _read_release(path: Path, release: str) -> pd.DataFrame:
    table = pd.read_excel(path, engine="openpyxl")
    table.columns = [str(column).strip().upper() for column in table.columns]
    table.insert(0, "GDSC_RELEASE", release)
    return table


def harmonise_gdsc_releases(raw_dir: Path, processed_dir: Path) -> dict:
    source_raw = safe_dataset_dir(raw_dir, "gdsc_fitted")
    source_processed = safe_dataset_dir(processed_dir, "gdsc_fitted")
    release_paths = {
        "GDSC1": source_raw / "GDSC1_fitted_dose_response_27Oct23.xlsx",
        "GDSC2": source_raw / "GDSC2_fitted_dose_response_27Oct23.xlsx",
    }
    missing = [str(path) for path in release_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing official GDSC workbooks: {missing}")
    tables = [_read_release(path, release) for release, path in release_paths.items()]
    combined = pd.concat(tables, ignore_index=True, sort=False)
    required = {"CELL_LINE_NAME", "DRUG_NAME", "LN_IC50", "AUC"}
    missing_columns = required - set(combined.columns)
    if missing_columns:
        raise ValueError(f"GDSC fitted data missing columns: {sorted(missing_columns)}")
    for column in ("LN_IC50", "AUC", "Z_SCORE", "COSMIC_ID"):
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined["DRUG_NAME_NORMALIZED"] = combined["DRUG_NAME"].astype(str).str.upper().str.strip()
    output = source_processed / "gdsc_fitted_combined.parquet"
    combined.to_parquet(output, index=False)
    summary = {
        "status": "created",
        "source_authority": "Wellcome Sanger Institute Cell Model Passports",
        "rows": len(combined),
        "releases": combined["GDSC_RELEASE"].value_counts().to_dict(),
        "cell_lines": int(combined["CELL_LINE_NAME"].nunique()),
        "drugs": int(combined["DRUG_NAME_NORMALIZED"].nunique()),
        "columns": list(combined.columns),
        "output": str(output),
    }
    summary_path = source_processed / "gdsc_harmonisation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
