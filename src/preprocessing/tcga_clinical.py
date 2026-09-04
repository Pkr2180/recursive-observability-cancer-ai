from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_KEY = "tcga_cdr"
RNA_SOURCE_KEY = "tcga_gdc_public_rnaseq_index"
HORIZONS_DAYS = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}


def _snake(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _fixed_horizon(event: pd.Series, time: pd.Series, horizon: int) -> pd.Series:
    event_number = pd.to_numeric(event, errors="coerce")
    time_number = pd.to_numeric(time, errors="coerce")
    labels = pd.Series(pd.NA, index=event.index, dtype="Int8")
    labels.loc[(event_number == 1) & (time_number <= horizon)] = 1
    labels.loc[
        ((event_number == 0) & (time_number >= horizon))
        | ((event_number == 1) & (time_number > horizon))
    ] = 0
    return labels


def _endpoint_columns(columns: set[str], prefix: str) -> tuple[str, str] | None:
    event_candidates = [prefix, f"{prefix}_event"]
    time_candidates = [f"{prefix}_time", f"{prefix}_days"]
    event = next((name for name in event_candidates if name in columns), None)
    time = next((name for name in time_candidates if name in columns), None)
    return (event, time) if event and time else None


def _safe_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def harmonise_tcga_cdr(raw_dir: Path, processed_dir: Path) -> dict:
    """Harmonise TCGA-CDR endpoints and link them to downloaded primary RNA samples."""
    source_file = raw_dir / SOURCE_KEY / "TCGA-CDR-SupplementalTableS1.xlsx"
    if not source_file.exists():
        raise FileNotFoundError(f"Missing official TCGA-CDR workbook: {source_file}")

    workbook = pd.ExcelFile(source_file)
    sheets = {
        sheet: pd.read_excel(source_file, sheet_name=sheet)
        for sheet in workbook.sheet_names
    }
    candidate = max(
        sheets,
        key=lambda name: sum(
            str(column).strip().lower() in {"bcr_patient_barcode", "os", "os.time"}
            for column in sheets[name].columns
        ),
    )
    clinical = sheets[candidate].copy()
    clinical.columns = [_snake(column) for column in clinical.columns]
    columns = set(clinical.columns)

    case_column = next(
        (name for name in ("bcr_patient_barcode", "case_submitter_id", "patient") if name in columns),
        None,
    )
    project_column = next(
        (name for name in ("type", "project", "project_id") if name in columns),
        None,
    )
    if case_column is None or project_column is None:
        raise ValueError(
            f"TCGA-CDR sheet {candidate!r} lacks patient/project identifiers; "
            f"columns={sorted(columns)}"
        )

    clinical["case_submitter_id"] = clinical[case_column].map(_safe_text).str.upper()
    project_text = clinical[project_column].map(_safe_text).str.upper()
    clinical["project_id"] = np.where(
        project_text.str.startswith("TCGA-"), project_text, "TCGA-" + project_text
    )
    clinical = clinical[clinical["case_submitter_id"].str.startswith("TCGA-")].copy()
    clinical = clinical.drop_duplicates("case_submitter_id", keep="first")

    endpoint_summary: dict[str, dict[str, Any]] = {}
    for prefix in ("os", "dss", "dfi", "pfi"):
        endpoint = _endpoint_columns(set(clinical.columns), prefix)
        if endpoint is None:
            continue
        event_column, time_column = endpoint
        clinical[event_column] = pd.to_numeric(clinical[event_column], errors="coerce")
        clinical[time_column] = pd.to_numeric(clinical[time_column], errors="coerce")
        endpoint_summary[prefix] = {
            "event_column": event_column,
            "time_column": time_column,
            "events": int((clinical[event_column] == 1).sum()),
            "known_event": int(clinical[event_column].notna().sum()),
            "known_time": int(clinical[time_column].notna().sum()),
        }
        for horizon_name, horizon_days in HORIZONS_DAYS.items():
            label_column = f"{prefix}_{horizon_name}_event"
            clinical[label_column] = _fixed_horizon(
                clinical[event_column], clinical[time_column], horizon_days
            )
            endpoint_summary[prefix][f"eligible_{horizon_name}"] = int(
                clinical[label_column].notna().sum()
            )
            endpoint_summary[prefix][f"events_{horizon_name}"] = int(
                (clinical[label_column] == 1).sum()
            )

    if "os" not in endpoint_summary:
        raise ValueError(f"TCGA-CDR sheet {candidate!r} has no supported OS endpoint")

    rna_meta_path = processed_dir / RNA_SOURCE_KEY / "gdc_sample_metadata.parquet"
    if not rna_meta_path.exists():
        raise FileNotFoundError(f"Missing harmonised GDC RNA metadata: {rna_meta_path}")
    samples = pd.read_parquet(rna_meta_path).copy()
    samples["case_submitter_id"] = (
        samples["case_submitter_ids"].map(_safe_text).str.split(";").str[0].str.upper()
    )
    samples["project_id"] = samples["project_ids"].map(_safe_text).str.split(";").str[0]
    samples["sample_type"] = samples["sample_types"].map(_safe_text)
    samples["is_primary_tumor"] = samples["sample_type"].str.casefold().eq("primary tumor")

    identifier_columns = {
        case_column,
        project_column,
        "case_submitter_id",
        "project_id",
        "bcr_patient_barcode",
        "patient",
        "project",
        "type",
    }
    patient_columns = [
        column for column in clinical.columns if column not in identifier_columns
    ]
    linked = samples.merge(
        clinical[["case_submitter_id", "project_id", *patient_columns]],
        on=["case_submitter_id", "project_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    linked["cdr_linked"] = linked["_merge"].eq("both")
    linked = linked.drop(columns="_merge")

    out_dir = processed_dir / SOURCE_KEY
    out_dir.mkdir(parents=True, exist_ok=True)
    clinical_path = out_dir / "tcga_cdr_patients.parquet"
    linked_path = out_dir / "tcga_cdr_rnaseq_linked.parquet"
    clinical.to_parquet(clinical_path, index=False)
    linked.to_parquet(linked_path, index=False)

    sample_type_counts = linked["sample_type"].value_counts(dropna=False).to_dict()
    linked_primary = linked[linked["is_primary_tumor"] & linked["cdr_linked"]]
    linked_endpoints: dict[str, dict[str, int]] = {}
    for prefix in endpoint_summary:
        endpoint = _endpoint_columns(set(linked.columns), prefix)
        if endpoint is None:
            continue
        linked_endpoints[prefix] = {
            "linked_known_time": int(linked_primary[endpoint[1]].notna().sum()),
            **{
                f"eligible_{horizon_name}": int(
                    linked_primary[f"{prefix}_{horizon_name}_event"].notna().sum()
                )
                for horizon_name in HORIZONS_DAYS
            },
            **{
                f"events_{horizon_name}": int(
                    (linked_primary[f"{prefix}_{horizon_name}_event"] == 1).sum()
                )
                for horizon_name in HORIZONS_DAYS
            },
        }

    summary = {
        "source_policy": "official_open_access_observed_patient_data",
        "simulation": False,
        "workbook": str(source_file),
        "sheet": candidate,
        "sheets": workbook.sheet_names,
        "patients": int(len(clinical)),
        "projects": int(clinical["project_id"].nunique()),
        "rna_samples": int(len(linked)),
        "rna_unique_cases": int(linked["case_submitter_id"].nunique()),
        "cdr_linked_samples": int(linked["cdr_linked"].sum()),
        "primary_tumor_samples": int(linked["is_primary_tumor"].sum()),
        "linked_primary_tumor_samples": int(len(linked_primary)),
        "linked_primary_unique_cases": int(linked_primary["case_submitter_id"].nunique()),
        "sample_types": {str(key): int(value) for key, value in sample_type_counts.items()},
        "endpoint_completeness_all_patients": endpoint_summary,
        "endpoint_completeness_linked_primary": linked_endpoints,
        "artifacts": {
            "patients": str(clinical_path),
            "rna_linked": str(linked_path),
        },
        "interpretation": (
            "TCGA-CDR supplies real patient outcome/follow-up endpoints linked to baseline RNA-seq. "
            "It does not supply repeated longitudinal molecular measurements."
        ),
    }
    summary_path = out_dir / "tcga_cdr_harmonisation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**summary, "summary": str(summary_path)}
