from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.paths import safe_dataset_dir
from src.downloaders.lincs import METADATA_FILES


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)


def _column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    return next((name for name in candidates if name in table.columns), "")


def harmonise_lincs_phase2_metadata(raw_dir: Path, processed_dir: Path) -> dict:
    """Harmonise real LINCS perturbation metadata and identify cancer-cell signatures."""
    source_raw = safe_dataset_dir(raw_dir, "lincs_gse70138")
    source_processed = safe_dataset_dir(processed_dir, "lincs_gse70138")
    paths = {filename: source_raw / filename for filename in METADATA_FILES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing LINCS metadata: {missing}")

    tables = {filename: _read(path) for filename, path in paths.items()}
    cell = tables[next(name for name in tables if "cell_info" in name)]
    gene = tables[next(name for name in tables if "gene_info" in name)]
    instance = tables[next(name for name in tables if "inst_info" in name)]
    perturbation = tables[next(name for name in tables if "pert_info" in name)]
    signature = tables[next(name for name in tables if "sig_info" in name)]
    metrics = tables[next(name for name in tables if "sig_metrics" in name)]

    cell_id = _column(cell, ("cell_id", "cell_name"))
    sample_type = _column(cell, ("sample_type", "cell_type"))
    sig_cell_id = _column(signature, ("cell_id", "cell_name"))
    sig_id = _column(signature, ("sig_id", "signature_id"))
    if not cell_id or not sig_cell_id or not sig_id:
        raise ValueError("LINCS metadata lacks cell/signature identifiers")

    cancer_mask = pd.Series(False, index=cell.index)
    for candidate in ("sample_type", "cell_type", "primary_site", "subtype"):
        if candidate in cell.columns:
            cancer_mask |= cell[candidate].fillna("").astype(str).str.contains(
                r"tumou?r|cancer|carcinoma|sarcoma|leuk|lymph|melanoma|myeloma",
                case=False,
                regex=True,
            )
    cancer_cells = set(cell.loc[cancer_mask, cell_id].dropna().astype(str))
    cancer_signatures = signature[signature[sig_cell_id].astype(str).isin(cancer_cells)].copy()
    if cancer_signatures.empty:
        raise ValueError(
            f"No cancer signatures identified; sample type values: "
            f"{cell[sample_type].value_counts().head(20).to_dict() if sample_type else {}}"
        )

    outputs = {
        "cell_info": source_processed / "lincs_cell_info.parquet",
        "gene_info": source_processed / "lincs_gene_info.parquet",
        "instance_info": source_processed / "lincs_instance_info.parquet",
        "perturbation_info": source_processed / "lincs_perturbation_info.parquet",
        "signature_info": source_processed / "lincs_signature_info.parquet",
        "signature_metrics": source_processed / "lincs_signature_metrics.parquet",
        "cancer_signature_info": source_processed / "lincs_cancer_signature_info.parquet",
    }
    cell.to_parquet(outputs["cell_info"], index=False)
    gene.to_parquet(outputs["gene_info"], index=False)
    instance.to_parquet(outputs["instance_info"], index=False)
    perturbation.to_parquet(outputs["perturbation_info"], index=False)
    signature.to_parquet(outputs["signature_info"], index=False)
    metrics.to_parquet(outputs["signature_metrics"], index=False)
    cancer_signatures.to_parquet(outputs["cancer_signature_info"], index=False)

    pert_type = _column(signature, ("pert_type", "perturbation_type"))
    time_column = _column(signature, ("pert_time", "pert_itime"))
    summary = {
        "status": "created",
        "source_authority": "NCBI GEO GSE70138 / NIH LINCS",
        "observed_data_only": True,
        "cells": int(cell[cell_id].nunique()),
        "cancer_cells": int(len(cancer_cells)),
        "genes": int(len(gene)),
        "instances": int(len(instance)),
        "perturbations": int(len(perturbation)),
        "signatures": int(len(signature)),
        "cancer_signatures": int(len(cancer_signatures)),
        "perturbation_types": signature[pert_type].value_counts().to_dict()
        if pert_type
        else {},
        "time_points": signature[time_column].value_counts().sort_index().to_dict()
        if time_column
        else {},
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    summary_path = source_processed / "lincs_metadata_harmonisation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
