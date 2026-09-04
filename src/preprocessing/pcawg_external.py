from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HORIZON_DAYS = 730
MIN_GENE_COVERAGE = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_ensembl(identifier: str) -> str:
    return str(identifier).strip().split(".", 1)[0]


def _fixed_horizon(status: pd.Series, time_days: pd.Series) -> pd.Series:
    status = status.astype("string").str.lower()
    time_days = pd.to_numeric(time_days, errors="coerce")
    result = pd.Series(np.nan, index=status.index, dtype=float)
    result[(status == "deceased") & (time_days <= HORIZON_DAYS)] = 1.0
    result[time_days >= HORIZON_DAYS] = 0.0
    return result


def harmonise_pcawg_external(
    raw_dir: Path,
    processed_dir: Path,
    model_dir: Path,
) -> dict:
    """Create a non-TCGA PCAWG cohort compatible with the frozen TCGA model."""
    source_dir = raw_dir / "pcawg_external"
    model_path = (
        model_dir / "tcga_outcome_observability" / "tcga_os2y_frozen_bundle.joblib"
    )
    required = [
        source_dir / "survival_donor.tsv",
        source_dir / "project_code_donor.tsv",
        source_dir / "donor_clinical.tsv",
        source_dir / "gene_expression_fpkm_uq_log.tsv",
        model_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PCAWG/model inputs: {missing}")

    bundle = joblib.load(model_path)
    selected_genes = [str(value) for value in bundle["selected_gene_ids"]]
    selected_stable = [_stable_ensembl(value) for value in selected_genes]
    selected_set = set(selected_stable)

    projects = pd.read_csv(source_dir / "project_code_donor.tsv", sep="\t")
    projects = projects.rename(columns={projects.columns[0]: "donor_id"})
    projects["donor_id"] = projects["donor_id"].astype(str)
    projects["dcc_project_code"] = projects["dcc_project_code"].astype("string")
    projects["tcga_derived"] = projects["dcc_project_code"].str.endswith("-US", na=True)
    independent = projects.loc[~projects["tcga_derived"], ["donor_id", "dcc_project_code"]]

    expression_path = source_dir / "gene_expression_fpkm_uq_log.tsv"
    with expression_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        expression_donors = [str(value) for value in header[1:]]
        allowed = set(independent["donor_id"])
        donor_indices = [index for index, donor in enumerate(expression_donors) if donor in allowed]
        retained_donors = [expression_donors[index] for index in donor_indices]
        values_by_gene: dict[str, list[np.ndarray]] = {}
        for row in reader:
            if not row:
                continue
            stable = _stable_ensembl(row[0])
            if stable not in selected_set:
                continue
            selected_values = np.asarray(
                [pd.to_numeric(row[index + 1], errors="coerce") for index in donor_indices],
                dtype=float,
            )
            values_by_gene.setdefault(stable, []).append(selected_values)

    fit_medians = np.asarray(bundle["selected_fit_medians"], dtype=float)
    matrix = np.empty((len(retained_donors), len(selected_stable)), dtype=np.float32)
    observed_gene = np.zeros(len(selected_stable), dtype=bool)
    for gene_index, stable in enumerate(selected_stable):
        rows = values_by_gene.get(stable, [])
        if rows:
            logged_fpkm_uq = np.nanmean(np.vstack(rows), axis=0)
            fpkm_uq = np.maximum(np.exp2(np.clip(logged_fpkm_uq, -20, 40)) - 0.001, 0)
            transformed = np.log2(1 + fpkm_uq)
            transformed[~np.isfinite(transformed)] = fit_medians[gene_index]
            matrix[:, gene_index] = transformed.astype(np.float32)
            observed_gene[gene_index] = True
        else:
            matrix[:, gene_index] = np.float32(fit_medians[gene_index])

    coverage = float(observed_gene.mean())
    if coverage < MIN_GENE_COVERAGE:
        raise ValueError(
            f"PCAWG expression covers {coverage:.3%} of frozen genes; "
            f"minimum is {MIN_GENE_COVERAGE:.1%}"
        )

    expression = pd.DataFrame(matrix, index=retained_donors, columns=selected_stable)
    expression.index.name = "donor_id"
    expression = expression.reset_index()

    survival = pd.read_csv(source_dir / "survival_donor.tsv", sep="\t")
    survival = survival.rename(columns={survival.columns[0]: "donor_id"})
    survival["donor_id"] = survival["donor_id"].astype(str)
    survival["os_time_days"] = pd.to_numeric(survival["_TIME_TO_EVENT"], errors="coerce")
    survival["os_2y_event"] = _fixed_horizon(
        survival["donor_vital_status"], survival["os_time_days"]
    )

    clinical = pd.read_csv(source_dir / "donor_clinical.tsv", sep="\t")
    clinical = clinical.rename(columns={clinical.columns[0]: "donor_id"})
    clinical["donor_id"] = clinical["donor_id"].astype(str)
    keep_clinical = [
        column
        for column in [
            "donor_id",
            "donor_age_at_diagnosis",
            "donor_sex",
            "donor_diagnosis_icd10",
            "first_therapy_type",
            "first_therapy_response",
        ]
        if column in clinical.columns
    ]
    patient_table = (
        independent.merge(survival, on="donor_id", how="inner")
        .merge(clinical[keep_clinical], on="donor_id", how="left")
    )
    patient_table["expression_available"] = patient_table["donor_id"].isin(retained_donors)
    patient_table["independent_non_tcga"] = True

    out_dir = processed_dir / "pcawg_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    expression_out = out_dir / "pcawg_non_tcga_frozen_gene_expression.parquet"
    patients_out = out_dir / "pcawg_non_tcga_patients.parquet"
    genes_out = out_dir / "pcawg_frozen_gene_mapping.csv"
    expression.to_parquet(expression_out, index=False)
    patient_table.to_parquet(patients_out, index=False)
    pd.DataFrame(
        {
            "frozen_gene_id": selected_genes,
            "stable_gene_id": selected_stable,
            "observed_in_pcawg": observed_gene,
            "imputation": np.where(observed_gene, "none", "TCGA_fit_median"),
        }
    ).to_csv(genes_out, index=False)

    eligible = patient_table[
        patient_table["expression_available"] & patient_table["os_2y_event"].notna()
    ]
    summary = {
        "status": "completed",
        "source": "PCAWG donor-centric UCSC Xena open hub",
        "simulation": False,
        "independence_rule": "all -US PCAWG projects excluded as TCGA-derived",
        "model_bundle": str(model_path),
        "model_bundle_sha256": _sha256(model_path),
        "frozen_genes": len(selected_genes),
        "observed_genes": int(observed_gene.sum()),
        "gene_coverage": coverage,
        "independent_expression_donors": len(retained_donors),
        "eligible_two_year_os_patients": int(len(eligible)),
        "eligible_events": int(eligible["os_2y_event"].sum()),
        "eligible_projects": int(eligible["dcc_project_code"].nunique()),
        "project_counts": {
            str(key): int(value)
            for key, value in eligible["dcc_project_code"].value_counts().sort_index().items()
        },
        "artifacts": {
            "expression": str(expression_out),
            "patients": str(patients_out),
            "gene_mapping": str(genes_out),
        },
        "transport_guardrail": (
            "PCAWG FPKM-UQ and TCGA TPM are not identical units. The frozen monotone transform "
            "is label-blind, and residual platform shift must be reported."
        ),
    }
    summary_path = out_dir / "pcawg_external_harmonisation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
