from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.paths import safe_dataset_dir


TPM_COLUMN = "tpm_unstranded"
COUNT_COLUMN = "unstranded"


def _read_star_counts(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    required = {"gene_id", "gene_name", TPM_COLUMN, COUNT_COLUMN}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing STAR count columns in {path.name}: {sorted(missing)}")
    table = table[~table["gene_id"].astype(str).str.startswith("N_")].copy()
    table["gene_id"] = table["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    return table


def harmonise_gdc_rnaseq(
    raw_dir: Path,
    processed_dir: Path,
    source_key: str,
    max_files: int | None = None,
) -> dict:
    """Convert downloaded GDC STAR gene-count files into matrices."""
    source_raw = safe_dataset_dir(raw_dir, source_key)
    files_dir = source_raw / "files"
    if not files_dir.exists():
        raise FileNotFoundError(f"Missing GDC files directory: {files_dir}")

    source_processed = safe_dataset_dir(processed_dir, source_key)
    index_path = source_raw / "gdc_file_index.parquet"
    index = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()

    download_manifest_path = source_raw / "gdc_download_manifest.json"
    if download_manifest_path.exists():
        manifest_records = json.loads(download_manifest_path.read_text(encoding="utf-8"))
        files = [
            Path(record["path"])
            for record in manifest_records
            if record.get("status") in {"exists", "downloaded"} and record.get("path")
        ]
    else:
        files = sorted(files_dir.glob("*.tsv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No downloaded GDC TSV files found in {files_dir}")

    tpm_parts = []
    count_parts = []
    gene_meta = None
    sample_rows = []

    for path in files:
        table = _read_star_counts(path)
        sample_id = path.name.replace(".rna_seq.augmented_star_gene_counts.tsv", "")
        if gene_meta is None:
            gene_meta = table[["gene_id", "gene_name", "gene_type"]].drop_duplicates("gene_id")

        tpm_parts.append(table[["gene_id", TPM_COLUMN]].rename(columns={TPM_COLUMN: sample_id}))
        count_parts.append(table[["gene_id", COUNT_COLUMN]].rename(columns={COUNT_COLUMN: sample_id}))

        index_match = index[index["file_name"] == path.name] if not index.empty else pd.DataFrame()
        meta = index_match.iloc[0].to_dict() if not index_match.empty else {}
        sample_rows.append(
            {
                "sample_id": sample_id,
                "file_name": path.name,
                "file_path": str(path),
                "case_submitter_ids": meta.get("case_submitter_ids", ""),
                "project_ids": meta.get("project_ids", ""),
                "primary_sites": meta.get("primary_sites", ""),
                "disease_types": meta.get("disease_types", ""),
                "sample_types": meta.get("sample_types", ""),
            }
        )

    tpm_matrix = tpm_parts[0]
    for part in tpm_parts[1:]:
        tpm_matrix = tpm_matrix.merge(part, on="gene_id", how="outer")

    count_matrix = count_parts[0]
    for part in count_parts[1:]:
        count_matrix = count_matrix.merge(part, on="gene_id", how="outer")

    tpm_path = source_processed / "gdc_rnaseq_tpm_matrix.parquet"
    count_path = source_processed / "gdc_rnaseq_count_matrix.parquet"
    gene_meta_path = source_processed / "gdc_gene_metadata.parquet"
    sample_meta_path = source_processed / "gdc_sample_metadata.parquet"

    tpm_matrix.to_parquet(tpm_path, index=False)
    count_matrix.to_parquet(count_path, index=False)
    if gene_meta is not None:
        gene_meta.to_parquet(gene_meta_path, index=False)
    pd.DataFrame(sample_rows).to_parquet(sample_meta_path, index=False)

    summary = {
        "source_key": source_key,
        "files": len(files),
        "genes": int(tpm_matrix.shape[0]),
        "samples": int(tpm_matrix.shape[1] - 1),
        "projects": pd.DataFrame(sample_rows)["project_ids"].value_counts().sort_index().to_dict(),
        "selection_source": str(download_manifest_path)
        if download_manifest_path.exists()
        else "directory_listing",
        "tpm_matrix": str(tpm_path),
        "count_matrix": str(count_path),
        "gene_metadata": str(gene_meta_path),
        "sample_metadata": str(sample_meta_path),
    }
    (source_processed / "gdc_rnaseq_harmonisation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
