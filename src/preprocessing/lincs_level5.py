from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd

from src.common.paths import safe_dataset_dir
from src.downloaders.lincs import LEVEL5_FILE


def _hdf5_layout(path: Path) -> list[dict]:
    layout: list[dict] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                layout.append(
                    {
                        "path": f"/{name}",
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                        "chunks": list(item.chunks) if item.chunks else None,
                        "compression": item.compression,
                    }
                )

        handle.visititems(visitor)
    return layout


def prepare_lincs_level5(raw_dir: Path, processed_dir: Path) -> dict:
    """Decompress the checksum-verified LINCS GCTX matrix and record its layout."""
    source_raw = safe_dataset_dir(raw_dir, "lincs_gse70138")
    source_processed = safe_dataset_dir(processed_dir, "lincs_gse70138")
    source = source_raw / LEVEL5_FILE
    if not source.exists():
        raise FileNotFoundError(source)
    output = source_processed / LEVEL5_FILE.removesuffix(".gz")
    status = "exists"
    if not output.exists() or output.stat().st_size == 0:
        temporary = output.with_suffix(output.suffix + ".part")
        with gzip.open(source, "rb") as compressed, temporary.open("wb") as decompressed:
            shutil.copyfileobj(compressed, decompressed, length=16 * 1024 * 1024)
        temporary.replace(output)
        status = "decompressed"

    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    summary = {
        "status": status,
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "gctx": str(output),
        "gctx_bytes": output.stat().st_size,
        "gctx_sha256": digest.hexdigest(),
        "layout": _hdf5_layout(output),
    }
    summary_path = source_processed / "lincs_level5_layout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def extract_lincs_cancer_landmarks(processed_dir: Path, batch_size: int = 512) -> dict:
    """Extract cancer-cell Level-5 signatures for measured L1000 landmark genes."""
    source_dir = safe_dataset_dir(processed_dir, "lincs_gse70138")
    gctx = source_dir / LEVEL5_FILE.removesuffix(".gz")
    cancer_metadata_path = source_dir / "lincs_cancer_signature_info.parquet"
    gene_info_path = source_dir / "lincs_gene_info.parquet"
    cell_info_path = source_dir / "lincs_cell_info.parquet"
    missing = [
        str(path)
        for path in (gctx, cancer_metadata_path, gene_info_path, cell_info_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing LINCS extraction inputs: {missing}")

    cancer_metadata = pd.read_parquet(cancer_metadata_path)
    gene_info = pd.read_parquet(gene_info_path)
    cell_info = pd.read_parquet(cell_info_path)
    cancer_ids = set(cancer_metadata["sig_id"].astype(str))
    landmark = gene_info[gene_info["pr_is_lm"].eq(1)].copy()
    landmark_ids = set(landmark["pr_gene_id"].astype(str))
    symbol_by_id = landmark.set_index(landmark["pr_gene_id"].astype(str))["pr_gene_symbol"].to_dict()

    output = source_dir / "lincs_cancer_landmark_level5.h5"
    temporary = output.with_suffix(output.suffix + ".part")
    with h5py.File(gctx, "r") as source:
        matrix = source["/0/DATA/0/matrix"]
        signature_ids = _decode(source["/0/META/COL/id"][:])
        gene_ids = _decode(source["/0/META/ROW/id"][:])
        selected_signature_indices = [
            index for index, signature_id in enumerate(signature_ids) if signature_id in cancer_ids
        ]
        selected_gene_indices = [
            index for index, gene_id in enumerate(gene_ids) if gene_id in landmark_ids
        ]
        selected_signature_ids = [signature_ids[index] for index in selected_signature_indices]
        selected_gene_ids = [gene_ids[index] for index in selected_gene_indices]
        if not selected_signature_indices or not selected_gene_indices:
            raise ValueError("No cancer signatures or landmark genes matched the GCTX identifiers")

        with h5py.File(temporary, "w") as target:
            target.attrs["source"] = str(gctx)
            target.attrs["data_policy"] = "observed LINCS Level-5 signatures only"
            target.create_dataset(
                "sig_id",
                data=np.asarray(selected_signature_ids, dtype="S64"),
            )
            target.create_dataset(
                "gene_id",
                data=np.asarray(selected_gene_ids, dtype="S32"),
            )
            target.create_dataset(
                "gene_symbol",
                data=np.asarray(
                    [str(symbol_by_id.get(gene_id, gene_id)) for gene_id in selected_gene_ids],
                    dtype="S64",
                ),
            )
            extracted = target.create_dataset(
                "matrix",
                shape=(len(selected_signature_indices), len(selected_gene_indices)),
                dtype="float32",
                chunks=(min(batch_size, len(selected_signature_indices)), len(selected_gene_indices)),
                compression="lzf",
            )
            for start in range(0, len(selected_signature_indices), batch_size):
                batch_indices = selected_signature_indices[start : start + batch_size]
                full_block = matrix[batch_indices, :]
                extracted[start : start + len(batch_indices), :] = full_block[
                    :, selected_gene_indices
                ]
    temporary.replace(output)

    extracted_metadata = cancer_metadata[
        cancer_metadata["sig_id"].astype(str).isin(selected_signature_ids)
    ].copy()
    extracted_metadata["sig_id"] = extracted_metadata["sig_id"].astype(str)
    extracted_metadata["matrix_row"] = extracted_metadata["sig_id"].map(
        {signature_id: index for index, signature_id in enumerate(selected_signature_ids)}
    )
    cell_columns = [
        column
        for column in ("cell_id", "base_cell_id", "primary_site", "subtype", "sample_type")
        if column in cell_info.columns
    ]
    extracted_metadata = extracted_metadata.merge(
        cell_info[cell_columns].drop_duplicates("cell_id"), on="cell_id", how="left"
    ).sort_values("matrix_row")
    metadata_output = source_dir / "lincs_cancer_landmark_signature_metadata.parquet"
    extracted_metadata.to_parquet(metadata_output, index=False)

    summary = {
        "status": "created",
        "observed_data_only": True,
        "signatures": int(len(selected_signature_ids)),
        "landmark_genes": int(len(selected_gene_ids)),
        "cancer_cells": int(extracted_metadata["cell_id"].nunique()),
        "primary_sites": int(extracted_metadata["primary_site"].nunique())
        if "primary_site" in extracted_metadata
        else None,
        "matrix": str(output),
        "matrix_bytes": output.stat().st_size,
        "metadata": str(metadata_output),
    }
    summary_path = source_dir / "lincs_cancer_landmark_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
