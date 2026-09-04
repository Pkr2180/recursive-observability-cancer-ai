from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def quantise_lincs_cancer_states(processed_dir: Path, results_dir: Path) -> dict:
    """Quantise observed cancer-cell LINCS signatures into compact quintile state tokens."""
    source = processed_dir / "lincs_gse70138" / "lincs_cancer_landmark_level5.h5"
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir = results_dir / "state_tokens" / "lincs_gse70138"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "lincs_cancer_landmark_quintile_tokens.h5"
    temporary = output.with_suffix(output.suffix + ".part")

    with h5py.File(source, "r") as source_handle:
        values = source_handle["matrix"][:]
        signature_ids = source_handle["sig_id"][:]
        gene_ids = source_handle["gene_id"][:]
        gene_symbols = source_handle["gene_symbol"][:]
    edges = np.nanquantile(values, [0.2, 0.4, 0.6, 0.8], axis=0).astype(np.float32)
    tokens = np.full(values.shape, 255, dtype=np.uint8)
    for gene_index in range(values.shape[1]):
        column = values[:, gene_index]
        finite = np.isfinite(column)
        tokens[finite, gene_index] = np.digitize(
            column[finite], edges[:, gene_index], right=True
        ).astype(np.uint8)

    with h5py.File(temporary, "w") as target:
        target.attrs["source"] = str(source)
        target.attrs["data_policy"] = "observed LINCS values only"
        target.attrs["token_mapping"] = json.dumps(
            {"0": "Q0", "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4", "255": "MISSING"}
        )
        target.create_dataset("sig_id", data=signature_ids)
        target.create_dataset("gene_id", data=gene_ids)
        target.create_dataset("gene_symbol", data=gene_symbols)
        target.create_dataset("quantile_edges", data=edges)
        target.create_dataset(
            "state_token_codes",
            data=tokens,
            chunks=(512, tokens.shape[1]),
            compression="lzf",
        )
    temporary.replace(output)
    del values, tokens

    summary = {
        "status": "created",
        "data_policy": "observed LINCS values only; no generated biological samples",
        "signatures": int(len(signature_ids)),
        "landmark_genes": int(len(gene_ids)),
        "token_cells": int(len(signature_ids) * len(gene_ids)),
        "bins": 5,
        "output": str(output),
        "bytes": output.stat().st_size,
    }
    summary_path = output_dir / "lincs_quantisation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
