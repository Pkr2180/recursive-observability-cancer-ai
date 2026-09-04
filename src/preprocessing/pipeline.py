from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.paths import safe_dataset_dir


SUPPORTED_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet"}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    return pd.read_csv(path)


def _profile_table(path: Path) -> dict:
    try:
        table = _read_table(path)
        return {
            "path": str(path),
            "rows": int(table.shape[0]),
            "columns": int(table.shape[1]),
            "column_names": list(map(str, table.columns[:50])),
            "status": "profiled",
        }
    except Exception as exc:  # noqa: BLE001 - profile should keep going across many files.
        return {"path": str(path), "status": "failed", "error": repr(exc)}


def preprocess_sources(
    raw_dir: Path,
    processed_dir: Path,
    source_keys: Optional[list[str]] = None,
) -> dict:
    """Create first-pass remote data profiles and harmonisation manifests."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = [path for path in raw_dir.iterdir() if path.is_dir()]
    if source_keys is not None:
        wanted = set(source_keys)
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]

    profiles = []
    for dataset_dir in sorted(dataset_dirs):
        out_dir = safe_dataset_dir(processed_dir, dataset_dir.name)
        table_files = [
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_TABLE_SUFFIXES
        ]
        dataset_profile = {
            "source_key": dataset_dir.name,
            "raw_dir": str(dataset_dir),
            "table_count": len(table_files),
            "tables": [_profile_table(path) for path in table_files[:25]],
        }
        (out_dir / "profile.json").write_text(
            json.dumps(dataset_profile, indent=2),
            encoding="utf-8",
        )
        profiles.append(dataset_profile)

    manifest_path = processed_dir / "preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    return {
        "processed_dir": str(processed_dir),
        "source_count": len(profiles),
        "manifest": str(manifest_path),
    }
