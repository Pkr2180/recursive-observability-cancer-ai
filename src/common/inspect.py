from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def inspect_csv(path: Path, nrows: int = 5) -> dict:
    table = pd.read_csv(path, nrows=nrows, low_memory=False)
    return {
        "path": str(path),
        "preview_rows": int(len(table)),
        "columns": list(map(str, table.columns)),
        "first_column": str(table.columns[0]) if len(table.columns) else "",
        "first_values": table.iloc[:, 0].astype(str).head(nrows).tolist()
        if len(table.columns)
        else [],
        "preview": table.head(nrows).astype(str).to_dict(orient="records"),
    }


def inspect_artifact(path: Path, nrows: int = 5) -> dict:
    """Inspect a JSON, CSV, or Parquet artifact without copying it locally."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"path": str(path), "type": "json", "payload": payload}
    if suffix == ".csv":
        return {"type": "csv", **inspect_csv(path, nrows=nrows)}
    if suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
        preview = table.head(nrows)
        return {
            "path": str(path),
            "type": "parquet",
            "rows": int(len(table)),
            "columns": list(map(str, table.columns)),
            "dtypes": {str(column): str(dtype) for column, dtype in table.dtypes.items()},
            "preview": preview.astype(str).to_dict(orient="records"),
        }
    raise ValueError(f"Unsupported artifact type: {suffix}")
