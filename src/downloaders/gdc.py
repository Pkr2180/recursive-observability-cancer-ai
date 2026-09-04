from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def query_gdc_files(source: dict[str, Any], out_dir: Path) -> dict:
    """Query public GDC metadata and save the file index.

    This does not download huge genomic files by default. It creates a remote manifest
    that can be reviewed before a controlled download job is launched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    projects = source.get("projects", ["TCGA-*"])
    data_category = source.get("data_category")
    data_type = source.get("data_type")
    access = source.get("access")
    workflow_type = source.get("analysis_workflow_type")
    size = int(source.get("page_size", 100))

    common_params = {
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "data_category",
                "data_type",
                "experimental_strategy",
                "access",
                "analysis.workflow_type",
                "cases.project.project_id",
                "cases.project.primary_site",
                "cases.project.disease_type",
                "cases.case_id",
                "cases.submitter_id",
                "cases.samples.sample_id",
                "cases.samples.submitter_id",
                "cases.samples.sample_type",
            ]
        ),
        "format": "JSON",
        "size": size,
    }
    hits = []
    # Query each project separately. A combined capped query can silently allow a
    # large cohort to crowd smaller cancers out of a nominally pan-cancer index.
    for project in projects:
        filters: dict[str, Any] = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}}
            ],
        }
        if data_category:
            filters["content"].append(
                {"op": "in", "content": {"field": "files.data_category", "value": data_category}}
            )
        if data_type:
            filters["content"].append(
                {"op": "in", "content": {"field": "files.data_type", "value": data_type}}
            )
        if access:
            filters["content"].append(
                {"op": "in", "content": {"field": "files.access", "value": access}}
            )
        if workflow_type:
            filters["content"].append(
                {
                    "op": "in",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": workflow_type,
                    },
                }
            )
        response = requests.post(
            GDC_FILES_ENDPOINT,
            json={**common_params, "filters": filters},
            timeout=120,
        )
        response.raise_for_status()
        hits.extend(response.json().get("data", {}).get("hits", []))
    rows = []
    for hit in hits:
        cases = hit.get("cases") or [{}]
        project_ids = sorted(
            {
                case.get("project", {}).get("project_id")
                for case in cases
                if case.get("project", {}).get("project_id")
            }
        )
        primary_sites = sorted(
            {
                case.get("project", {}).get("primary_site")
                for case in cases
                if case.get("project", {}).get("primary_site")
            }
        )
        disease_types = sorted(
            {
                case.get("project", {}).get("disease_type")
                for case in cases
                if case.get("project", {}).get("disease_type")
            }
        )
        case_ids = sorted({case.get("case_id") for case in cases if case.get("case_id")})
        case_submitters = sorted(
            {case.get("submitter_id") for case in cases if case.get("submitter_id")}
        )
        sample_ids = []
        sample_submitters = []
        sample_types = []
        for case in cases:
            for sample in case.get("samples", []) or []:
                if sample.get("sample_id"):
                    sample_ids.append(sample["sample_id"])
                if sample.get("submitter_id"):
                    sample_submitters.append(sample["submitter_id"])
                if sample.get("sample_type"):
                    sample_types.append(sample["sample_type"])
        rows.append(
            {
                "file_id": hit.get("file_id"),
                "file_name": hit.get("file_name"),
                "data_category": hit.get("data_category"),
                "data_type": hit.get("data_type"),
                "experimental_strategy": hit.get("experimental_strategy"),
                "access": hit.get("access"),
                "analysis_workflow_type": (hit.get("analysis") or {}).get("workflow_type"),
                "project_ids": ";".join(project_ids),
                "primary_sites": ";".join(primary_sites),
                "disease_types": ";".join(disease_types),
                "case_ids": ";".join(case_ids),
                "case_submitter_ids": ";".join(case_submitters),
                "sample_ids": ";".join(sorted(set(sample_ids))),
                "sample_submitter_ids": ";".join(sorted(set(sample_submitters))),
                "sample_types": ";".join(sorted(set(sample_types))),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.drop_duplicates("file_id").sort_values(
            ["project_ids", "file_id"], kind="stable"
        )
    out_path = out_dir / "gdc_file_index.parquet"
    table.to_parquet(out_path, index=False)
    return {
        "endpoint": GDC_FILES_ENDPOINT,
        "rows": len(table),
        "projects": table["project_ids"].value_counts().sort_index().to_dict(),
        "path": str(out_path),
        "note": "Metadata index only. File downloads should be launched deliberately.",
    }


def download_indexed_gdc_files(out_dir: Path, max_files: int = 5) -> dict:
    """Download a controlled number of files listed in a saved GDC index."""
    index_path = out_dir / "gdc_file_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing GDC index: {index_path}")

    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_parquet(index_path)
    if "file_id" not in index.columns or "file_name" not in index.columns:
        raise ValueError(f"GDC index missing required columns: {index_path}")

    eligible = index.dropna(subset=["file_id", "file_name"]).copy()
    if "project_ids" in eligible.columns:
        # Stable round-robin selection preserves project coverage at every budget.
        eligible["project_rank"] = eligible.groupby("project_ids").cumcount()
        selected = eligible.sort_values(["project_rank", "project_ids"], kind="stable").head(max_files)
    else:
        selected = eligible.head(max_files)
    downloads = []
    for row in selected.itertuples(index=False):
        file_id = getattr(row, "file_id")
        file_name = getattr(row, "file_name")
        destination = files_dir / str(file_name)
        if destination.exists() and destination.stat().st_size > 0:
            downloads.append(
                {
                    "file_id": file_id,
                    "file_name": file_name,
                    "status": "exists",
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                }
            )
            continue

        url = f"{GDC_DATA_ENDPOINT}/{file_id}"
        last_error = ""
        for attempt in range(1, 4):
            tmp_path = destination.with_suffix(destination.suffix + ".part")
            try:
                with requests.get(url, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    bytes_written = 0
                    with tmp_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            bytes_written += len(chunk)
                    tmp_path.replace(destination)
                downloads.append(
                    {
                        "file_id": file_id,
                        "file_name": file_name,
                        "status": "downloaded",
                        "attempts": attempt,
                        "path": str(destination),
                        "bytes": bytes_written,
                    }
                )
                break
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                tmp_path.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(2**attempt)
        else:
            downloads.append(
                {
                    "file_id": file_id,
                    "file_name": file_name,
                    "status": "failed",
                    "attempts": 3,
                    "path": str(destination),
                    "bytes": 0,
                    "error": last_error,
                }
            )

    manifest_path = out_dir / "gdc_download_manifest.json"
    manifest_path.write_text(
        pd.DataFrame(downloads).to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    return {
        "index": str(index_path),
        "files_dir": str(files_dir),
        "requested": int(max_files),
        "completed": len(downloads),
        "successful": sum(item["status"] in {"exists", "downloaded"} for item in downloads),
        "failed": sum(item["status"] == "failed" for item in downloads),
        "projects": selected["project_ids"].value_counts().sort_index().to_dict()
        if "project_ids" in selected.columns
        else {},
        "manifest": str(manifest_path),
        "downloads": downloads,
    }
