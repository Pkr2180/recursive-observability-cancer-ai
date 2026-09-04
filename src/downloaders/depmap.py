from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


DEPMAP_FILES_ENDPOINT = "https://depmap.org/portal/api/download/files"
DEPMAP_24Q2_FIGSHARE_ARTICLE_ID = 25880521


def fetch_depmap_file_manifest(
    out_dir: Path,
    release_filter: str = "25Q2",
    file_names: Optional[list[str]] = None,
) -> dict:
    """Fetch the DepMap portal download file manifest.

    DepMap staff recommend this CSV endpoint for bulk download planning. Some
    URLs may be signed and should be refreshed near the time of download.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(DEPMAP_FILES_ENDPOINT, timeout=120)
    response.raise_for_status()

    manifest_path = out_dir / "depmap_portal_file_manifest.csv"
    manifest_path.write_bytes(response.content)
    table = pd.read_csv(manifest_path)

    filtered = table.copy()
    release_columns = [
        column
        for column in filtered.columns
        if "release" in column.lower() or "version" in column.lower() or "file_set" in column.lower()
    ]
    if release_filter and release_columns:
        mask = pd.Series(False, index=filtered.index)
        for column in release_columns:
            mask = mask | filtered[column].astype(str).str.contains(
                release_filter, case=False, na=False
            )
        filtered = filtered[mask]

    if file_names:
        lowered = {name.lower() for name in file_names}
        name_columns = [
            column
            for column in filtered.columns
            if "file" in column.lower() or "name" in column.lower()
        ]
        mask = pd.Series(False, index=filtered.index)
        for column in name_columns:
            mask = mask | filtered[column].astype(str).str.lower().isin(lowered)
        filtered = filtered[mask]

    filtered_path = out_dir / "depmap_selected_files.csv"
    filtered.to_csv(filtered_path, index=False)
    preview = filtered.head(20).astype(str).to_dict(orient="records")

    return {
        "endpoint": DEPMAP_FILES_ENDPOINT,
        "manifest": str(manifest_path),
        "selected_manifest": str(filtered_path),
        "columns": list(table.columns),
        "total_rows": int(len(table)),
        "selected_rows": int(len(filtered)),
        "preview": preview,
    }


def _find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for column in columns:
        lower = column.lower()
        if any(candidate.lower() in lower for candidate in candidates):
            return column
    return None


def download_depmap_selected_files(
    out_dir: Path,
    release_filter: str = "25Q2",
    file_names: Optional[list[str]] = None,
    max_files: int = 1,
) -> dict:
    """Download selected DepMap files listed by the portal manifest."""
    manifest_info = fetch_depmap_file_manifest(
        out_dir=out_dir,
        release_filter=release_filter,
        file_names=file_names,
    )
    selected_path = Path(manifest_info["selected_manifest"])
    selected = pd.read_csv(selected_path)
    if selected.empty:
        return {"status": "skipped", "reason": "no matching DepMap files", **manifest_info}

    url_column = _find_column(list(selected.columns), ["url", "download_url", "downloadUrl"])
    name_column = _find_column(list(selected.columns), ["file_name", "filename", "name", "file"])
    if url_column is None:
        raise ValueError(f"Could not identify URL column in DepMap manifest: {selected.columns}")
    if name_column is None:
        name_column = url_column

    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    downloads = []
    for _, row in selected.head(max_files).iterrows():
        url = str(row[url_column])
        raw_name = str(row[name_column]).split("/")[-1].split("?")[0]
        file_name = raw_name or f"depmap_file_{len(downloads) + 1}.dat"
        destination = files_dir / file_name

        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            tmp_path = destination.with_suffix(destination.suffix + ".part")
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
                "file_name": file_name,
                "path": str(destination),
                "bytes": bytes_written,
                "url_column": url_column,
                "name_column": name_column,
            }
        )

    download_manifest = out_dir / "depmap_download_manifest.json"
    download_manifest.write_text(json.dumps(downloads, indent=2), encoding="utf-8")
    return {
        "status": "downloaded",
        "release_filter": release_filter,
        "requested_files": file_names,
        "completed": len(downloads),
        "download_manifest": str(download_manifest),
        "downloads": downloads,
        "source_manifest": manifest_info,
    }


def fetch_depmap_figshare_manifest(
    out_dir: Path,
    article_id: int = DEPMAP_24Q2_FIGSHARE_ARTICLE_ID,
    file_names: Optional[list[str]] = None,
    name_contains: str = "",
) -> dict:
    """Fetch a public DepMap Figshare release manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    response = requests.get(api_url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    files = payload.get("files", [])
    rows = []
    for item in files:
        rows.append(
            {
                "article_id": article_id,
                "release_title": payload.get("title"),
                "file_id": item.get("id"),
                "file_name": item.get("name"),
                "size": item.get("size"),
                "download_url": item.get("download_url"),
            }
        )
    table = pd.DataFrame(rows)
    manifest_path = out_dir / f"depmap_figshare_{article_id}_manifest.csv"
    table.to_csv(manifest_path, index=False)

    selected = table.copy()
    if file_names:
        wanted = {name.lower() for name in file_names}
        selected = selected[selected["file_name"].astype(str).str.lower().isin(wanted)]
    if name_contains:
        selected = selected[
            selected["file_name"].astype(str).str.contains(name_contains, case=False, na=False)
        ]
    selected_path = out_dir / f"depmap_figshare_{article_id}_selected.csv"
    selected.to_csv(selected_path, index=False)

    return {
        "api_url": api_url,
        "article_id": article_id,
        "title": payload.get("title"),
        "manifest": str(manifest_path),
        "selected_manifest": str(selected_path),
        "total_rows": int(len(table)),
        "selected_rows": int(len(selected)),
        "preview": selected.head(20).astype(str).to_dict(orient="records"),
    }


def download_depmap_figshare_files(
    out_dir: Path,
    article_id: int = DEPMAP_24Q2_FIGSHARE_ARTICLE_ID,
    file_names: Optional[list[str]] = None,
    name_contains: str = "",
    max_files: int = 1,
) -> dict:
    """Download selected files from a public DepMap Figshare release."""
    manifest_info = fetch_depmap_figshare_manifest(
        out_dir=out_dir,
        article_id=article_id,
        file_names=file_names,
        name_contains=name_contains,
    )
    selected = pd.read_csv(manifest_info["selected_manifest"])
    if selected.empty:
        return {"status": "skipped", "reason": "no matching files", **manifest_info}

    files_dir = out_dir / "figshare_files"
    files_dir.mkdir(parents=True, exist_ok=True)
    downloads = []
    for _, row in selected.head(max_files).iterrows():
        url = str(row["download_url"])
        file_name = str(row["file_name"])
        destination = files_dir / file_name
        if destination.exists() and destination.stat().st_size > 0:
            downloads.append(
                {
                    "file_name": file_name,
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                    "status": "exists",
                }
            )
            continue

        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            tmp_path = destination.with_suffix(destination.suffix + ".part")
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
                "file_name": file_name,
                "path": str(destination),
                "bytes": bytes_written,
                "status": "downloaded",
            }
        )

    download_manifest = out_dir / f"depmap_figshare_{article_id}_download_manifest.json"
    download_manifest.write_text(json.dumps(downloads, indent=2), encoding="utf-8")
    return {
        "status": "downloaded",
        "article_id": article_id,
        "completed": len(downloads),
        "download_manifest": str(download_manifest),
        "downloads": downloads,
        "source_manifest": manifest_info,
    }
