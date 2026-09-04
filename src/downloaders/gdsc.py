from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


GDSC_RELEASES = {
    "GDSC1": "https://cog.sanger.ac.uk/cmp/download/GDSC1_fitted_dose_response_27Oct23.xlsx",
    "GDSC2": "https://cog.sanger.ac.uk/cmp/download/GDSC2_fitted_dose_response_27Oct23.xlsx",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_gdsc_fitted_releases(out_dir: Path) -> dict:
    """Download fitted GDSC workbooks from the official Sanger host."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads = []
    for release, url in GDSC_RELEASES.items():
        destination = out_dir / url.rsplit("/", 1)[-1]
        status = "exists"
        if not destination.exists() or destination.stat().st_size == 0:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            temporary.replace(destination)
            status = "downloaded"
        downloads.append(
            {
                "release": release,
                "official_url": url,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "status": status,
            }
        )
    manifest = out_dir / "gdsc_official_download_manifest.json"
    manifest.write_text(json.dumps(downloads, indent=2), encoding="utf-8")
    return {
        "status": "created",
        "source_authority": "Wellcome Sanger Institute Cell Model Passports",
        "files": len(downloads),
        "bytes": sum(item["bytes"] for item in downloads),
        "manifest": str(manifest),
        "downloads": downloads,
    }
