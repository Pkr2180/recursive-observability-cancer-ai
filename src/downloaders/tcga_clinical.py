from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.downloaders.http import stream_download


TCGA_CDR_FILE_ID = "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
TCGA_CDR_URL = f"https://api.gdc.cancer.gov/data/{TCGA_CDR_FILE_ID}"
TCGA_CDR_PUBLICATION_URL = (
    "https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tcga_cdr(out_dir: Path) -> dict:
    """Download the official open-access TCGA Pan-Cancer Clinical Data Resource."""
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "TCGA-CDR-SupplementalTableS1.xlsx"
    if destination.exists() and destination.stat().st_size > 0:
        result = {
            "url": TCGA_CDR_URL,
            "destination": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "status": "exists",
        }
    else:
        result = stream_download(TCGA_CDR_URL, destination)
        result["status"] = "downloaded"

    manifest = {
        "source_name": "TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR)",
        "source_policy": "official_open_access_observed_patient_data",
        "publication_url": TCGA_CDR_PUBLICATION_URL,
        "gdc_file_id": TCGA_CDR_FILE_ID,
        "download": result,
        "simulation": False,
    }
    manifest_path = out_dir / "tcga_cdr_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**result, "manifest": str(manifest_path), "simulation": False}
