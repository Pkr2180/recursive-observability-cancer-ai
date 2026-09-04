from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.downloaders.http import stream_download


PCAWG_HUB = "https://pcawg.xenahubs.net"
PCAWG_MARKER_PAPER = "https://doi.org/10.1038/s41586-020-1969-6"
DATASETS = {
    "survival_donor.tsv": "survival_donor",
    "project_code_donor.tsv": "project_code_donor",
    "donor_clinical.tsv": "pcawg_donor_clinical_August2016_v9",
    "gene_expression_fpkm_uq_log.tsv": "tophat_star_fpkm_uq.v2_aliquot_gl.donor.log",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pcawg_external(out_dir: Path) -> dict:
    """Download open PCAWG donor RNA-seq and clinical matrices from UCSC Xena."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, dataset in DATASETS.items():
        destination = out_dir / filename
        url = f"{PCAWG_HUB}/download/{dataset}"
        if destination.exists() and destination.stat().st_size > 0:
            record = {
                "url": url,
                "destination": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "status": "exists",
            }
        else:
            record = stream_download(url, destination)
            record["status"] = "downloaded"
        record["xena_dataset"] = dataset
        records.append(record)

    manifest = {
        "source_name": "PCAWG donor-centric open-access cohort",
        "source_policy": "official_public_observed_patient_data",
        "hub": PCAWG_HUB,
        "marker_paper": PCAWG_MARKER_PAPER,
        "independence_rule": (
            "exclude all dcc_project_code values ending -US because PCAWG -US projects are "
            "TCGA-derived; retain only non-US ICGC projects"
        ),
        "downloads": records,
        "simulation": False,
    }
    manifest_path = out_dir / "pcawg_external_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "status": "completed",
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "manifest": str(manifest_path),
        "simulation": False,
    }
