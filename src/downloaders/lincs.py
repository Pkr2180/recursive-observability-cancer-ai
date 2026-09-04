from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

import requests


GEO_ROOT = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl"
CHECKSUM_FILE = "GSE70138_SHA512SUMS.txt.gz"
LEVEL5_FILE = "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
METADATA_FILES = (
    "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz",
    "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
    "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz",
    "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
    "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
    "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz",
)


def _parse_official_checksums(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.readlines()
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        checksums[Path(parts[1].lstrip("* ")).name] = parts[0].lower()
    return checksums


def _download(
    out_dir: Path,
    filename: str,
    expected_sha512: str | None = None,
) -> dict:
    destination = out_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        sha512 = hashlib.sha512()
        sha256 = hashlib.sha256()
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha512.update(chunk)
                sha256.update(chunk)
        actual_sha512 = sha512.hexdigest()
        verified = expected_sha512 is None or actual_sha512 == expected_sha512
        if verified:
            return {
                "filename": filename,
                "status": "exists",
                "bytes": destination.stat().st_size,
                "sha256": sha256.hexdigest(),
                "sha512": actual_sha512,
                "official_sha512_verified": expected_sha512 is not None,
                "path": str(destination),
            }
        destination.unlink()

    url = f"{GEO_ROOT}/{filename}"
    last_error = ""
    for attempt in range(1, 4):
        temporary = destination.with_suffix(destination.suffix + ".part")
        sha512 = hashlib.sha512()
        sha256 = hashlib.sha256()
        bytes_written = 0
        try:
            with requests.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        sha512.update(chunk)
                        sha256.update(chunk)
                        bytes_written += len(chunk)
            actual_sha512 = sha512.hexdigest()
            if expected_sha512 is not None and actual_sha512 != expected_sha512:
                raise ValueError(
                    f"Official SHA512 mismatch for {filename}: {actual_sha512}"
                )
            temporary.replace(destination)
            return {
                "filename": filename,
                "status": "downloaded",
                "attempts": attempt,
                "bytes": bytes_written,
                "sha256": sha256.hexdigest(),
                "sha512": actual_sha512,
                "official_sha512_verified": expected_sha512 is not None,
                "path": str(destination),
                "url": url,
            }
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {filename}: {last_error}")


def _write_manifest(out_dir: Path, name: str, records: list[dict]) -> dict:
    manifest = out_dir / name
    result = {
        "status": "completed",
        "source_authority": "NCBI Gene Expression Omnibus",
        "accession": "GSE70138",
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "official_sha512_verified": sum(
            bool(record["official_sha512_verified"]) for record in records
        ),
        "records": records,
    }
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest"] = str(manifest)
    return result


def download_lincs_phase2_metadata(out_dir: Path) -> dict:
    """Download official LINCS Phase II metadata and verify GEO checksums."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checksum_record = _download(out_dir, CHECKSUM_FILE)
    expected = _parse_official_checksums(out_dir / CHECKSUM_FILE)
    records = [checksum_record]
    for filename in METADATA_FILES:
        records.append(_download(out_dir, filename, expected.get(filename)))
    return _write_manifest(out_dir, "lincs_phase2_metadata_manifest.json", records)


def download_lincs_phase2_level5(out_dir: Path) -> dict:
    """Download the official Level-5 post-perturbation expression matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = out_dir / CHECKSUM_FILE
    checksum_record = None
    if not checksum_path.exists():
        checksum_record = _download(out_dir, CHECKSUM_FILE)
    expected = _parse_official_checksums(checksum_path)
    records = [_download(out_dir, LEVEL5_FILE, expected.get(LEVEL5_FILE))]
    if checksum_record is not None:
        records.insert(0, checksum_record)
    return _write_manifest(out_dir, "lincs_phase2_level5_manifest.json", records)
