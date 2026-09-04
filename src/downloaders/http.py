from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import requests


def stream_download(url: str, destination: Path, sha256: Optional[str] = None) -> dict:
    """Download a URL to a remote Modal Volume path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        digest = hashlib.sha256()
        bytes_written = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)

    observed = digest.hexdigest()
    if sha256 and observed.lower() != sha256.lower():
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for {url}: expected {sha256}, observed {observed}"
        )

    tmp_path.replace(destination)
    return {
        "url": url,
        "destination": str(destination),
        "bytes": bytes_written,
        "sha256": observed,
    }

