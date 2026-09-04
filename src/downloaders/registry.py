from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.common.config import load_yaml
from src.common.paths import safe_dataset_dir
from src.downloaders.gdc import query_gdc_files
from src.downloaders.http import stream_download


def _selected_sources(config: dict, source_keys: Optional[list[str]]) -> list[dict]:
    sources = config.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("configs/data_sources.yaml must contain a list named 'sources'")
    if source_keys is None:
        return sources
    wanted = set(source_keys)
    return [source for source in sources if source.get("key") in wanted]


def build_source_index(config_path: str, raw_dir: Path) -> dict:
    config = load_yaml(config_path)
    index = []
    for source in config.get("sources", []):
        key = source["key"]
        index.append(
            {
                "key": key,
                "kind": source.get("kind"),
                "enabled": bool(source.get("enabled", True)),
                "remote_dir": str(safe_dataset_dir(raw_dir, key)),
                "requires_secret": source.get("requires_secret", []),
                "notes": source.get("notes", ""),
            }
        )
    out_path = raw_dir / "source_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {"source_count": len(index), "path": str(out_path), "sources": index}


def download_sources(
    config_path: str,
    raw_dir: Path,
    source_keys: Optional[list[str]] = None,
    dry_run: bool = True,
) -> dict:
    config = load_yaml(config_path)
    selected = _selected_sources(config, source_keys)
    results = []

    for source in selected:
        key = source["key"]
        if not source.get("enabled", True):
            results.append({"key": key, "status": "skipped", "reason": "disabled"})
            continue

        out_dir = safe_dataset_dir(raw_dir, key)
        kind = source.get("kind")
        if dry_run:
            results.append(
                {
                    "key": key,
                    "status": "planned",
                    "kind": kind,
                    "remote_dir": str(out_dir),
                    "notes": source.get("notes", ""),
                }
            )
            continue

        if kind == "http":
            filename = source.get("filename") or Path(source["url"]).name
            result = stream_download(
                url=source["url"],
                destination=out_dir / filename,
                sha256=source.get("sha256"),
            )
            results.append({"key": key, "status": "downloaded", **result})
        elif kind == "gdc_index":
            result = query_gdc_files(source=source, out_dir=out_dir)
            results.append({"key": key, "status": "indexed", **result})
        else:
            results.append(
                {
                    "key": key,
                    "status": "skipped",
                    "reason": f"unsupported or credential-gated source kind: {kind}",
                }
            )

    return {"dry_run": dry_run, "source_count": len(selected), "results": results}

