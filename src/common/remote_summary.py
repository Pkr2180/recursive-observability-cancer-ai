from __future__ import annotations

from pathlib import Path


def summarize_remote_tree(paths: list[Path], max_entries: int = 200) -> dict:
    """Return a small file inventory from mounted Modal volume paths."""
    entries = []
    total_files = 0
    total_bytes = 0
    for base_path in paths:
        if not base_path.exists():
            continue
        for path in sorted(base_path.rglob("*")):
            if not path.is_file():
                continue
            size = path.stat().st_size
            total_files += 1
            total_bytes += size
            if len(entries) < max_entries:
                entries.append({"path": str(path), "bytes": size})
    return {
        "paths": [str(path) for path in paths],
        "total_files": total_files,
        "total_bytes": total_bytes,
        "entries_returned": len(entries),
        "bytes_in_returned_entries": sum(item["bytes"] for item in entries),
        "entries": entries,
    }
