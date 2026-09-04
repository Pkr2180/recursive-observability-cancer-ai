from __future__ import annotations

from pathlib import Path


def ensure_remote_layout(remote_root: Path) -> dict[str, Path]:
    """Create the expected Modal directory layout."""
    layout = {
        "raw": remote_root / "raw",
        "processed": remote_root / "processed",
        "models": remote_root / "models",
        "results": remote_root / "results",
        "logs": remote_root / "results" / "logs",
        "manifests": remote_root / "results" / "manifests",
        "state_tokens": remote_root / "results" / "state_tokens",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def safe_dataset_dir(base_dir: Path, source_key: str) -> Path:
    """Return a stable dataset folder without allowing path traversal."""
    clean_key = source_key.replace("/", "_").replace("\\", "_").replace("..", "_")
    path = base_dir / clean_key
    path.mkdir(parents=True, exist_ok=True)
    return path

