from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def run_distillation_smoke_test(results_dir: Path) -> dict:
    """Synthetic teacher-student distillation check."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 12))
    teacher_weights = rng.normal(size=(12, 4))
    teacher_logits = x @ teacher_weights + rng.normal(scale=0.05, size=(200, 4))

    x_train, x_test, y_train, y_test = train_test_split(
        x, teacher_logits, test_size=0.25, random_state=7
    )
    student = Ridge(alpha=1.0)
    student.fit(x_train, y_train)
    pred = student.predict(x_test)
    dfs = float(r2_score(y_test, pred, multioutput="variance_weighted"))

    out = {
        "dfs": dfs,
        "status": "passed" if dfs > 0.8 else "warning",
        "note": "Synthetic Distillation Fidelity Score only; not a biological result.",
    }
    out_dir = results_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "distillation_smoke_test.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )
    return out

