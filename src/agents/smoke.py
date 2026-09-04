from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.agents.metrics import (
    agentic_divergence_index,
    internal_reinforcement_instability,
    observer_system_instability_coupling,
)
from src.observability import TransitionEvent, build_recursive_observability_report


def run_smoke_test(results_dir: Path) -> dict:
    """Run deterministic synthetic checks for IRI, ADI and OSIC."""
    rng = np.random.default_rng(42)
    future_probs = rng.dirichlet(alpha=np.ones(6), size=8)
    agent_vectors = rng.normal(size=(8, 6))

    iri = [
        internal_reinforcement_instability(
            future_probabilities=row,
            policy_variance=float(rng.uniform(0.05, 0.40)),
            confidence_volatility=float(rng.uniform(0.05, 0.40)),
            perturbation_sensitivity=float(rng.uniform(0.05, 0.40)),
        )
        for row in future_probs
    ]
    adi = agentic_divergence_index(agent_vectors)
    u_pcr = np.asarray(iri)
    u_cancer = u_pcr * 0.7 + rng.normal(0, 0.05, size=u_pcr.shape)
    osic = observer_system_instability_coupling(u_pcr, u_cancer)

    events = [
        TransitionEvent(
            episode_id="synthetic-001", cancer_state_id=f"state-{i}", agent_id=f"agent-{i}",
            agent_state_before=agent_vectors[i].tolist(),
            agent_state_after=(agent_vectors[i] + 0.01).tolist(), action="observe", reward=float(1 - iri[i]),
            agent_uncertainty=float(iri[i]), master_state_before=agent_vectors.mean(axis=0).tolist(),
            master_state_after=(agent_vectors.mean(axis=0) + 0.01).tolist(),
            master_uncertainty=float(u_pcr[i]), meta_uncertainty=float(abs(u_pcr[i] - u_cancer[i])),
            evidence_ids=[f"synthetic-evidence-{i}"], parent_trace="synthetic-root",
            future_states_considered={f"future-{j}": float(p) for j, p in enumerate(future_probs[i])},
            quantum_inspired_state={f"amplitude-{j}": float(np.sqrt(p)) for j, p in enumerate(future_probs[i])},
            gate_applied="instability", hypothesis_before="H0", hypothesis_after="H1",
            time=f"2026-01-01T00:00:{i:02d}+00:00",
        ).to_record()
        for i in range(8)
    ]
    master_state = agent_vectors.mean(axis=0)
    recursive = build_recursive_observability_report(
        events=events, cancer_instability=u_cancer, individual_agent_instability=u_pcr,
        inter_agent_disagreement=np.full(8, adi), master_instability=u_pcr * 0.9,
        meta_uncertainty=np.abs(u_pcr - u_cancer), master_state=master_state,
        reconstructed_master_state=master_state + 0.001,
        self_observation=u_pcr, reconstructed_self_observation=u_pcr + 0.001,
        agent_influence_identifiability=0.95, uncertainty_visibility=1.0,
    )

    out = {
        "iri_mean": float(np.mean(iri)),
        "adi": adi,
        "osic": osic,
        "recursive_observability": recursive.to_dict(),
        "status": "passed",
        "note": "Synthetic smoke test only; not a biological result.",
    }
    out_dir = results_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agent_smoke_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
