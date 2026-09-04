from __future__ import annotations

import numpy as np


def entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs)]
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum())


def internal_reinforcement_instability(
    future_probabilities: np.ndarray,
    policy_variance: float,
    confidence_volatility: float,
    perturbation_sensitivity: float,
) -> float:
    """Bootstrap Internal Reinforcement Instability score."""
    future_entropy = entropy(future_probabilities)
    components = np.array(
        [
            future_entropy,
            float(policy_variance),
            float(confidence_volatility),
            float(perturbation_sensitivity),
        ],
        dtype=float,
    )
    return float(np.nanmean(components))


def agentic_divergence_index(agent_vectors: np.ndarray) -> float:
    """Mean pairwise distance between small-brain state vectors."""
    vectors = np.asarray(agent_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        return 0.0
    distances = []
    for i in range(vectors.shape[0]):
        for j in range(i + 1, vectors.shape[0]):
            distances.append(float(np.linalg.norm(vectors[i] - vectors[j])))
    return float(np.mean(distances)) if distances else 0.0


def observer_system_instability_coupling(u_pcr: np.ndarray, u_cancer: np.ndarray) -> float:
    """Bootstrap OSIC as Pearson correlation for early smoke tests."""
    left = np.asarray(u_pcr, dtype=float)
    right = np.asarray(u_cancer, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return 0.0
    left = left[mask]
    right = right[mask]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])

