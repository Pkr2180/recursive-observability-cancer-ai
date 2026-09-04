from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OBSERVABILITY_LEVELS = {
    0: "external_trace",
    1: "individual_agent",
    2: "inter_agent",
    3: "master_state",
    4: "master_self_observation",
    5: "recursive_meta_observation",
    6: "observer_observer_system_coupling",
}

# These fields are deliberately explicit: a model output without the transition
# that produced it is not a scientifically observable event.
REQUIRED_TRANSITION_FIELDS = (
    "episode_id",
    "time",
    "cancer_state_id",
    "agent_id",
    "agent_state_before",
    "agent_state_after",
    "action",
    "reward",
    "agent_uncertainty",
    "master_state_before",
    "master_state_after",
    "master_uncertainty",
    "meta_uncertainty",
    "evidence_ids",
    "parent_trace",
    "future_states_considered",
    "quantum_inspired_state",
    "gate_applied",
    "hypothesis_before",
    "hypothesis_after",
)


@dataclass(frozen=True)
class TransitionEvent:
    episode_id: str
    cancer_state_id: str
    agent_id: str
    agent_state_before: Sequence[float]
    agent_state_after: Sequence[float]
    action: str
    reward: float
    agent_uncertainty: float
    master_state_before: Sequence[float]
    master_state_after: Sequence[float]
    master_uncertainty: float
    meta_uncertainty: float
    evidence_ids: Sequence[str]
    parent_trace: str
    future_states_considered: Mapping[str, float]
    quantum_inspired_state: Mapping[str, float]
    gate_applied: str
    hypothesis_before: str
    hypothesis_after: str
    time: str = ""

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if not record["time"]:
            record["time"] = datetime.now(timezone.utc).isoformat()
        return record


def append_transition_event(path: Path, event: TransitionEvent) -> None:
    """Append one validated transition to an audit-friendly JSONL event stream."""
    record = event.to_record()
    validate_transition_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _finite_vector(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) == 0:
        return False
    try:
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except (TypeError, ValueError):
        return False


def _finite_nonnegative_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping) or len(value) == 0:
        return False
    try:
        numbers = np.asarray(list(value.values()), dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numbers).all() and (numbers >= 0).all() and numbers.sum() > 0)


def transition_validation_errors(record: Mapping[str, Any]) -> list[str]:
    """Return fail-closed scientific telemetry errors for one transition record."""
    errors = [
        f"missing:{field}"
        for field in REQUIRED_TRANSITION_FIELDS
        if not _present(record.get(field))
    ]
    for field in (
        "agent_state_before",
        "agent_state_after",
        "master_state_before",
        "master_state_after",
    ):
        if _present(record.get(field)) and not _finite_vector(record.get(field)):
            errors.append(f"invalid_finite_state_vector:{field}")

    for before, after in (
        ("agent_state_before", "agent_state_after"),
        ("master_state_before", "master_state_after"),
    ):
        if _finite_vector(record.get(before)) and _finite_vector(record.get(after)):
            if len(record[before]) != len(record[after]):
                errors.append(f"state_dimension_mismatch:{before}:{after}")

    for field in ("reward", "agent_uncertainty", "master_uncertainty", "meta_uncertainty"):
        if not _present(record.get(field)):
            continue
        try:
            value = float(record[field])
        except (TypeError, ValueError):
            errors.append(f"invalid_numeric:{field}")
            continue
        if not np.isfinite(value):
            errors.append(f"nonfinite_numeric:{field}")
        if "uncertainty" in field and value < 0:
            errors.append(f"negative_uncertainty:{field}")

    for field in ("future_states_considered", "quantum_inspired_state"):
        if _present(record.get(field)) and not _finite_nonnegative_mapping(record.get(field)):
            errors.append(f"invalid_nonnegative_mapping:{field}")

    futures = record.get("future_states_considered")
    amplitudes = record.get("quantum_inspired_state")
    if _finite_nonnegative_mapping(futures) and _finite_nonnegative_mapping(amplitudes):
        if set(futures) != set(amplitudes):
            errors.append("future_quantum_key_mismatch")
        else:
            future_values = np.asarray(list(futures.values()), dtype=float)
            amplitude_values = np.asarray([amplitudes[key] for key in futures], dtype=float)
            normalized_future = future_values / future_values.sum()
            normalized_amplitude = amplitude_values**2 / np.sum(amplitude_values**2)
            if not np.allclose(normalized_future, normalized_amplitude, atol=1e-5, rtol=1e-4):
                errors.append("quantum_amplitudes_do_not_encode_future_probabilities")
    return errors


def validate_transition_record(record: Mapping[str, Any]) -> None:
    """Raise when a transition cannot support independent scientific observability."""
    errors = transition_validation_errors(record)
    if errors:
        raise ValueError("transition event failed scientific observability: " + ", ".join(errors))


def validate_transition_stream(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate a complete event stream and expose exact failures instead of imputing them."""
    failures = []
    for index, event in enumerate(events):
        errors = transition_validation_errors(event)
        if errors:
            failures.append({"event_index": index, "errors": errors})
    return {
        "events": len(events),
        "valid_events": len(events) - len(failures),
        "invalid_events": len(failures),
        "strict_validity": float((len(events) - len(failures)) / len(events)) if events else 0.0,
        "failures": failures,
    }


def trace_completeness(events: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of required scientific-state fields present across all events."""
    if not events:
        return 0.0
    observed = sum(_present(event.get(field)) for event in events for field in REQUIRED_TRANSITION_FIELDS)
    return float(observed / (len(events) * len(REQUIRED_TRANSITION_FIELDS)))


def normalized_reconstructability(actual: np.ndarray, reconstructed: np.ndarray) -> float:
    """Bounded state reconstructability, 1 for an exact reconstruction."""
    actual = np.asarray(actual, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    if actual.shape != reconstructed.shape or actual.size == 0:
        raise ValueError("actual and reconstructed states must have the same non-empty shape")
    error = np.linalg.norm(actual - reconstructed)
    scale = np.linalg.norm(actual) + np.finfo(float).eps
    return float(np.clip(1.0 - error / scale, 0.0, 1.0))


def recursive_consistency(self_estimate: np.ndarray, meta_reconstruction: np.ndarray) -> float:
    return normalized_reconstructability(self_estimate, meta_reconstruction)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return 0.0
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def oosc_profile(
    cancer_instability: np.ndarray,
    agent_instability: np.ndarray,
    master_instability: np.ndarray,
    meta_uncertainty: np.ndarray,
) -> dict[str, float]:
    """Transparent OOSC profile; no unjustified single composite is created."""
    return {
        "cancer_agent": _safe_corr(cancer_instability, agent_instability),
        "cancer_master": _safe_corr(cancer_instability, master_instability),
        "cancer_meta": _safe_corr(cancer_instability, meta_uncertainty),
        "agent_master": _safe_corr(agent_instability, master_instability),
        "master_meta": _safe_corr(master_instability, meta_uncertainty),
    }


@dataclass(frozen=True)
class RecursiveObservabilityReport:
    rmoi_profile: Mapping[str, float]
    instability_channels: Mapping[str, Sequence[float]]
    osic: float
    oosc: Mapping[str, float]
    observer_depth: int
    primary_endpoint: str = "recursive_scientific_observability"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_recursive_observability_report(
    *,
    events: Sequence[Mapping[str, Any]],
    cancer_instability: np.ndarray,
    individual_agent_instability: np.ndarray,
    inter_agent_disagreement: np.ndarray,
    master_instability: np.ndarray,
    meta_uncertainty: np.ndarray,
    master_state: np.ndarray,
    reconstructed_master_state: np.ndarray,
    self_observation: np.ndarray,
    reconstructed_self_observation: np.ndarray,
    agent_influence_identifiability: float,
    uncertainty_visibility: float,
) -> RecursiveObservabilityReport:
    """Build the required O0-O6 report while keeping all instabilities separate."""
    streams = [
        np.asarray(x, dtype=float)
        for x in (
            cancer_instability,
            individual_agent_instability,
            inter_agent_disagreement,
            master_instability,
            meta_uncertainty,
        )
    ]
    lengths = {stream.shape[0] for stream in streams}
    if len(lengths) != 1:
        raise ValueError("all instability streams must share one time axis")

    sr = normalized_reconstructability(master_state, reconstructed_master_state)
    rc = recursive_consistency(self_observation, reconstructed_self_observation)
    tc = trace_completeness(events)
    rmoi = {
        "state_reconstructability": sr,
        "trace_completeness": tc,
        "agent_influence_identifiability": float(np.clip(agent_influence_identifiability, 0, 1)),
        "uncertainty_visibility": float(np.clip(uncertainty_visibility, 0, 1)),
        "meta_uncertainty_observability": float(np.mean(np.isfinite(streams[4]))),
        "recursive_consistency": rc,
    }
    depth_checks = [tc > 0, streams[1].size > 0, streams[2].size > 0, sr > 0, streams[3].size > 0, rc > 0]
    observer_depth = sum(1 for available in depth_checks if available)
    oosc = oosc_profile(streams[0], streams[1], streams[3], streams[4])
    return RecursiveObservabilityReport(
        rmoi_profile=rmoi,
        instability_channels={
            "cancer_future_state": streams[0].tolist(),
            "individual_agents": streams[1].tolist(),
            "inter_agent_disagreement": streams[2].tolist(),
            "master_state": streams[3].tolist(),
            "master_self_observation_uncertainty": streams[4].tolist(),
        },
        osic=_safe_corr(streams[0], streams[3]),
        oosc=oosc,
        observer_depth=observer_depth,
    )
