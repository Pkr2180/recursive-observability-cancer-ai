from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.observability import (
    TransitionEvent,
    append_transition_event,
    trace_completeness,
    transition_validation_errors,
    validate_transition_stream,
)
from src.observability.recursive import normalized_reconstructability, oosc_profile


def complete_event() -> TransitionEvent:
    return TransitionEvent(
        episode_id="e1", time="2026-01-01T00:00:00+00:00", cancer_state_id="c1",
        agent_id="a1", agent_state_before=[0.0], agent_state_after=[1.0], action="observe",
        reward=0.5, agent_uncertainty=0.2, master_state_before=[0.0],
        master_state_after=[0.5], master_uncertainty=0.3, meta_uncertainty=0.1,
        evidence_ids=["source-1"], parent_trace="root", future_states_considered={"f1": 1.0},
        quantum_inspired_state={"f1": 1.0}, gate_applied="branch", hypothesis_before="H0",
        hypothesis_after="H1",
    )


class RecursiveObservabilityTests(unittest.TestCase):
    def test_complete_event_scores_one_and_persists(self) -> None:
        event = complete_event()
        self.assertEqual(trace_completeness([event.to_record()]), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            append_transition_event(path, event)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_missing_evidence_is_observability_failure(self) -> None:
        record = complete_event().to_record()
        record["evidence_ids"] = []
        self.assertLess(trace_completeness([record]), 1.0)
        self.assertIn("missing:evidence_ids", transition_validation_errors(record))

    def test_invalid_quantum_encoding_fails_closed(self) -> None:
        record = complete_event().to_record()
        record["future_states_considered"] = {"f1": 0.25, "f2": 0.75}
        record["quantum_inspired_state"] = {"f1": 0.5, "f2": 0.5}
        audit = validate_transition_stream([record])
        self.assertEqual(audit["invalid_events"], 1)
        self.assertIn(
            "quantum_amplitudes_do_not_encode_future_probabilities",
            audit["failures"][0]["errors"],
        )

    def test_nonfinite_state_fails_closed(self) -> None:
        record = complete_event().to_record()
        record["master_state_after"] = [float("nan")]
        self.assertIn(
            "invalid_finite_state_vector:master_state_after",
            transition_validation_errors(record),
        )

    def test_reconstruction_and_oosc_keep_dimensions_visible(self) -> None:
        state = np.array([1.0, 2.0])
        self.assertEqual(normalized_reconstructability(state, state), 1.0)
        profile = oosc_profile(state, state, state, state)
        self.assertEqual(
            set(profile),
            {"cancer_agent", "cancer_master", "cancer_meta", "agent_master", "master_meta"},
        )


if __name__ == "__main__":
    unittest.main()
