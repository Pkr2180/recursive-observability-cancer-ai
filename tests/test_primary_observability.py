from __future__ import annotations

import unittest

from src.observability.primary import (
    _empirical_rank_ratio,
    _hypothesis_traceability,
    _quantum_and_gate_observability,
)


def event(index: int, episode: str, master_after: list[float]) -> dict:
    return {
        "_event_index": index,
        "episode_id": episode,
        "agent_id": "agent-a",
        "agent_state_after": master_after,
        "master_state_before": [0.0] * len(master_after),
        "master_state_after": master_after,
        "gate_applied": "branch_gate",
        "hypothesis_before": "H0",
        "hypothesis_after": "H1",
        "action": "integrate",
        "evidence_ids": ["source:1"],
        "future_states_considered": {"f1": 0.25, "f2": 0.75},
        "quantum_inspired_state": {"f1": 0.5, "f2": 0.75**0.5},
    }


class PrimaryObservabilityTests(unittest.TestCase):
    def test_empirical_rank_ratio_recovers_visible_dimensions(self) -> None:
        events = [
            event(0, "e1", [1.0, 0.0]),
            event(1, "e2", [0.0, 1.0]),
            event(2, "e3", [-1.0, -1.0]),
        ]
        profile = _empirical_rank_ratio(events)
        self.assertEqual(profile["empirical_master_state_telemetry_rank"], 2)
        self.assertEqual(profile["observability_rank_ratio"], 1.0)

    def test_hypothesis_quantum_and_gate_outputs_are_observable(self) -> None:
        events = [event(0, "e1", [0.5]), event(1, "e1", [0.7])]
        graph, htt = _hypothesis_traceability(events)
        gate, quantum = _quantum_and_gate_observability(events)
        self.assertEqual(htt, 1.0)
        self.assertEqual(int(graph["count"].sum()), 2)
        self.assertEqual(quantum["quantum_state_observability"], 1.0)
        self.assertEqual(quantum["normalized_amplitude_rate"], 1.0)
        self.assertEqual(len(gate), 2)


if __name__ == "__main__":
    unittest.main()
