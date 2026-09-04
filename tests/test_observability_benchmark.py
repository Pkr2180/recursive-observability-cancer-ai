from __future__ import annotations

import unittest

from src.observability.benchmark import benchmark_metrics, disrupt_events
from tests.test_recursive_observability import complete_event


def events() -> list[dict]:
    rows = []
    for episode in range(6):
        for agent in range(3):
            row = complete_event().to_record()
            row["episode_id"] = f"e{episode}"
            row["agent_id"] = f"a{agent}"
            row["agent_state_after"] = [float(episode + agent), float(agent + 1)]
            row["agent_state_before"] = [float(episode), float(agent)]
            row["master_state_after"] = [float(episode), float(episode + 1)]
            row["master_state_before"] = [float(episode - 1), float(episode)]
            row["agent_uncertainty"] = 0.1 + 0.01 * agent
            row["master_uncertainty"] = 0.2 + 0.02 * episode
            row["meta_uncertainty"] = 0.3 + 0.03 * episode
            rows.append(row)
    return rows


class ObservabilityBenchmarkTests(unittest.TestCase):
    def test_hidden_provenance_fails_closed_monotonically(self) -> None:
        original = events()
        partial = benchmark_metrics(disrupt_events(original, "hidden_provenance", 0.5), original)
        complete = benchmark_metrics(disrupt_events(original, "hidden_provenance", 1.0), original)
        self.assertGreater(partial["strict_validity"], complete["strict_validity"])
        self.assertEqual(complete["strict_validity"], 0.0)

    def test_agent_collapse_removes_identifiability(self) -> None:
        original = events()
        collapsed = benchmark_metrics(disrupt_events(original, "collapsed_agents", 1.0), original)
        self.assertAlmostEqual(collapsed["agent_identifiability"], 0.0)

    def test_master_corruption_reduces_fidelity(self) -> None:
        original = events()
        corrupted = benchmark_metrics(disrupt_events(original, "corrupted_master_state", 1.0), original)
        self.assertLess(corrupted["master_fidelity"], 1.0)


if __name__ == "__main__":
    unittest.main()
