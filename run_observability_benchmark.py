from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.observability.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recursive scientific observability benchmark.")
    parser.add_argument("--input-root", type=Path, default=Path("retrieved_results_2026-08-25/results"))
    parser.add_argument("--output-root", type=Path, default=Path("final_results/recursive_observability_benchmark"))
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.input_root, args.output_root, args.seed), indent=2))


if __name__ == "__main__":
    main()

