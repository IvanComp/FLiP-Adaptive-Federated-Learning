#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from prostate_flip.data import ensure_data, load_clients  # noqa: E402
from prostate_flip.simulation import run_scenario  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real-world FLiP scenario.")
    parser.add_argument(
        "--method",
        choices=["never", "random", "always", "fliponline"],
        default="fliponline",
    )
    parser.add_argument("--network", choices=["stable", "unstable"], default="unstable")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument(
        "--data-path", type=Path, default=EXPERIMENT_ROOT / "data" / "data.pkl"
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = json.loads((EXPERIMENT_ROOT / "config" / "experiment.json").read_text())
    clients_path = EXPERIMENT_ROOT / "config" / "clients.json"
    ensure_data(args.data_path, config["data"]["url"], config["data"]["sha256"])
    clients, split_random_state = load_clients(args.data_path, clients_path, config)
    clients_metadata = json.loads(clients_path.read_text())
    output = (
        args.output or EXPERIMENT_ROOT / "results" / f"{args.network}_{args.method}"
    )
    summary = run_scenario(
        config,
        clients_metadata,
        clients,
        method=args.method,
        network=args.network,
        output_dir=output,
        rounds_override=args.rounds,
    )
    print(f"Split random state: {split_random_state}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
