#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Docker experiment run.")
    parser.add_argument("run_id")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = EXPERIMENT_ROOT / "docker_results" / args.run_id
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    rounds = pd.read_csv(output_dir / "round_metrics.csv")
    clients = pd.read_csv(output_dir / "client_round_metrics.csv")
    local = pd.read_csv(output_dir / "local_metrics.csv")

    expected_rounds = int(manifest["rounds"])
    assert manifest["client_count"] == 11
    assert manifest["execution"].startswith("Flower server plus 11")
    assert rounds["round"].nunique() == expected_rounds
    assert len(rounds) == expected_rounds
    assert len(clients) == expected_rounds * 11
    assert len(local) == expected_rounds * 11
    assert clients.groupby("round")["client_id"].nunique().eq(11).all()
    assert local.groupby("round")["client_id"].nunique().eq(11).all()
    assert summary["rounds"] == expected_rounds
    if manifest["network"] == "stable":
        assert clients["delay_seconds"].eq(0).all()
    else:
        selected = clients[clients["selected"]]
        assert selected["delay_seconds"].between(0, 20).all()
        assert (
            selected["delay_seconds"] == selected["delay_seconds"].astype(int)
        ).all()
    print(
        f"Validated {args.run_id}: {expected_rounds} rounds, "
        f"{len(clients)} client-rounds, 11 isolated clients."
    )


if __name__ == "__main__":
    main()
