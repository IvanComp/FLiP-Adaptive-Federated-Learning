#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = json.loads((EXPERIMENT_ROOT / "config" / "experiment.json").read_text())
    clients = json.loads((EXPERIMENT_ROOT / "config" / "clients.json").read_text())
    results = EXPERIMENT_ROOT / "results"
    manifest = json.loads((results / "campaign_manifest.json").read_text())
    rounds = pd.read_csv(results / "all_round_metrics.csv")
    client_rounds = pd.read_csv(results / "all_client_round_metrics.csv")
    summary = pd.read_csv(results / "campaign_summary.csv")

    scenario_count = len(config["policy"]["methods"]) * len(
        config["network"]["conditions"]
    )
    expected_round_rows = scenario_count * config["rounds"]
    expected_client_rows = expected_round_rows * len(clients)
    assert len(summary) == scenario_count
    assert len(rounds) == expected_round_rows
    assert len(client_rounds) == expected_client_rows
    assert manifest["rounds"] == config["rounds"]
    assert manifest["repetitions"] == config["repetitions"] == 1
    assert manifest["data_sha256"] == config["data"]["sha256"]
    assert manifest["silos"] == [client["silo"] for client in clients]

    scenario_round_counts = rounds.groupby(["network", "method"])["round"].nunique()
    assert (scenario_round_counts == config["rounds"]).all()
    assert rounds["auc"].between(0, 1).all()
    assert rounds["f1"].between(0, 1).all()
    assert rounds["accuracy"].between(0, 1).all()
    assert summary["final_auc"].between(0, 1).all()

    stable_delays = client_rounds.loc[
        client_rounds["network"] == "stable", "network_delay_seconds"
    ]
    assert (stable_delays == 0).all()
    unstable_selected = client_rounds[
        (client_rounds["network"] == "unstable") & client_rounds["selected"]
    ]
    delay_config = config["network"]["unstable_delay_seconds"]
    assert (
        unstable_selected["network_delay_seconds"]
        .between(delay_config["minimum"], delay_config["maximum"])
        .all()
    )
    assert (
        unstable_selected["network_delay_seconds"]
        == unstable_selected["network_delay_seconds"].astype(int)
    ).all()

    required_outputs = [
        "final_auc.png",
        "auc_over_rounds.png",
        "relative_tradeoffs.png",
    ]
    assert all((results / filename).stat().st_size > 0 for filename in required_outputs)
    print(
        f"Validated {scenario_count} scenarios, {len(rounds)} round rows, "
        f"and {len(client_rounds)} client-round rows."
    )


if __name__ == "__main__":
    main()
