#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from prostate_flip.analysis import create_plots  # noqa: E402
from prostate_flip.data import ensure_data, load_clients  # noqa: E402
from prostate_flip.simulation import run_scenario  # noqa: E402


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-world FLiP campaign.")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument(
        "--methods", type=_csv_list, default=["never", "random", "always", "fliponline"]
    )
    parser.add_argument("--networks", type=_csv_list, default=["stable", "unstable"])
    parser.add_argument(
        "--data-path", type=Path, default=EXPERIMENT_ROOT / "data" / "data.pkl"
    )
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = json.loads((EXPERIMENT_ROOT / "config" / "experiment.json").read_text())
    for method in args.methods:
        if method not in config["policy"]["methods"]:
            raise ValueError(f"Unknown method: {method}")
    for network in args.networks:
        if network not in config["network"]["conditions"]:
            raise ValueError(f"Unknown network condition: {network}")

    clients_path = EXPERIMENT_ROOT / "config" / "clients.json"
    ensure_data(args.data_path, config["data"]["url"], config["data"]["sha256"])
    clients, split_random_state = load_clients(args.data_path, clients_path, config)
    clients_metadata = json.loads(clients_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    summaries = []
    for network in args.networks:
        for method in args.methods:
            scenario_name = f"{network}_{method}"
            print(f"Running {scenario_name}...")
            summaries.append(
                run_scenario(
                    config,
                    clients_metadata,
                    clients,
                    method=method,
                    network=network,
                    output_dir=args.output / scenario_name,
                    rounds_override=args.rounds,
                )
            )

    summary_frame = pd.DataFrame(summaries)
    baseline = summary_frame[summary_frame["method"] == "never"][
        [
            "network",
            "final_auc",
            "final_f1",
            "y1_f1_over_cumulative_time",
            "y2_cumulative_f1",
            "y3_cumulative_communication",
            "total_time_seconds",
            "total_wire_bytes",
        ]
    ].rename(
        columns={
            column: f"never_{column}"
            for column in [
                "final_auc",
                "final_f1",
                "y1_f1_over_cumulative_time",
                "y2_cumulative_f1",
                "y3_cumulative_communication",
                "total_time_seconds",
                "total_wire_bytes",
            ]
        }
    )
    summary_frame = summary_frame.merge(baseline, on="network", how="left")
    summary_frame["delta_auc_vs_never"] = (
        summary_frame["final_auc"] - summary_frame["never_final_auc"]
    )
    summary_frame["delta_f1_vs_never"] = (
        summary_frame["final_f1"] - summary_frame["never_final_f1"]
    )
    summary_frame["y1_gain_vs_never_pct"] = 100 * (
        summary_frame["y1_f1_over_cumulative_time"]
        / summary_frame["never_y1_f1_over_cumulative_time"]
        - 1
    )
    summary_frame["y2_gain_vs_never_pct"] = 100 * (
        summary_frame["y2_cumulative_f1"] / summary_frame["never_y2_cumulative_f1"] - 1
    )
    summary_frame["communication_reduction_vs_never_pct"] = 100 * (
        1
        - summary_frame["y3_cumulative_communication"]
        / summary_frame["never_y3_cumulative_communication"]
    )
    summary_frame["time_reduction_vs_never_pct"] = 100 * (
        1
        - summary_frame["total_time_seconds"]
        / summary_frame["never_total_time_seconds"]
    )
    summary_frame["wire_reduction_vs_never_pct"] = 100 * (
        1 - summary_frame["total_wire_bytes"] / summary_frame["never_total_wire_bytes"]
    )
    summary_frame.to_csv(args.output / "campaign_summary.csv", index=False)

    round_frames = []
    client_frames = []
    for network in args.networks:
        for method in args.methods:
            scenario_dir = args.output / f"{network}_{method}"
            round_frames.append(pd.read_csv(scenario_dir / "round_metrics.csv"))
            client_frames.append(pd.read_csv(scenario_dir / "client_round_metrics.csv"))
    combined_rounds = pd.concat(round_frames, ignore_index=True)
    combined_clients = pd.concat(client_frames, ignore_index=True)
    combined_rounds.to_csv(args.output / "all_round_metrics.csv", index=False)
    combined_clients.to_csv(args.output / "all_client_round_metrics.csv", index=False)
    create_plots(summary_frame, combined_rounds, args.output)

    manifest = {
        "split_random_state": split_random_state,
        "rounds": int(args.rounds or config["rounds"]),
        "repetitions": 1,
        "methods": args.methods,
        "networks": args.networks,
        "silos": [client.silo for client in clients],
        "data_sha256": config["data"]["sha256"],
    }
    (args.output / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        summary_frame[
            [
                "method",
                "network",
                "final_auc",
                "final_f1",
                "delta_auc_vs_never",
                "y1_gain_vs_never_pct",
                "time_reduction_vs_never_pct",
            ]
        ]
    )


if __name__ == "__main__":
    main()
