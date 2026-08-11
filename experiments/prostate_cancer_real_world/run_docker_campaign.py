#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = EXPERIMENT_ROOT / "docker-compose.yml"
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from prostate_flip.analysis import create_plots  # noqa: E402


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete real 11-container clinical campaign."
    )
    parser.add_argument(
        "--methods",
        type=csv_list,
        default=["never", "random", "always", "fliponline"],
    )
    parser.add_argument("--networks", type=csv_list, default=["stable", "unstable"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=EXPERIMENT_ROOT, check=True)


def main() -> None:
    args = parse_arguments()
    allowed_methods = {"never", "random", "always", "fliponline"}
    allowed_networks = {"stable", "unstable"}
    if not set(args.methods) <= allowed_methods:
        raise ValueError(f"Unsupported methods: {set(args.methods) - allowed_methods}")
    if not set(args.networks) <= allowed_networks:
        raise ValueError(
            f"Unsupported networks: {set(args.networks) - allowed_networks}"
        )

    if not args.no_build:
        run(["docker", "compose", "-f", str(COMPOSE_FILE), "build", "server"])

    output_root = EXPERIMENT_ROOT / "docker_results"
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_dirs: list[Path] = []
    for network in args.networks:
        for method in args.methods:
            run_id = f"{network}_{method}_rep{args.repetition}"
            scenario_dir = output_root / run_id
            summary_path = scenario_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if int(summary["rounds"]) != args.rounds:
                    raise RuntimeError(
                        f"{run_id} exists with {summary['rounds']} rounds, "
                        f"but {args.rounds} were requested."
                    )
                print(f"Skipping completed scenario {run_id}.")
            else:
                run(
                    [
                        sys.executable,
                        str(EXPERIMENT_ROOT / "run_docker_experiment.py"),
                        "--method",
                        method,
                        "--network",
                        network,
                        "--rounds",
                        str(args.rounds),
                        "--repetition",
                        str(args.repetition),
                        "--no-build",
                    ]
                )
            scenario_dirs.append(scenario_dir)

    summaries = [
        json.loads((path / "summary.json").read_text()) for path in scenario_dirs
    ]
    summary_frame = pd.DataFrame(summaries)
    has_baseline = "never" in set(summary_frame["method"])
    if has_baseline:
        baseline = summary_frame[summary_frame["method"] == "never"][
            [
                "network",
                "final_auc",
                "final_f1",
                "y1_f1_over_cumulative_time",
                "total_time_seconds",
                "total_wire_bytes",
            ]
        ].rename(
            columns={
                column: f"never_{column}"
                for column in (
                    "final_auc",
                    "final_f1",
                    "y1_f1_over_cumulative_time",
                    "total_time_seconds",
                    "total_wire_bytes",
                )
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
        summary_frame["time_reduction_vs_never_pct"] = 100 * (
            1
            - summary_frame["total_time_seconds"]
            / summary_frame["never_total_time_seconds"]
        )
        summary_frame["wire_reduction_vs_never_pct"] = 100 * (
            1
            - summary_frame["total_wire_bytes"]
            / summary_frame["never_total_wire_bytes"]
        )

    summary_frame.to_csv(output_root / "campaign_summary.csv", index=False)
    round_frame = pd.concat(
        [pd.read_csv(path / "round_metrics.csv") for path in scenario_dirs],
        ignore_index=True,
    )
    client_frame = pd.concat(
        [pd.read_csv(path / "client_round_metrics.csv") for path in scenario_dirs],
        ignore_index=True,
    )
    round_frame.to_csv(output_root / "all_round_metrics.csv", index=False)
    client_frame.to_csv(output_root / "all_client_round_metrics.csv", index=False)
    if has_baseline:
        create_plots(summary_frame, round_frame, output_root)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
