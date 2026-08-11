#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = EXPERIMENT_ROOT / "docker-compose.yml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real 11-container Flower/FLiP clinical experiment."
    )
    parser.add_argument(
        "--method",
        choices=["never", "random", "always", "fliponline"],
        default="fliponline",
    )
    parser.add_argument("--network", choices=["stable", "unstable"], default="unstable")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--keep-containers", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=EXPERIMENT_ROOT, env=env, check=True)


def main() -> None:
    args = parse_arguments()
    if args.rounds < 1 or args.repetition < 1:
        raise ValueError("Rounds and repetition must be positive integers.")

    run([sys.executable, str(EXPERIMENT_ROOT / "prepare_docker_data.py")])
    run_id = f"{args.network}_{args.method}_rep{args.repetition}"
    output_dir = EXPERIMENT_ROOT / "docker_results" / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        if (output_dir / "summary.json").exists():
            raise FileExistsError(
                f"Refusing to overwrite completed Docker result: {output_dir}"
            )
        failed_root = EXPERIMENT_ROOT / "docker_results" / "failed_runs"
        failed_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failed_output = failed_root / f"{run_id}_{timestamp}"
        shutil.move(str(output_dir), failed_output)
        print(f"Archived incomplete result in {failed_output}.")

    env = os.environ.copy()
    env.update(
        {
            "METHOD": args.method,
            "NETWORK_CONDITION": args.network,
            "ROUNDS": str(args.rounds),
            "REPETITION": str(args.repetition),
            "RUN_ID": run_id,
        }
    )
    compose = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    run([*compose, "down", "--remove-orphans"], env=env)
    try:
        command = [*compose, "up", "--exit-code-from", "server"]
        if not args.no_build:
            command.append("--build")
        run(command, env=env)
    finally:
        if not args.keep_containers:
            run([*compose, "down", "--remove-orphans"], env=env)

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Experiment ended without a summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
