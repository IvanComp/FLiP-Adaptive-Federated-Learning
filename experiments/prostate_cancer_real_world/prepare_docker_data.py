#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from prostate_flip.data import ensure_data, load_clients  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one isolated clinical dataset per Docker client."
    )
    parser.add_argument(
        "--data-path", type=Path, default=EXPERIMENT_ROOT / "data" / "data.pkl"
    )
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT_ROOT / "data" / "clients"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = json.loads((EXPERIMENT_ROOT / "config" / "experiment.json").read_text())
    clients_path = EXPERIMENT_ROOT / "config" / "clients.json"
    metadata = json.loads(clients_path.read_text())

    ensure_data(args.data_path, config["data"]["url"], config["data"]["sha256"])
    clients, split_random_state = load_clients(args.data_path, clients_path, config)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_clients = []
    for client, details in zip(clients, metadata, strict=True):
        destination = args.output / f"{client.code}.npz"
        np.savez_compressed(
            destination,
            X_train=client.X_train,
            X_test=client.X_test,
            y_train=client.y_train,
            y_test=client.y_test,
        )
        manifest_clients.append(
            {
                **details,
                "train_patients": int(client.y_train.size),
                "test_patients": int(client.y_test.size),
                "file": destination.name,
            }
        )

    manifest = {
        "source_sha256": config["data"]["sha256"],
        "seed": config["seed"],
        "split_random_state": split_random_state,
        "clients": manifest_clients,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {len(clients)} isolated client files with "
        f"{sum(client.patient_count for client in clients)} patients in {args.output}."
    )


if __name__ == "__main__":
    main()
