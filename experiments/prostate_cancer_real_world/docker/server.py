from __future__ import annotations

import json
import os
import pickle
import time
import zlib
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from prostate_flip.policies import PatternState, build_policy


CONFIG_ROOT = Path("/app/config")
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "/app/results"))
METHOD = os.getenv("METHOD", "fliponline")
NETWORK_CONDITION = os.getenv("NETWORK_CONDITION", "unstable")
ROUNDS = int(os.getenv("ROUNDS", "20"))
REPETITION = int(os.getenv("REPETITION", "1"))
RUN_ID = os.getenv("RUN_ID", f"{NETWORK_CONDITION}_{METHOD}_rep{REPETITION}")
CLIENT_COUNT = 11


def serialize_parameters(parameters: list[np.ndarray], compressed: bool) -> bytes:
    raw = pickle.dumps(parameters, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, level=1) if compressed else raw


def weighted_metric(rows: list[dict[str, Any]], metric: str) -> float:
    weights = np.asarray([row["test_examples"] for row in rows], dtype=float)
    return float(np.average([row[metric] for row in rows], weights=weights))


class ProstateStrategy(FedAvg):
    def __init__(self) -> None:
        self.config = json.loads((CONFIG_ROOT / "experiment.json").read_text())
        self.clients_metadata = json.loads((CONFIG_ROOT / "clients.json").read_text())
        if METHOD not in self.config["policy"]["methods"]:
            raise ValueError(f"Unsupported method: {METHOD}")
        if NETWORK_CONDITION not in self.config["network"]["conditions"]:
            raise ValueError(f"Unsupported network condition: {NETWORK_CONDITION}")

        initial_ndarrays = [np.zeros((1, 8), dtype=np.float64), np.zeros(1)]
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=CLIENT_COUNT,
            min_evaluate_clients=CLIENT_COUNT,
            min_available_clients=CLIENT_COUNT,
            accept_failures=False,
            initial_parameters=ndarrays_to_parameters(initial_ndarrays),
        )
        self.policy = build_policy(
            METHOD,
            self.clients_metadata,
            alpha=float(self.config["policy"]["linucb_alpha"]),
            seed=int(self.config["seed"]) + 40_000 + REPETITION - 1,
        )
        self.state: PatternState = self.policy.initial_state()
        self.metrics_history = {
            "ProstateLR": {
                "val_auc": [],
                "val_f1": [],
                "val_accuracy": [],
                "val_loss": [],
                "jsd": [],
            }
        }
        self.round_started = 0.0
        self.pending_fit_rows: dict[int, dict[str, Any]] = {}
        self.pending_communication_seconds = 0.0
        self.pending_fit_wire_bytes = 0
        self.round_rows: list[dict[str, Any]] = []
        self.client_rows: list[dict[str, Any]] = []
        self.local_rows: list[dict[str, Any]] = []
        self.cumulative_time = 0.0
        self.cumulative_f1 = 0.0
        self.cumulative_communication = 0.0
        self.cumulative_wire_bytes = 0
        self.output_dir = RESULTS_ROOT / RUN_ID
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": RUN_ID,
            "method": METHOD,
            "network": NETWORK_CONDITION,
            "rounds": ROUNDS,
            "repetition": REPETITION,
            "client_count": CLIENT_COUNT,
            "execution": "Flower server plus 11 isolated Docker client containers",
            "actual_network_sleep": NETWORK_CONDITION == "unstable",
            "clients": self.clients_metadata,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    def _fit_instruction(self, parameters: Parameters, server_round: int) -> FitIns:
        ndarrays = parameters_to_ndarrays(parameters)
        compressed = self.state.compressor
        config: dict[str, Any] = {
            "server_round": server_round,
            "network_condition": NETWORK_CONDITION,
            "client_selector": self.state.selector,
            "cpu_threshold": int(
                self.config["patterns"]["client_selector"]["cpu_threshold"]
            ),
            "hdh_clients": ",".join(
                str(index + 1) for index in sorted(self.state.hdh_clients)
            ),
            "compressed": compressed,
        }
        if compressed:
            config["compressed_parameters"] = serialize_parameters(
                ndarrays, compressed=True
            )
            parameters = Parameters(tensors=[], tensor_type="compressed-zlib")
        return FitIns(parameters, config)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        client_manager.wait_for(CLIENT_COUNT)
        clients = client_manager.sample(
            num_clients=CLIENT_COUNT, min_num_clients=CLIENT_COUNT
        )
        self.round_started = time.perf_counter()
        instruction = self._fit_instruction(parameters, server_round)
        return [(client, instruction) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Any]]:
        if failures:
            raise RuntimeError(f"Fit failures in round {server_round}: {failures}")
        if len(results) != CLIENT_COUNT:
            raise RuntimeError(
                f"Expected {CLIENT_COUNT} fit responses, received {len(results)}."
            )

        aggregation_input: list[tuple[list[np.ndarray], int]] = []
        fit_rows: dict[int, dict[str, Any]] = {}
        for _, fit_result in results:
            metrics = dict(fit_result.metrics)
            client_id = int(metrics["client_id"])
            fit_rows[client_id] = metrics
            if fit_result.num_examples <= 0:
                continue
            if "compressed_update" in metrics:
                raw = zlib.decompress(bytes(metrics.pop("compressed_update")))
                ndarrays = [
                    np.asarray(value, dtype=np.float64) for value in pickle.loads(raw)
                ]
            else:
                ndarrays = parameters_to_ndarrays(fit_result.parameters)
            aggregation_input.append((ndarrays, fit_result.num_examples))

        if not aggregation_input:
            raise RuntimeError(f"No client trained in round {server_round}.")
        aggregated = ndarrays_to_parameters(aggregate(aggregation_input))
        self.pending_fit_rows = fit_rows
        self.pending_communication_seconds = max(
            float(row["communication_seconds"]) for row in fit_rows.values()
        )
        self.pending_fit_wire_bytes = sum(
            int(row["download_wire_bytes"]) + int(row["upload_wire_bytes"])
            for row in fit_rows.values()
        )
        return aggregated, {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        client_manager.wait_for(CLIENT_COUNT)
        clients = client_manager.sample(
            num_clients=CLIENT_COUNT, min_num_clients=CLIENT_COUNT
        )
        ndarrays = parameters_to_ndarrays(parameters)
        config: dict[str, Any] = {"compressed": self.state.compressor}
        if self.state.compressor:
            config["compressed_parameters"] = serialize_parameters(
                ndarrays, compressed=True
            )
            parameters = Parameters(tensors=[], tensor_type="compressed-zlib")
        instruction = EvaluateIns(parameters, config)
        return [(client, instruction) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Any]]:
        if failures:
            raise RuntimeError(
                f"Evaluation failures in round {server_round}: {failures}"
            )
        if len(results) != CLIENT_COUNT:
            raise RuntimeError(
                f"Expected {CLIENT_COUNT} evaluation responses, received {len(results)}."
            )

        evaluation_rows = [dict(result.metrics) for _, result in results]
        evaluation_rows.sort(key=lambda row: int(row["client_id"]))
        global_metrics = {
            metric: weighted_metric(evaluation_rows, metric)
            for metric in ("auc", "f1", "accuracy", "loss")
        }
        history = self.metrics_history["ProstateLR"]
        history["val_auc"].append(global_metrics["auc"])
        history["val_f1"].append(global_metrics["f1"])
        history["val_accuracy"].append(global_metrics["accuracy"])
        history["val_loss"].append(global_metrics["loss"])
        history["jsd"].append(
            tuple(
                float(self.pending_fit_rows[client_id]["jsd"])
                for client_id in range(1, CLIENT_COUNT + 1)
            )
        )

        round_seconds = time.perf_counter() - self.round_started
        evaluation_wire_bytes = sum(
            int(row["download_wire_bytes"]) for row in evaluation_rows
        )
        round_wire_bytes = self.pending_fit_wire_bytes + evaluation_wire_bytes
        self.cumulative_time += round_seconds
        self.cumulative_f1 += global_metrics["f1"]
        self.cumulative_communication += self.pending_communication_seconds
        self.cumulative_wire_bytes += round_wire_bytes
        current_state = self.state.as_record()
        next_state, explanations = self.policy.next_state(
            self.metrics_history,
            round_seconds,
            self.pending_communication_seconds,
        )

        evaluation_by_id = {int(row["client_id"]): row for row in evaluation_rows}
        for client_id in range(1, CLIENT_COUNT + 1):
            fit_row = self.pending_fit_rows[client_id]
            evaluation_row = evaluation_by_id[client_id]
            combined = {
                "method": METHOD,
                "network": NETWORK_CONDITION,
                "repetition": REPETITION,
                "round": server_round,
                **fit_row,
                "test_examples": int(evaluation_row["test_examples"]),
                "auc": float(evaluation_row["auc"]),
                "f1": float(evaluation_row["f1"]),
                "accuracy": float(evaluation_row["accuracy"]),
                "loss": float(evaluation_row["loss"]),
                "evaluation_download_wire_bytes": int(
                    evaluation_row["download_wire_bytes"]
                ),
            }
            self.client_rows.append(combined)
            self.local_rows.append(
                {
                    "method": METHOD,
                    "network": NETWORK_CONDITION,
                    "repetition": REPETITION,
                    "round": server_round,
                    "client_id": client_id,
                    "silo": evaluation_row["silo"],
                    "code": evaluation_row["code"],
                    "test_examples": int(evaluation_row["test_examples"]),
                    "auc": float(evaluation_row["auc"]),
                    "f1": float(evaluation_row["f1"]),
                    "accuracy": float(evaluation_row["accuracy"]),
                    "loss": float(evaluation_row["loss"]),
                }
            )

        self.round_rows.append(
            {
                "method": METHOD,
                "network": NETWORK_CONDITION,
                "repetition": REPETITION,
                "round": server_round,
                "selected_clients": sum(
                    bool(row["selected"]) for row in self.pending_fit_rows.values()
                ),
                **current_state,
                **global_metrics,
                "round_seconds": round_seconds,
                "communication_seconds": self.pending_communication_seconds,
                "wire_bytes": round_wire_bytes,
                "cumulative_time_seconds": self.cumulative_time,
                "cumulative_f1": self.cumulative_f1,
                "cumulative_communication_seconds": self.cumulative_communication,
                "cumulative_wire_bytes": self.cumulative_wire_bytes,
                "y1_f1_over_cumulative_time": global_metrics["f1"]
                / self.cumulative_time,
                "y2_cumulative_f1": self.cumulative_f1,
                "y3_cumulative_communication": self.cumulative_communication,
                "next_round_explanation": " | ".join(explanations),
            }
        )
        self.state = next_state
        self._persist(server_round)
        return global_metrics["loss"], {
            "auc": global_metrics["auc"],
            "f1": global_metrics["f1"],
            "accuracy": global_metrics["accuracy"],
        }

    def _persist(self, server_round: int) -> None:
        pd.DataFrame(self.round_rows).to_csv(
            self.output_dir / "round_metrics.csv", index=False
        )
        pd.DataFrame(self.client_rows).to_csv(
            self.output_dir / "client_round_metrics.csv", index=False
        )
        pd.DataFrame(self.local_rows).to_csv(
            self.output_dir / "local_metrics.csv", index=False
        )
        if server_round != ROUNDS:
            return
        final = self.round_rows[-1]
        summary = {
            "method": METHOD,
            "network": NETWORK_CONDITION,
            "rounds": ROUNDS,
            "repetitions": 1,
            "final_auc": final["auc"],
            "final_f1": final["f1"],
            "final_accuracy": final["accuracy"],
            "final_loss": final["loss"],
            "total_time_seconds": final["cumulative_time_seconds"],
            "total_communication_seconds": final["cumulative_communication_seconds"],
            "total_wire_bytes": int(final["cumulative_wire_bytes"]),
            "y1_f1_over_cumulative_time": final["y1_f1_over_cumulative_time"],
            "y2_cumulative_f1": final["y2_cumulative_f1"],
            "y3_cumulative_communication": final["y3_cumulative_communication"],
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


if __name__ == "__main__":
    strategy = ProstateStrategy()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
