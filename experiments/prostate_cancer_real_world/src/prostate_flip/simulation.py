from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from .data import SiloDataset
from .patterns import label_jsd, random_minority_oversample, transmit_parameters
from .policies import build_policy


@dataclass(frozen=True)
class Parameters:
    coefficient: np.ndarray
    intercept: np.ndarray


def _fit_local_model(
    initial: Parameters,
    features: np.ndarray,
    labels: np.ndarray,
    model_config: dict[str, Any],
) -> tuple[Parameters, int]:
    model = LogisticRegression(
        penalty=model_config["penalty"],
        C=float(model_config["C"]),
        solver=model_config["solver"],
        class_weight=model_config["class_weight"],
        fit_intercept=True,
        tol=float(model_config["tolerance"]),
        max_iter=int(model_config["local_max_iter_per_round"]),
        warm_start=True,
    )
    model.classes_ = np.array([0, 1], dtype=np.int64)
    model.coef_ = initial.coefficient.copy()
    model.intercept_ = initial.intercept.copy()
    model.n_features_in_ = features.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        model.fit(features, labels)
    return Parameters(model.coef_.copy(), model.intercept_.copy()), int(
        model.n_iter_[0]
    )


def _aggregate(updates: list[tuple[Parameters, int]]) -> Parameters:
    total_examples = sum(example_count for _, example_count in updates)
    if total_examples <= 0:
        raise ValueError("FedAvg received no examples.")
    coefficient = sum(
        parameters.coefficient * (example_count / total_examples)
        for parameters, example_count in updates
    )
    intercept = sum(
        parameters.intercept * (example_count / total_examples)
        for parameters, example_count in updates
    )
    return Parameters(coefficient, intercept)


def _evaluate(parameters: Parameters, clients: list[SiloDataset]):
    rows: list[dict[str, Any]] = []
    for client in clients:
        probabilities = expit(
            client.X_test @ parameters.coefficient.reshape(-1)
            + float(parameters.intercept[0])
        )
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            {
                "client_id": client.client_id,
                "silo": client.silo,
                "code": client.code,
                "test_examples": int(client.y_test.size),
                "auc": float(roc_auc_score(client.y_test, probabilities)),
                "f1": float(f1_score(client.y_test, predictions, average="macro")),
                "accuracy": float(accuracy_score(client.y_test, predictions)),
                "loss": float(log_loss(client.y_test, probabilities, labels=[0, 1])),
            }
        )
    weights = np.array([row["test_examples"] for row in rows], dtype=float)
    aggregates = {
        metric: float(np.average([row[metric] for row in rows], weights=weights))
        for metric in ("auc", "f1", "accuracy", "loss")
    }
    return rows, aggregates


def _network_delays(
    config: dict[str, Any], network: str, client_count: int, rounds: int
) -> np.ndarray:
    if network == "stable":
        return np.zeros((rounds, client_count), dtype=int)
    if network != "unstable":
        raise ValueError(f"Unknown network condition: {network}")
    delay_config = config["network"]["unstable_delay_seconds"]
    rng = np.random.default_rng(config["seed"] + 20_026)
    return rng.integers(
        int(delay_config["minimum"]),
        int(delay_config["maximum"]) + 1,
        size=(rounds, client_count),
    )


def run_scenario(
    config: dict[str, Any],
    clients_metadata: list[dict[str, Any]],
    clients: list[SiloDataset],
    method: str,
    network: str,
    output_dir: Path,
    rounds_override: int | None = None,
) -> dict[str, Any]:
    rounds = int(rounds_override or config["rounds"])
    if method not in config["policy"]["methods"]:
        raise ValueError(f"Method {method!r} is not configured.")
    if network not in config["network"]["conditions"]:
        raise ValueError(f"Network {network!r} is not configured.")
    if len(clients) != len(clients_metadata):
        raise ValueError("Client data and metadata lengths differ.")

    output_dir.mkdir(parents=True, exist_ok=True)
    delays = _network_delays(config, network, len(clients), rounds)
    policy = build_policy(
        method,
        clients_metadata,
        alpha=float(config["policy"]["linucb_alpha"]),
        seed=config["seed"] + 40_000,
    )
    state = policy.initial_state()
    parameter_count = clients[0].X_train.shape[1]
    global_parameters = Parameters(
        coefficient=np.zeros((1, parameter_count), dtype=np.float64),
        intercept=np.zeros(1, dtype=np.float64),
    )
    metrics_history = {
        "ProstateLR": {
            "val_auc": [],
            "val_f1": [],
            "val_accuracy": [],
            "val_loss": [],
            "jsd": [],
        }
    }
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    local_auc_rows: list[dict[str, Any]] = []
    cumulative_time = 0.0
    cumulative_f1 = 0.0
    cumulative_communication = 0.0
    cumulative_wire_bytes = 0

    for round_index in range(rounds):
        round_number = round_index + 1
        updates: list[tuple[Parameters, int]] = []
        round_client_durations: list[float] = []
        round_communications: list[float] = []
        round_wire_bytes = 0
        effective_jsds: list[float] = []
        selected_count = 0
        state_record = state.as_record()

        for client_index, client in enumerate(clients):
            selected = not state.selector or (
                client.cpu >= config["patterns"]["client_selector"]["cpu_threshold"]
            )
            hdh_enabled = client_index in state.hdh_clients
            feature_data = client.X_train
            label_data = client.y_train
            hdh_seconds = 0.0
            if hdh_enabled:
                hdh_started = time.perf_counter()
                oversampling_rng = np.random.default_rng(
                    config["seed"] + 100_000 * round_number + client.client_id
                )
                feature_data, label_data = random_minority_oversample(
                    feature_data, label_data, oversampling_rng
                )
                hdh_seconds = time.perf_counter() - hdh_started
            effective_jsds.append(label_jsd(label_data))

            if not selected:
                client_rows.append(
                    {
                        "method": method,
                        "network": network,
                        "round": round_number,
                        "client_id": client.client_id,
                        "silo": client.silo,
                        "code": client.code,
                        "cpu": client.cpu,
                        "selected": False,
                        "hdh_enabled": hdh_enabled,
                        "train_examples": 0,
                        "jsd": effective_jsds[-1],
                        "fit_seconds": 0.0,
                        "hdh_seconds": hdh_seconds,
                        "network_delay_seconds": 0,
                        "communication_seconds": 0.0,
                        "wire_bytes": 0,
                        "raw_bytes": 0,
                        "compression_ratio": None,
                        "optimizer_iterations": 0,
                    }
                )
                continue

            selected_count += 1
            server_payload = transmit_parameters(
                global_parameters.coefficient,
                global_parameters.intercept,
                compressed=state.compressor,
                compression_level=config["patterns"]["message_compressor"]["level"],
            )
            transmitted_global = Parameters(*server_payload.values)

            fit_started = time.perf_counter()
            local_parameters, optimizer_iterations = _fit_local_model(
                transmitted_global, feature_data, label_data, config["model"]
            )
            fit_seconds = time.perf_counter() - fit_started
            client_payload = transmit_parameters(
                local_parameters.coefficient,
                local_parameters.intercept,
                compressed=state.compressor,
                compression_level=config["patterns"]["message_compressor"]["level"],
            )
            received_parameters = Parameters(*client_payload.values)
            updates.append((received_parameters, int(label_data.size)))

            network_delay = int(delays[round_index, client_index])
            communication_seconds = (
                float(network_delay)
                + server_payload.processing_seconds
                + client_payload.processing_seconds
            )
            # Docker clients execute concurrently. CPU=1 is represented as a
            # twofold local-compute cost relative to the 2-CPU clients.
            effective_fit_seconds = fit_seconds * (2.0 / client.cpu)
            client_duration = (
                effective_fit_seconds + hdh_seconds + communication_seconds
            )
            wire_bytes = server_payload.wire_bytes + client_payload.wire_bytes
            raw_bytes = server_payload.raw_bytes + client_payload.raw_bytes
            round_client_durations.append(client_duration)
            round_communications.append(communication_seconds)
            round_wire_bytes += wire_bytes

            client_rows.append(
                {
                    "method": method,
                    "network": network,
                    "round": round_number,
                    "client_id": client.client_id,
                    "silo": client.silo,
                    "code": client.code,
                    "cpu": client.cpu,
                    "selected": True,
                    "hdh_enabled": hdh_enabled,
                    "train_examples": int(label_data.size),
                    "jsd": effective_jsds[-1],
                    "fit_seconds": effective_fit_seconds,
                    "hdh_seconds": hdh_seconds,
                    "network_delay_seconds": network_delay,
                    "communication_seconds": communication_seconds,
                    "wire_bytes": wire_bytes,
                    "raw_bytes": raw_bytes,
                    "compression_ratio": wire_bytes / raw_bytes,
                    "optimizer_iterations": optimizer_iterations,
                }
            )

        aggregation_started = time.perf_counter()
        global_parameters = _aggregate(updates)
        aggregation_seconds = time.perf_counter() - aggregation_started
        round_seconds = max(round_client_durations) + aggregation_seconds
        communication_seconds = max(round_communications)
        evaluation_rows, global_metrics = _evaluate(global_parameters, clients)
        for evaluation_row in evaluation_rows:
            local_auc_rows.append(
                {
                    "method": method,
                    "network": network,
                    "round": round_number,
                    **evaluation_row,
                }
            )

        history = metrics_history["ProstateLR"]
        history["val_auc"].append(global_metrics["auc"])
        history["val_f1"].append(global_metrics["f1"])
        history["val_accuracy"].append(global_metrics["accuracy"])
        history["val_loss"].append(global_metrics["loss"])
        history["jsd"].append(tuple(effective_jsds))

        cumulative_time += round_seconds
        cumulative_f1 += global_metrics["f1"]
        cumulative_communication += communication_seconds
        cumulative_wire_bytes += round_wire_bytes
        next_state, explanations = policy.next_state(
            metrics_history, round_seconds, communication_seconds
        )
        round_rows.append(
            {
                "method": method,
                "network": network,
                "round": round_number,
                "selected_clients": selected_count,
                **state_record,
                "auc": global_metrics["auc"],
                "f1": global_metrics["f1"],
                "accuracy": global_metrics["accuracy"],
                "loss": global_metrics["loss"],
                "round_seconds": round_seconds,
                "communication_seconds": communication_seconds,
                "aggregation_seconds": aggregation_seconds,
                "wire_bytes": round_wire_bytes,
                "cumulative_time_seconds": cumulative_time,
                "cumulative_f1": cumulative_f1,
                "cumulative_communication_seconds": cumulative_communication,
                "cumulative_wire_bytes": cumulative_wire_bytes,
                "y1_f1_over_cumulative_time": global_metrics["f1"] / cumulative_time,
                "y2_cumulative_f1": cumulative_f1,
                "y3_cumulative_communication": cumulative_communication,
                "next_round_explanation": " | ".join(explanations),
            }
        )
        state = next_state

    round_frame = pd.DataFrame(round_rows)
    client_frame = pd.DataFrame(client_rows)
    local_auc_frame = pd.DataFrame(local_auc_rows)
    round_frame.to_csv(output_dir / "round_metrics.csv", index=False)
    client_frame.to_csv(output_dir / "client_round_metrics.csv", index=False)
    local_auc_frame.to_csv(output_dir / "local_auc.csv", index=False)

    final = round_rows[-1]
    summary = {
        "method": method,
        "network": network,
        "rounds": rounds,
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
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
