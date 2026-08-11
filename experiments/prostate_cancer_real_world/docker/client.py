from __future__ import annotations

import os
import pickle
import random
import time
import warnings
import zlib
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from prostate_flip.patterns import label_jsd, random_minority_oversample


DATA_PATH = Path(os.getenv("CLIENT_DATA_PATH", "/app/data/client.npz"))
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "server:8080")
CLIENT_ID = int(os.environ["CLIENT_ID"])
SILO = os.environ["SILO"]
CLIENT_CODE = os.environ["CLIENT_CODE"]
CPU_LIMIT = int(os.environ["CPU_LIMIT"])
SEED = int(os.getenv("SEED", "1234"))
REPETITION = int(os.getenv("REPETITION", "1"))


def serialize_parameters(parameters: list[np.ndarray], compressed: bool) -> bytes:
    raw = pickle.dumps(parameters, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, level=1) if compressed else raw


def deserialize_parameters(payload: bytes, compressed: bool) -> list[np.ndarray]:
    raw = zlib.decompress(payload) if compressed else payload
    values = pickle.loads(raw)
    return [np.asarray(value, dtype=np.float64) for value in values]


def parameters_from_config(
    parameters: list[np.ndarray], config: dict[str, Any]
) -> tuple[list[np.ndarray], int]:
    if bool(config.get("compressed", False)):
        payload = bytes(config["compressed_parameters"])
        return deserialize_parameters(payload, compressed=True), len(payload)
    raw = serialize_parameters(parameters, compressed=False)
    return [np.asarray(value, dtype=np.float64) for value in parameters], len(raw)


def fit_logistic_regression(
    initial: list[np.ndarray], features: np.ndarray, labels: np.ndarray
) -> tuple[list[np.ndarray], int]:
    coefficient, intercept = initial
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight=None,
        fit_intercept=True,
        tol=0.0001,
        max_iter=100,
        warm_start=True,
    )
    model.classes_ = np.array([0, 1], dtype=np.int64)
    model.coef_ = coefficient.copy()
    model.intercept_ = intercept.copy()
    model.n_features_in_ = features.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(features, labels)
    return [model.coef_.copy(), model.intercept_.copy()], int(model.n_iter_[0])


def evaluate_parameters(
    parameters: list[np.ndarray], features: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    coefficient, intercept = parameters
    probabilities = expit(features @ coefficient.reshape(-1) + float(intercept[0]))
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


class ProstateClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        with np.load(DATA_PATH) as data:
            self.X_train = data["X_train"].astype(np.float64)
            self.X_test = data["X_test"].astype(np.float64)
            self.y_train = data["y_train"].astype(np.int64)
            self.y_test = data["y_test"].astype(np.int64)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return [
            np.zeros((1, self.X_train.shape[1]), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        ]

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        round_number = int(config["server_round"])
        compressed = bool(config.get("compressed", False))
        global_parameters, download_wire_bytes = parameters_from_config(
            parameters, config
        )
        selector_enabled = bool(config.get("client_selector", False))
        selected = not selector_enabled or CPU_LIMIT >= int(
            config.get("cpu_threshold", 2)
        )
        hdh_clients = {
            int(value)
            for value in str(config.get("hdh_clients", "")).split(",")
            if value
        }
        hdh_enabled = CLIENT_ID in hdh_clients
        features = self.X_train
        labels = self.y_train
        hdh_seconds = 0.0

        if selected and hdh_enabled:
            hdh_started = time.perf_counter()
            rng = np.random.default_rng(SEED + 100_000 * round_number + CLIENT_ID)
            features, labels = random_minority_oversample(features, labels, rng)
            hdh_seconds = time.perf_counter() - hdh_started

        if not selected:
            return (
                global_parameters,
                0,
                {
                    "client_id": CLIENT_ID,
                    "silo": SILO,
                    "code": CLIENT_CODE,
                    "cpu": CPU_LIMIT,
                    "selected": False,
                    "hdh_enabled": hdh_enabled,
                    "train_examples": 0,
                    "jsd": label_jsd(self.y_train),
                    "training_seconds": 0.0,
                    "hdh_seconds": 0.0,
                    "delay_seconds": 0,
                    "communication_seconds": 0.0,
                    "download_wire_bytes": download_wire_bytes,
                    "upload_wire_bytes": 0,
                    "raw_upload_bytes": 0,
                    "optimizer_iterations": 0,
                },
            )

        training_started = time.perf_counter()
        updated_parameters, optimizer_iterations = fit_logistic_regression(
            global_parameters, features, labels
        )
        training_seconds = time.perf_counter() - training_started

        serialization_started = time.perf_counter()
        raw_payload = serialize_parameters(updated_parameters, compressed=False)
        upload_payload = (
            zlib.compress(raw_payload, level=1) if compressed else raw_payload
        )
        serialization_seconds = time.perf_counter() - serialization_started

        delay_seconds = 0
        if config.get("network_condition") == "unstable":
            delay_rng = random.Random(
                SEED
                + 20_026
                + 1_000_000 * (REPETITION - 1)
                + 1_000 * round_number
                + CLIENT_ID
            )
            delay_seconds = delay_rng.randint(0, 20)
            time.sleep(delay_seconds)

        metrics: dict[str, Any] = {
            "client_id": CLIENT_ID,
            "silo": SILO,
            "code": CLIENT_CODE,
            "cpu": CPU_LIMIT,
            "selected": True,
            "hdh_enabled": hdh_enabled,
            "train_examples": int(labels.size),
            "jsd": label_jsd(labels),
            "training_seconds": training_seconds,
            "hdh_seconds": hdh_seconds,
            "delay_seconds": delay_seconds,
            "communication_seconds": delay_seconds + serialization_seconds,
            "download_wire_bytes": download_wire_bytes,
            "upload_wire_bytes": len(upload_payload),
            "raw_upload_bytes": len(raw_payload),
            "optimizer_iterations": optimizer_iterations,
        }
        if compressed:
            metrics["compressed_update"] = upload_payload
            return [], int(labels.size), metrics
        return updated_parameters, int(labels.size), metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        global_parameters, download_wire_bytes = parameters_from_config(
            parameters, config
        )
        metrics = evaluate_parameters(global_parameters, self.X_test, self.y_test)
        return (
            metrics["loss"],
            int(self.y_test.size),
            {
                "client_id": CLIENT_ID,
                "silo": SILO,
                "code": CLIENT_CODE,
                "test_examples": int(self.y_test.size),
                "auc": metrics["auc"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "download_wire_bytes": download_wire_bytes,
            },
        )


if __name__ == "__main__":
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=ProstateClient().to_client(),
    )
