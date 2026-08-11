from __future__ import annotations

import hashlib
import json
import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_RAW_COLUMNS = ["sig_cancer", "age", "PSA", "PV", "PIRADS", "5ARI"]
FEATURE_COLUMNS = [
    "age",
    "PSA",
    "PV",
    "PSA_SPLINE_2",
    "PSA_SPLINE_3",
    "PIRADS_3",
    "PIRADS_4",
    "PIRADS_5",
]


@dataclass(frozen=True)
class SiloDataset:
    client_id: int
    silo: str
    code: str
    country: str
    cpu: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def patient_count(self) -> int:
        return int(self.y_train.size + self.y_test.size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_data(data_path: Path, url: str, expected_sha256: str) -> Path:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists() and sha256(data_path) == expected_sha256:
        return data_path
    if data_path.exists():
        data_path.unlink()

    temporary_path = data_path.with_suffix(".download")
    if temporary_path.exists():
        temporary_path.unlink()
    urllib.request.urlretrieve(url, temporary_path)
    actual_sha256 = sha256(temporary_path)
    if actual_sha256 != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded data hash mismatch: expected {expected_sha256}, "
            f"received {actual_sha256}."
        )
    temporary_path.replace(data_path)
    return data_path


def _spline(value: float, knot_index: int) -> float:
    knots = [3.80, 6.60, 9.40, 18.47]
    return (
        max(value - knots[knot_index], 0) ** 3
        - max(value - knots[2], 0) ** 3
        * (knots[3] - knots[knot_index])
        / (knots[3] - knots[2])
        + max(value - knots[3], 0) ** 3
        * (knots[2] - knots[knot_index])
        / (knots[3] - knots[2])
    )


def transform_silo(raw: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(EXPECTED_RAW_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Raw silo is missing columns: {sorted(missing_columns)}")
    if raw[EXPECTED_RAW_COLUMNS].isna().any().any():
        raise ValueError("Processed source data unexpectedly contain missing values.")

    data = raw[EXPECTED_RAW_COLUMNS].copy()
    data["PV"] = np.where(data["5ARI"] == 1, data["PV"] / 0.7, data["PV"])
    data["PSA"] = np.where(data["5ARI"] == 1, data["PSA"] * 2.0, data["PSA"])
    data["PSA_SPLINE_2"] = data["PSA"].map(lambda value: _spline(float(value), 0))
    data["PSA_SPLINE_3"] = data["PSA"].map(lambda value: _spline(float(value), 1))

    pirads = data["PIRADS"].where(data["PIRADS"] >= 3, 0).astype(int)
    for score in (3, 4, 5):
        data[f"PIRADS_{score}"] = (pirads == score).astype(float)

    data["sig_cancer"] = data["sig_cancer"].astype(int)
    return data[["sig_cancer", *FEATURE_COLUMNS]].astype(float)


def load_raw_silos(data_path: Path, expected_sha256: str) -> dict[str, pd.DataFrame]:
    actual_sha256 = sha256(data_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Refusing to unpickle unverified data: expected {expected_sha256}, "
            f"received {actual_sha256}."
        )
    with data_path.open("rb") as stream:
        raw_silos = pickle.load(stream)
    if not isinstance(raw_silos, dict):
        raise TypeError("Expected data.pkl to contain a dictionary of silos.")
    return raw_silos


def load_clients(
    data_path: Path,
    clients_path: Path,
    config: dict[str, Any],
) -> tuple[list[SiloDataset], int]:
    raw_silos = load_raw_silos(data_path, config["data"]["sha256"])
    for excluded in config["data"]["excluded_silos"]:
        raw_silos.pop(excluded, None)

    clients_metadata = json.loads(clients_path.read_text())
    configured_silos = [client["silo"] for client in clients_metadata]
    if set(raw_silos) != set(configured_silos):
        raise ValueError(
            "Silo mismatch after exclusions. "
            f"Data={sorted(raw_silos)}, config={sorted(configured_silos)}"
        )

    random_state = int(np.random.RandomState(config["seed"]).randint(0, 10_000_000))
    clients: list[SiloDataset] = []
    for metadata in clients_metadata:
        transformed = transform_silo(raw_silos[metadata["silo"]])
        if len(transformed) != metadata["expected_patients"]:
            raise ValueError(
                f"Unexpected patient count for {metadata['silo']}: "
                f"expected {metadata['expected_patients']}, got {len(transformed)}."
            )

        X = transformed[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = transformed["sig_cancer"].to_numpy(dtype=np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["test_size"],
            random_state=random_state,
            stratify=y,
        )
        clients.append(
            SiloDataset(
                client_id=int(metadata["client_id"]),
                silo=metadata["silo"],
                code=metadata["code"],
                country=metadata["country"],
                cpu=int(metadata["cpu"]),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
        )
    return clients, random_state
