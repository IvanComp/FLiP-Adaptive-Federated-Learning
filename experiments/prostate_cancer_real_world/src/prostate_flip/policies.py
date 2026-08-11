from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Docker" / "adaptation_utils" / "strategy.py").exists():
            return parent
    raise FileNotFoundError("Could not locate the main FLiP Docker implementation.")


REPOSITORY_ROOT = _repository_root()
DOCKER_ROOT = REPOSITORY_ROOT / "Docker"
if str(DOCKER_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCKER_ROOT))

from adaptation_utils import strategy as flip_strategy  # noqa: E402


# The main logger writes to Docker/performance. Keep this isolated experiment
# read-only with respect to the main implementation.
flip_strategy.log = lambda *_args, **_kwargs: None


@dataclass(frozen=True)
class PatternState:
    selector: bool
    hdh_clients: frozenset[int]
    compressor: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "client_selector": self.selector,
            "heterogeneous_data_handler": bool(self.hdh_clients),
            "hdh_client_count": len(self.hdh_clients),
            "hdh_clients": ";".join(
                str(index + 1) for index in sorted(self.hdh_clients)
            ),
            "message_compressor": self.compressor,
        }


class Policy:
    def initial_state(self) -> PatternState:
        raise NotImplementedError

    def next_state(
        self,
        metrics_history: dict[str, dict[str, list[Any]]],
        round_seconds: float,
        communication_seconds: float,
    ) -> tuple[PatternState, list[str]]:
        raise NotImplementedError


class StaticPolicy(Policy):
    def __init__(self, mode: str, client_count: int, seed: int):
        self.mode = mode
        self.client_count = client_count
        self.rng = np.random.default_rng(seed)
        self._state = self._draw_state()

    def _draw_state(self) -> PatternState:
        if self.mode == "never":
            return PatternState(False, frozenset(), False)
        if self.mode == "always":
            return PatternState(True, frozenset(range(self.client_count)), True)
        if self.mode == "random":
            return PatternState(
                selector=bool(self.rng.integers(0, 2)),
                hdh_clients=frozenset(
                    index
                    for index in range(self.client_count)
                    if bool(self.rng.integers(0, 2))
                ),
                compressor=bool(self.rng.integers(0, 2)),
            )
        raise ValueError(f"Unknown static policy: {self.mode}")

    def initial_state(self) -> PatternState:
        return self._state

    def next_state(
        self,
        metrics_history: dict[str, dict[str, list[Any]]],
        round_seconds: float,
        communication_seconds: float,
    ) -> tuple[PatternState, list[str]]:
        if self.mode == "random":
            self._state = self._draw_state()
        return self._state, [f"{self.mode} baseline"]


class FLiPOnlinePolicy(Policy):
    model_type = "ProstateLR"

    def __init__(self, clients_metadata: list[dict[str, Any]], alpha: float, seed: int):
        client_details = [
            {
                "client_id": client["client_id"],
                "cpu": client["cpu"],
                "ram": 4,
                "dataset": client["silo"],
                "data_distribution_type": "non-IID",
                "data_persistence_type": "Same Data",
                "model": self.model_type,
                "epochs": 1,
            }
            for client in clients_metadata
        ]
        clients_config = {"client_details": client_details}
        self.client_count = len(client_details)
        self.selector = flip_strategy.ContextualBanditActivationCriterion(
            "client_selector", "val_f1", "f1overtime", clients_config, alpha=alpha
        )
        self.hdh = flip_strategy.ContextualBanditLocalActivationCriterion(
            "heterogeneous_data_handler",
            "val_f1,jsd",
            "f1overtime",
            clients_config,
            alpha=alpha,
        )
        self.compressor = flip_strategy.ContextualBanditActivationCriterion(
            "message_compressor",
            "communication_time",
            "f1overtime",
            clients_config,
            alpha=alpha,
        )
        self._numpy_state = np.random.RandomState(seed).get_state()
        # Matches the active states in FLiP's online-all configuration: CS and
        # MC start enabled; HDH has an initially empty enabled-client list.
        self._state = PatternState(True, frozenset(), True)

    def initial_state(self) -> PatternState:
        return self._state

    def _activate_with_isolated_rng(self, criterion: Any, args: dict[str, Any]):
        external_state = np.random.get_state()
        np.random.set_state(self._numpy_state)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                result = criterion.activate_pattern(args)
            self._numpy_state = np.random.get_state()
        finally:
            np.random.set_state(external_state)
        return result

    def next_state(
        self,
        metrics_history: dict[str, dict[str, list[Any]]],
        round_seconds: float,
        communication_seconds: float,
    ) -> tuple[PatternState, list[str]]:
        args = {
            "model_type": self.model_type,
            "metrics": metrics_history,
            "time": round_seconds,
            "communication_time": communication_seconds,
        }
        selector_on, _, selector_explanation = self._activate_with_isolated_rng(
            self.selector, args
        )
        hdh_on, hdh_parameters, hdh_explanation = self._activate_with_isolated_rng(
            self.hdh, args
        )
        compressor_on, _, compressor_explanation = self._activate_with_isolated_rng(
            self.compressor, args
        )

        enabled_clients: frozenset[int] = frozenset()
        if hdh_on and hdh_parameters:
            raw_enabled = hdh_parameters.get("enabled_clients", [])
            enabled_clients = frozenset(
                int(client_name.split()[-1]) - 1 for client_name in raw_enabled
            )
        self._state = PatternState(
            bool(selector_on), enabled_clients, bool(compressor_on)
        )
        return self._state, [
            selector_explanation,
            hdh_explanation,
            compressor_explanation,
        ]


def build_policy(
    method: str,
    clients_metadata: list[dict[str, Any]],
    alpha: float,
    seed: int,
) -> Policy:
    if method == "fliponline":
        return FLiPOnlinePolicy(clients_metadata, alpha=alpha, seed=seed)
    return StaticPolicy(method, client_count=len(clients_metadata), seed=seed)
