from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from prostate_flip.policies import FLiPOnlinePolicy  # noqa: E402


def test_online_policy_comes_from_main_flip_implementation() -> None:
    clients = [
        {"client_id": 1, "cpu": 2, "silo": "A"},
        {"client_id": 2, "cpu": 1, "silo": "B"},
    ]
    policy = FLiPOnlinePolicy(clients, alpha=1.0, seed=1234)
    assert policy.selector.__class__.__module__ == "adaptation_utils.strategy"
    assert policy.hdh.__class__.__module__ == "adaptation_utils.strategy"
    assert policy.compressor.__class__.__module__ == "adaptation_utils.strategy"
