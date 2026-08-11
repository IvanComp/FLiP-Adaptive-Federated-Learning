import json
from pathlib import Path

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_compose_has_one_server_and_eleven_isolated_clients():
    compose = yaml.safe_load((EXPERIMENT_ROOT / "docker-compose.yml").read_text())
    metadata = json.loads((EXPERIMENT_ROOT / "config" / "clients.json").read_text())
    services = compose["services"]
    client_services = {
        name: service
        for name, service in services.items()
        if name.startswith("client-")
    }

    assert set(services) == {"server", *client_services}
    assert len(client_services) == 11
    assert not any("/app/data" in volume for volume in services["server"]["volumes"])

    for client in metadata:
        service = client_services[f"client-{client['code']}"]
        assert float(service["cpus"]) == client["cpu"]
        assert service["environment"]["CLIENT_ID"] == str(client["client_id"])
        assert service["environment"]["SILO"] == client["silo"]
        assert len(service["volumes"]) == 1
        assert service["volumes"][0].endswith(
            f"/data/clients/{client['code']}.npz:/app/data/client.npz:ro"
        )
