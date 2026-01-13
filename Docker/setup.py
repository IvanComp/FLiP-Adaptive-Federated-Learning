import sys

# Usage: python setup.py <config_name> <N_iid> <N_high> <N_low> [<threshold>]

N_iid = int(sys.argv[2])
N_high = int(sys.argv[3])
N_low = int(sys.argv[4])
persistence = sys.argv[5]
delay = sys.argv[6].lower() == 'yes'
data_persistence_types = {'new': 'New Data', 'same': 'Same Data', 'remove': 'Remove Data'}

N_other = 0

if 'text' in sys.argv[1]:
    json_tplt = """{{
                \"client_id\": {},
                \"cpu\": {},
                \"ram\": 4,
                \"dataset\": \"AGNEWS\",
                \"data_distribution_type\": \"{}\",
                \"data_persistence_type\": \"{}\",
                \"delay_combobox\": \"{}\",
                \"model\": \"TextMLP\",
                \"epochs\": 1
            }}"""
else:
    json_tplt = """{{
                \"client_id\": {},
                \"cpu\": {},
                \"ram\": 4,
                \"dataset\": \"CIFAR10\",
                \"data_distribution_type\": \"{}\",
                \"data_persistence_type\": \"{}\",
                \"delay_combobox\": \"{}\",
                \"model\": \"CNN 16k\",
                \"epochs\": 1
            }}"""

with open("configuration/config.json", "w") as config_file:
    with open("configuration/{}.json".format(sys.argv[1]), "r") as default_config_file:
        to_copy = default_config_file.read()

        to_copy = to_copy.replace("**CLIENTS_COUNT**", str(N_high + N_low))

        delay_str = 'Yes' if delay else 'No'

        clients_config = []
        for i in range(N_high):
            if i < N_high * N_iid / 100:
                clients_config.append(
                    json_tplt.format(i + 1 + N_other, 2, "IID", data_persistence_types[persistence], delay_str))
            else:
                clients_config.append(
                    json_tplt.format(i + 1 + N_other, 2, "non-IID", data_persistence_types[persistence], delay_str))
        for i in range(N_low):
            if i < N_low * N_iid / 100:
                clients_config.append(
                    json_tplt.format(i + 1 + N_high + N_other, 1, "IID", data_persistence_types[persistence],
                                     delay_str))
            else:
                clients_config.append(
                    json_tplt.format(i + 1 + N_high + N_other, 1, "non-IID", data_persistence_types[persistence],
                                     delay_str))
        to_copy = to_copy.replace("**CLIENTS**", ",\n".join(clients_config))

        if len(sys.argv) > 7:
            to_copy = to_copy.replace("**TH**", sys.argv[7])

        config_file.writelines(to_copy)

print("Configuration file created with {} high-spec, {} low-spec clients, {} IID, {} data persistence.".format(N_high,
                                                                                                               N_low,
                                                                                                               N_iid,
                                                                                                               persistence,
                                                                                                               delay_str))

docker_client_tplt = """  {}:
    build:
      context: .
      dockerfile: Dockerfile.client
    command: sh -c "sleep 20 && python client.py"
    container_name: {}
    cpus: {}
    cpuset: \"{}\"
    depends_on:
      - server
    environment:
      CLIENT_ID: '{}'
      DELAY_FLAG: '{}'
      NUM_CPUS: '{}'
      NUM_RAM: '4'
      NUM_ROUNDS: '20'
      SERVER_ADDRESS: server:8080
    labels:
      - type=client
    mem_limit: 4g
    networks:
      - flwr_network
    volumes:
      - ./data:/app/data
      - ./performance:/app/performance
      - ./configuration:/app/configuration
      - ./model_weights:/app/model_weights"""

with open("docker-compose.dynamic.yml", "w") as docker_file:
    with open("template.docker-compose.dynamic.yml") as default_docker_file:
        docker_tplt = default_docker_file.read()
        cpu_set = list(range(N_other, N_other + 2 * N_high + N_low))
        docker_clients = []
        for i in range(N_high):
            docker_clients.append(docker_client_tplt.format(
                "client{}".format(i + 1 + N_other),
                "Client{}".format(i + 1 + N_other),
                2,
                ",".join(str(cpu) for cpu in cpu_set[i * 2:i * 2 + 2]),
                i + 1 + N_other,
                '1' if delay else '0',
                2
            ))
        for i in range(N_low):
            docker_clients.append(docker_client_tplt.format(
                "client{}".format(i + 1 + N_high + N_other),
                "Client{}".format(i + 1 + N_high + N_other),
                1,
                str(cpu_set[i + 2 * N_high]),
                i + 1 + N_high + N_other,
                '1' if delay else '0',
                1
            ))
        docker_file.write(docker_tplt.replace("**CLIENTS**", "\n".join(docker_clients)))

print("Docker Compose file created with {} high-spec, {} low-spec clients.".format(N_high, N_low))
