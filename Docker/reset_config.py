with open("configuration/config.json", "w") as config_file:
    with open("configuration/default_config.json", "r") as default_config_file:
        to_copy = default_config_file.readlines()
        config_file.writelines(to_copy)