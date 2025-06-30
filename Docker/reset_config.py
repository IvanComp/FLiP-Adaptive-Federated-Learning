import sys

with open("configuration/config.json", "w") as config_file:
    with open("configuration/{}.json".format(sys.argv[1]), "r") as default_config_file:
        to_copy = default_config_file.readlines()
        config_file.writelines(to_copy)