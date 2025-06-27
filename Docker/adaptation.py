import json
import os
from logging import INFO
from typing import List

from adaptation_utils.strategy import get_patterns, get_activation_criteria, ActivationCriterion
from logger import log

# Path to the 'configuration' directory
current_dir = os.getcwd().replace('/adaptation', '')
config_dir = os.path.join(current_dir, 'configuration')
config_file = os.path.join(config_dir, 'config.json')

adaptation_config_file = os.path.join(config_dir, 'adaptation_strategy.json')


def get_model_type():
    with open(config_file, 'r') as f:
        config = json.load(f)
        # FIXME: also works only if all clients have the same model type
        return config['client_details'][0]['model']


class AdaptationManager:
    def __init__(self, enabled, default_config):
        self.name = 'AdaptationManager'
        self.enabled = enabled

        adaptation_config = json.load(open(adaptation_config_file, 'r'))

        self.patterns = get_patterns(adaptation_config)
        self.activation_criteria: List[ActivationCriterion] = get_activation_criteria(adaptation_config)

        self.model_type = get_model_type()

        self.default_config = default_config
        self.cached_config = dict()
        self.update_config(default_config)
        self.update_json(default_config)

        self.cached_aggregated_metrics = None

    def describe(self):
        return log(INFO, '\n'.join([str(cr) for cr in self.activation_criteria]))

    def update_metrics(self, new_aggregated_metrics):
        self.cached_aggregated_metrics = new_aggregated_metrics

    def update_config(self, new_config):
        for pattern in new_config:
            if pattern in self.cached_config:
                if 'enabled' in self.cached_config[pattern]:
                    self.cached_config[pattern]['enabled'] = new_config[pattern]['enabled']
                else:
                    self.cached_config[pattern] = {'enabled': new_config[pattern]['enabled']}
            else:
                self.cached_config[pattern] = {'enabled': new_config[pattern]['enabled']}

    def update_json(self, new_config):
        with open(config_file, 'r') as f:
            config = json.load(f)
            for pattern in new_config:
                config['patterns'][pattern]['enabled'] = new_config[pattern]['enabled']
            json.dump(config, open(config_file, 'w'), indent=4)

    def config_next_round(self, new_aggregated_metrics, last_round_time):
        if self.enabled:
            log(INFO, f"{self.name}: Configuring next round...")
            log(INFO, self.default_config)
            log(INFO, self.cached_config)
        else:
            return self.default_config

        if self.cached_aggregated_metrics is None:
            log(INFO, f"{self.name}: Less than 2 rounds completed. Keeping default config.")

            self.update_metrics(new_aggregated_metrics)
            return self.default_config

        new_config = self.cached_config.copy()

        for p_i, pattern in enumerate(self.patterns):
            if self.default_config[pattern]['enabled']:
                if self.model_type not in new_aggregated_metrics:
                    log(INFO, f"{self.name}: Wrong global metrics format. Keeping default config.")

                    self.update_metrics(new_aggregated_metrics)
                    return self.default_config
                else:
                    args = {"model_type": self.model_type, "metrics": new_aggregated_metrics, "time": last_round_time}
                    activate, expl = self.activation_criteria[p_i].activate_pattern(args)

                    if not activate:
                        new_config[pattern]['enabled'] = False
                        log(INFO, expl)
                    else:
                        new_config[pattern]['enabled'] = True
                        log(INFO, expl)

        self.update_metrics(new_aggregated_metrics)
        self.update_config(new_config)
        self.update_json(new_config)

        return new_config
