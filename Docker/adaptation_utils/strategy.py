import random
from pickle import load

import joblib
import numpy as np
from skopt import gp_minimize

from logging import INFO
from logger import log


def get_patterns(json_config):
    return json_config['patterns']


def get_activation_criteria(json_config, default_config):
    # TODO should work with multiple metrics
    pattern_name = json_config['activation_criteria']['metrics'][0]['pattern']
    metric_name = json_config['activation_criteria']['metrics'][0]['metric']
    threshold_config = json_config['activation_criteria']['metrics'][0]['threshold']
    strategy_name = json_config['activation_criteria']['metrics'][0]['name']

    if threshold_config['calculation_method'] == 'predictor_based':
        model_path = threshold_config['predictor']['model_path']
        with open(model_path, "rb") as f:
            model = load(f)

        return [PredictorBasedActivationCriterion(pattern_name, metric_name, model, model_path, strategy_name,
                                                  default_config)]
    elif threshold_config['calculation_method'] == 'predictor-local':
        model_path = threshold_config['predictor']['model_path']
        with open(model_path, "rb") as f:
            model = load(f)

        return [PredictorBasedLocalActivationCriterion(pattern_name, metric_name, model, model_path, strategy_name,
                                                       default_config)]
    elif threshold_config['calculation_method'] == 'bayesian_optimization':
        model_path = threshold_config['predictor']['model_path']
        model = joblib.load(threshold_config['predictor']['model_path'])

        return [BayesianOptimizationActivationCriterion(pattern_name, metric_name, model, model_path, strategy_name,
                                                        default_config)]
    elif threshold_config['calculation_method'] == 'fixed-global':
        return [FixedGlobalThresholdActivationCriterion(pattern_name, metric_name, float(threshold_config['value']),
                                                        strategy_name,
                                                        default_config)]
    elif threshold_config['calculation_method'] == 'fixed-local':
        return [FixedLocalThresholdActivationCriterion(pattern_name, metric_name, float(threshold_config['value']),
                                                       strategy_name,
                                                       default_config)]
    else:
        return [RandomActivationCriterion(pattern_name, metric_name, strategy_name, default_config)]


def get_no_iid_clients(clients_config):
    no_clients = len(clients_config['client_details'])
    iid_clients = sum(
        [client["data_distribution_type"] == "IID" for client in clients_config['client_details']]) / no_clients * 100
    return iid_clients


def get_high_low_clients(clients_config):
    high_clients = sum([client["cpu"] >= 2 for client in clients_config['client_details']])
    low_clients = sum([client["cpu"] < 2 for client in clients_config['client_details']])
    return high_clients, low_clients


class ActivationCriterion:
    def __init__(self, pattern, metric, strategy_name, clients_config):
        self.pattern = pattern
        self.metric = metric
        self.strategy_name = strategy_name
        self.clients_config = clients_config

    def __str__(self):
        return 'Abstract activation criterion (keeps default config).'

    def activate_pattern(self, args):
        return True


class RandomActivationCriterion(ActivationCriterion):
    def __init__(self, pattern, metric, strategy_name, clients_config):
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Random activation criterion'

    def activate_pattern(self, args):
        activate = random.choice([True, False])

        if activate:
            apply_pattern = []
            for client_i in range(len(self.clients_config['client_details'])):
                if random.choice([True, False]):
                    apply_pattern.append(f"Client {client_i + 1}")

        if not activate:
            return False, None, f"{self.pattern} de-activated ❌"
        else:
            return True, {"enabled_clients": apply_pattern}, f"{self.pattern} activated ✅"


class FixedGlobalThresholdActivationCriterion(ActivationCriterion):
    def __init__(self, pattern, metric, value, strategy_name, clients_config):
        self.value = value
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Global Metric: {self.metric}, comparison with fixed value: {self.value}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']

        last_metric = new_aggregated_metrics[model_type][self.metric][-1]
        if len(new_aggregated_metrics[model_type][self.metric]) < 2:
            second_last_metric = None
        else:
            second_last_metric = new_aggregated_metrics[model_type][self.metric][-2]

        decreasing_metric = second_last_metric is None or last_metric < second_last_metric
        low_metric = last_metric < self.value

        if decreasing_metric or low_metric:
            return False, None, f"{self.metric} too low or decreasing, {self.pattern} de-activated ❌"
        else:
            return True, None, f"{self.metric} ok: {self.pattern} activated ✅"


class FixedLocalThresholdActivationCriterion(ActivationCriterion):
    def __init__(self, pattern, metric, value, strategy_name, clients_config):
        self.value = value
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Local Metric: {self.metric}, comparison with fixed value: {self.value}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']

        last_metric = new_aggregated_metrics[model_type][self.metric][-1]
        apply_pattern = []
        for client_i in range(len(self.clients_config['client_details'])):
            if last_metric[client_i] > self.value:
                apply_pattern.append(f"Client {client_i + 1}")

        if len(apply_pattern) == 0:
            return False, None, f"No client has {self.metric} above {self.value}, {self.pattern} de-activated ❌"
        else:
            return True, {"enabled_clients": apply_pattern}, f"{self.pattern} activated ✅ for clients: {apply_pattern}"


class PredictorBasedActivationCriterion(ActivationCriterion):
    def __init__(self, pattern, metric, model, model_name, strategy_name, clients_config):
        self.model = model
        self.model_name = model_name
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, predictor-based: {self.model_name}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']

        if len(new_aggregated_metrics[model_type][self.metric]) < 1:
            return True, "Less than 1 round completed, keeping default config."

        last_metric = new_aggregated_metrics[model_type][self.metric][-1]

        iid_clients = get_no_iid_clients(self.clients_config)
        n_high, n_low = get_high_low_clients(self.clients_config)

        # TODO should be parametric w.r.t. predictor input
        prediction_w_pattern = self.model.predict([[n_high, n_low, iid_clients, True, last_metric / last_round_time]])[
            0]
        prediction_wo_pattern = \
            self.model.predict([[n_high, n_low, iid_clients, False, last_metric / last_round_time]])[0]

        expl = "{}-iid clients, predicted wo:{:.4f} vs w:{:.4f}, ".format(iid_clients, prediction_wo_pattern,
                                                                          prediction_w_pattern)

        if prediction_wo_pattern > prediction_w_pattern:
            return False, None, expl + f'{self.pattern} de-activated ❌'
        else:
            return True, None, expl + f'{self.pattern} activated ✅'


class PredictorBasedLocalActivationCriterion(ActivationCriterion):
    def __init__(self, pattern, metric, model, model_name, strategy_name, clients_config):
        self.model = model
        self.model_name = model_name
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, local predictor-based: {self.model_name}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']
        performed_rounds = len(new_aggregated_metrics[model_type]['val_f1'])

        # TODO should be parametric w.r.t. metric name
        if performed_rounds < 1:
            return True, "Less than 1 round completed, keeping default config."

        metrics = self.metric.split(',')
        last_metrics = {m: new_aggregated_metrics[model_type][m][-1] for m in metrics}

        # TODO should be parametric w.r.t. metric name        
        last_val_f1 = last_metrics['val_f1']

        n_high, n_low = get_high_low_clients(self.clients_config)

        apply_pattern = []
        for client_i in range(len(self.clients_config['client_details'])):
            # TODO should be parametric w.r.t. metric name
            last_jsd = last_metrics['jsd'][client_i]

            # TODO should be parametric w.r.t. predictor input
            prediction_w_pattern = \
            self.model.predict([[performed_rounds + 1, True, last_jsd, last_val_f1 / last_round_time]])[0]
            prediction_wo_pattern = \
            self.model.predict([[performed_rounds + 1, False, last_jsd, last_val_f1 / last_round_time]])[0]

            expl = f"client {client_i + 1}, predicted wo:{prediction_wo_pattern:.4f} vs w:{prediction_w_pattern:.4f}"
            log(INFO, expl)

            if prediction_w_pattern >= prediction_wo_pattern:
                apply_pattern.append(f"Client {client_i + 1}")

        if len(apply_pattern) == 0:
            return False, None, f'{self.pattern} de-activated ❌'
        else:
            return True, {"enabled_clients": apply_pattern}, f'{self.pattern} activated ✅ for clients: {apply_pattern}'


class BayesianOptimizationActivationCriterion(ActivationCriterion):
    def __init__(self, pattern, metric, model, model_name, strategy_name, clients_config):
        self.model = model
        self.model_name = model_name
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, bo-based: {self.model_name}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']

        iid_clients = get_no_iid_clients(self.clients_config)
        n_high, n_low = get_high_low_clients(self.clients_config)

        scaler = joblib.load('predictors/bo_scaler.pkl')

        def objective(pattern_on, next_round, prev_f1_overtime):  # binary: 0 or 1
            X = [[n_high, n_low, iid_clients, pattern_on[0], next_round, prev_f1_overtime]]
            X_scaled = scaler.transform(X)
            y_pred = -self.model.predict(X_scaled)[0]  # maximize → minimize negative
            return y_pred

        if len(new_aggregated_metrics[model_type][self.metric]) < 1:
            # If this is the first round, we get the average best policy
            # over a given range of F1/time scores
            f1_over_time = np.arange(0, 0.0006, 0.00005)

            best_policy = []
            for i in range(len(f1_over_time)):
                def wrapped_objective(pattern_on):
                    return objective(pattern_on, 1, f1_over_time[i])

                res = gp_minimize(wrapped_objective,  # objective fn
                                  [(0, 1)],  # pattern_on ∈ {0,1}
                                  acq_func="EI",  # acquisition function
                                  n_calls=10, random_state=42)

                best_policy.append(res.x[0])

            if sum(best_policy) / len(best_policy) < 0.5:
                return False, None, f"First round, average best policy: {self.pattern} de-activated ❌"
            else:
                return True, None, f"First round, average best policy: {self.pattern} activated ✅"

                # Otherwise, we know the last round's F1 score and time
        last_metric = new_aggregated_metrics[model_type][self.metric][-1]
        next_round = len(new_aggregated_metrics[model_type][self.metric]) + 1

        def wrapped_objective(pattern_on):
            return objective(pattern_on, next_round, last_metric / last_round_time)

        res = gp_minimize(wrapped_objective,  # objective fn
                          [(0, 1)],  # pattern_on ∈ {0,1}
                          acq_func="EI",  # acquisition function
                          n_calls=10, random_state=42)

        expl = "round {}, {} high, {} low, {}-iid clients, predicted best policy: ".format(next_round, n_high, n_low,
                                                                                           iid_clients, res.x[0])

        if not res.x[0]:
            return False, None, expl + f'{self.pattern} de-activated ❌'
        else:
            return True, None, expl + f'{self.pattern} activated ✅'
