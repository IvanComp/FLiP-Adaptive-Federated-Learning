import random
from pickle import load

import joblib
import numpy as np
from skopt import gp_minimize


def get_patterns(json_config):
    return json_config['patterns']


def get_activation_criteria(json_config, default_config):
    # TODO should work with multiple metrics
    metric_name = json_config['activation_criteria']['metrics'][0]['metric']
    threshold_config = json_config['activation_criteria']['metrics'][0]['threshold']
    strategy_name = json_config['activation_criteria']['metrics'][0]['name']

    if threshold_config['calculation_method'] == 'predictor_based':
        model_path = threshold_config['predictor']['model_path']
        with open(model_path, "rb") as f:
            model = load(f)

        return [PredictorBasedActivationCriterion(metric_name, model, model_path, strategy_name, default_config)]
    elif threshold_config['calculation_method'] == 'bayesian_optimization':
        model_path = threshold_config['predictor']['model_path']
        model = joblib.load(threshold_config['predictor']['model_path'])

        return [BayesianOptimizationActivationCriterion(metric_name, model, model_path, strategy_name, default_config)]
    elif threshold_config['calculation_method'] == 'fixed':
        return [FixedThresholdActivationCriterion(metric_name, float(threshold_config['value']), strategy_name,
                                                  default_config)]
    else:
        return [RandomActivationCriterion(metric_name, strategy_name, default_config)]


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
    def __init__(self, metric, strategy_name, clients_config):
        self.metric = metric
        self.strategy_name = strategy_name
        self.clients_config = clients_config

    def __str__(self):
        return 'Abstract activation criterion (keeps default config).'

    def activate_pattern(self, args):
        return True


class RandomActivationCriterion(ActivationCriterion):
    def __init__(self, metric, strategy_name, clients_config):
        super().__init__(metric, strategy_name, clients_config)

    def __str__(self):
        return f'Random activation criterion'

    def activate_pattern(self, args):
        activate = random.choice([True, False])

        if not activate:
            return False, "Selector de-activated ❌"
        else:
            return True, "Selector activated ✅"


class FixedThresholdActivationCriterion(ActivationCriterion):
    def __init__(self, metric, value, strategy_name, clients_config):
        self.value = value
        super().__init__(metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, comparison with fixed value: {self.value}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']

        last_f1 = new_aggregated_metrics[model_type][self.metric][-1]
        if len(new_aggregated_metrics[model_type][self.metric]) < 2:
            second_last_f1 = None
        else:
            second_last_f1 = new_aggregated_metrics[model_type][self.metric][-2]

        decreasing_f1 = second_last_f1 is None or last_f1 < second_last_f1
        low_f1 = last_f1 < self.value

        if decreasing_f1 or low_f1:
            return False, "Accuracy too low or decreasing, Selector de-activated ❌"
        else:
            return True, "Accuracy ok: Selector activated ✅"


class PredictorBasedActivationCriterion(ActivationCriterion):
    def __init__(self, metric, model, model_name, strategy_name, clients_config):
        self.model = model
        self.model_name = model_name
        super().__init__(metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, predictor-based: {self.model_name}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']

        if len(new_aggregated_metrics[model_type][self.metric]) < 1:
            return True, "Less than 1 round completed, keeping default config."

        last_f1 = new_aggregated_metrics[model_type][self.metric][-1]

        iid_clients = get_no_iid_clients(self.clients_config)
        n_high, n_low = get_high_low_clients(self.clients_config)

        # TODO should be parametric w.r.t. predictor input
        prediction_w_pattern = self.model.predict([[n_high, n_low, iid_clients, True, last_f1 / last_round_time]])[0]
        prediction_wo_pattern = self.model.predict([[n_high, n_low, iid_clients, False, last_f1 / last_round_time]])[0]

        expl = "{}-iid clients, predicted wo:{:.4f} vs w:{:.4f}, ".format(iid_clients, prediction_wo_pattern,
                                                                          prediction_w_pattern)

        if prediction_wo_pattern > prediction_w_pattern:
            return False, expl + 'pattern de-activated ❌'
        else:
            return True, expl + 'pattern activated ✅'


class BayesianOptimizationActivationCriterion(ActivationCriterion):
    def __init__(self, metric, model, model_name, strategy_name, clients_config):
        self.model = model
        self.model_name = model_name
        super().__init__(metric, strategy_name, clients_config)

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
                    return objective(pattern_on, 1, f1_over_time1[i])

                res = gp_minimize(wrapped_objective,  # objective fn
                                  [(0, 1)],  # pattern_on ∈ {0,1}
                                  acq_func="EI",  # acquisition function
                                  n_calls=10, random_state=42)

                best_policy.append(res.x[0])

            if sum(best_policy) / len(best_policy) < 0.5:
                return False, "First round, average best policy: pattern de-activated ❌"
            else:
                return True, "First round, average best policy: pattern activated ✅"

                # Otherwise, we know the last round's F1 score and time
        last_f1 = new_aggregated_metrics[model_type][self.metric][-1]
        next_round = len(new_aggregated_metrics[model_type][self.metric]) + 1

        def wrapped_objective(pattern_on):
            return objective(pattern_on, next_round, last_f1 / last_round_time)

        res = gp_minimize(wrapped_objective,  # objective fn
                          [(0, 1)],  # pattern_on ∈ {0,1}
                          acq_func="EI",  # acquisition function
                          n_calls=10, random_state=42)

        expl = "round {}, {} high, {} low, {}-iid clients, predicted best policy: ".format(next_round, n_high, n_low,
                                                                                           iid_clients, res.x[0])

        if not res.x[0]:
            return False, expl + 'pattern de-activated ❌'
        else:
            return True, expl + 'pattern activated ✅'
