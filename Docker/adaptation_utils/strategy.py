import random
from logging import INFO
from pickle import load

import joblib
import numpy as np
from logger import log
from skopt import gp_minimize


def get_patterns(json_config):
    return json_config['patterns']


def get_activation_criteria(json_config, default_config):
    # TODO should work with multiple metrics
    pattern_name = json_config['activation_criteria']['metrics'][0]['pattern']
    metric_name = json_config['activation_criteria']['metrics'][0]['metric']
    metric_type = json_config['activation_criteria']['metrics'][0]['threshold']['type']
    threshold_config = json_config['activation_criteria']['metrics'][0]['threshold']
    strategy_name = json_config['activation_criteria']['metrics'][0]['name']

    if threshold_config['calculation_method'] == 'predictor_based':
        model_path = threshold_config['predictor']['model_path']
        with open(model_path, "rb") as f:
            model = load(f)

        return [PredictorBasedActivationCriterion(pattern_name, metric_name, model, model_path, strategy_name,
                                                  default_config, metric_type)]
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

    elif threshold_config['calculation_method'] == 'bayesian_optimization-local':
        model_path = threshold_config['predictor']['model_path']
        model = joblib.load(threshold_config['predictor']['model_path'])

        return [
            BayesianOptimizationLocalActivationCriterion(pattern_name, metric_name, model, model_path, strategy_name,
                                                         default_config)]
    elif threshold_config['calculation_method'] == 'fixed-global':
        return [FixedGlobalThresholdActivationCriterion(pattern_name, metric_name, float(threshold_config['value']),
                                                        strategy_name, default_config, metric_type)]
    elif threshold_config['calculation_method'] == 'fixed-local':
        return [FixedLocalThresholdActivationCriterion(pattern_name, metric_name, float(threshold_config['value']),
                                                       strategy_name,
                                                       default_config)]
    elif threshold_config['calculation_method'] == "contextual_bandit":
        return [ContextualBanditActivationCriterion(pattern_name, metric_name, strategy_name, default_config,
                                                    alpha=threshold_config.get("alpha", 1.0))]
    elif threshold_config['calculation_method'] == "contextual_bandit-local":
        return [ContextualBanditLocalActivationCriterion(pattern_name, metric_name, strategy_name, default_config,
                                                         alpha=threshold_config.get("alpha", 1.0))]
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


def compare_w_threshold(metric, value, metric_type):
    # if metric is supposed to increase, activate pattern if below a threshold
    # if metric is supposed to decrease, activate pattern if above a threshold
    if metric_type == 'increasing':
        return metric <= value
    elif metric_type == 'decreasing':
        return metric > value
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


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
    def __init__(self, pattern, metric, value, strategy_name, clients_config, metric_type):
        self.value = value
        self.metric_type = metric_type
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Global Metric: {self.metric}, comparison with fixed value: {self.value}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']

        if self.metric != 'time':
            last_metric = new_aggregated_metrics[model_type][self.metric][-1]
        else:
            last_metric = last_round_time

        if self.metric != 'time' and len(new_aggregated_metrics[model_type][self.metric]) < 2:
            second_last_metric = None
        else:
            # WARNING: we don't currently keep track of time beyond the last round 
            if self.metric != 'time':
                second_last_metric = new_aggregated_metrics[model_type][self.metric][-2]

        critical_metric = compare_w_threshold(last_metric, self.value, self.metric_type)
        if self.metric != 'time':
            decreasing_metric = second_last_metric is None or last_metric < second_last_metric
            # selector can be activated to save resources if metric is not decreasing and not below a threshold
            activate_pattern = not decreasing_metric and not critical_metric
        else:
            # compressor activated if time is above a threshold
            activate_pattern = critical_metric

        if activate_pattern:
            return True, None, f"{self.metric}={last_metric}, {self.pattern} activated ✅"
        else:
            return False, None, f"{self.metric}={last_metric}, {self.pattern} de-activated ❌"


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
    def __init__(self, pattern, metric, model, model_name, strategy_name, clients_config, metric_type):
        self.model = model
        self.model_name = model_name
        self.metric_type = metric_type
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, predictor-based: {self.model_name}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']

        if len(new_aggregated_metrics[model_type][self.metric]) < 1:
            return True, "Less than 1 round completed, keeping default config."

        if self.metric != 'time':
            last_metric = new_aggregated_metrics[model_type][self.metric][-1]
            last_metric = last_metric / last_round_time  # normalize w.r.t. time
        else:
            last_metric = last_round_time

        iid_clients = get_no_iid_clients(self.clients_config)
        n_high, n_low = get_high_low_clients(self.clients_config)

        # TODO should be parametric w.r.t. predictor input
        prediction_w_pattern = self.model.predict([[n_high, n_low, iid_clients, True, last_metric]])[
            0]
        prediction_wo_pattern = \
            self.model.predict([[n_high, n_low, iid_clients, False, last_metric]])[0]

        expl = "{}-iid clients, predicted wo:{:.4f} vs w:{:.4f}, ".format(iid_clients, prediction_wo_pattern,
                                                                          prediction_w_pattern)

        activate_pattern = compare_w_threshold(prediction_w_pattern, prediction_wo_pattern, self.metric_type)

        if activate_pattern:
            return True, None, expl + f'{self.pattern} activated ✅'
        else:
            return False, None, expl + f'{self.pattern} de-activated ❌'


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

            if prediction_w_pattern > prediction_wo_pattern:
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

        if self.metric != 'time':
            scaler = joblib.load('predictors/bo_scaler.pkl')
        else:
            scaler = joblib.load('predictors/bo_scaler_selector_new.pkl')

        if self.metric != 'time':
            def objective(pattern_on, next_round, prev_f1_overtime):  # binary: 0 or 1
                X = [[n_high, n_low, iid_clients, pattern_on[0], next_round, prev_f1_overtime]]
                X_scaled = scaler.transform(X)
                y_pred = -self.model.predict(X_scaled)[0]  # maximize → minimize negative
                return y_pred
        else:
            def objective(policy_on, high_spec, low_spec, iid, prev_time):  # binary: 0 or 1
                X = [[high_spec, low_spec, iid, policy_on[0], prev_time]]
                X_scaled = scaler.transform(X)
                y_pred = self.model.predict(X_scaled)[0]  # minimize time
                return y_pred

        # Otherwise, we know the last round's F1 score and time
        if self.metric != 'time':
            last_metric = new_aggregated_metrics[model_type][self.metric][-1]
        else:
            last_metric = last_round_time

        # Note: any metric works to retrieve how many rounds have been performed
        next_round = len(new_aggregated_metrics[model_type]['val_f1']) + 1

        if self.metric != 'time':
            def wrapped_objective(pattern_on):
                return objective(pattern_on, next_round, last_metric / last_round_time)
        else:
            def wrapped_objective(policy_on):
                return objective(policy_on, n_high, n_low, iid_clients, last_round_time)

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


class BayesianOptimizationLocalActivationCriterion(ActivationCriterion):
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

        # TODO should be parametric 
        scaler = joblib.load('predictors/bo_scaler_hdh.pkl')

        def objective(policy_on, curr_round, last_jsd, prev_f1):  # binary: 0 or 1
            X = [[curr_round, policy_on[0], last_jsd, prev_f1]]
            X_scaled = scaler.transform(X)
            y_pred = -self.model.predict(X_scaled)[0]  # maximize → minimize negative
            return y_pred

        # Otherwise, we know the last round's F1 score and time
        metrics = self.metric.split(',')

        if len(new_aggregated_metrics[model_type][metrics[0]]) < 1:
            return True, "Less than 1 round completed, keeping default config."

        last_metrics = {m: new_aggregated_metrics[model_type][m][-1] for m in metrics}

        # TODO should be parametric w.r.t. metric name
        last_val_f1 = last_metrics['val_f1']

        next_round = len(new_aggregated_metrics[model_type][metrics[0]]) + 1

        n_high, n_low = get_high_low_clients(self.clients_config)

        apply_pattern = []
        for client_i in range(len(self.clients_config['client_details'])):
            # TODO should be parametric w.r.t. metric name
            last_jsd = last_metrics['jsd'][client_i]

            def wrapped_objective(pattern_on):
                return objective(pattern_on, next_round, last_jsd, last_val_f1)

            res = gp_minimize(wrapped_objective,  # objective fn
                              [(0, 1)],  # policy_on ∈ {0,1}
                              acq_func="EI",  # acquisition function
                              n_calls=10, random_state=42)

            if res.x[0]:
                apply_pattern.append(f"Client {client_i + 1}")

        if len(apply_pattern) == 0:
            return False, None, f'{self.pattern} de-activated ❌'
        else:
            return True, {"enabled_clients": apply_pattern}, f'{self.pattern} activated ✅ for clients: {apply_pattern}'


class ContextualBanditActivationCriterion(ActivationCriterion):
    """
    Contextual bandit (LinUCB) activation strategy.
    Fully online, no offline training cost.
    """

    def __init__(self, pattern, metric, strategy_name, clients_config, alpha=1.0):
        self.alpha = alpha

        self._cached_time = None
        self._cached_communication_time = None

        # two arms: 0 = OFF, 1 = ON
        self.arms = [0, 1]

        self.d = None  # context dimension
        self.A = {}
        self.b = {}

        # stored for update step
        self._last_context = None
        self._last_arm = None
        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, online learning with alpha {self.alpha}'

    def activate_pattern(self, args):
        """
        Decide whether to activate the pattern.
        Args is the same structure used by all other strategies.
        """
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']
        last_communication_time = args['communication_time']
        performed_rounds = len(new_aggregated_metrics[model_type]['val_f1'])
        log(INFO, args)

        # TODO should be parametric w.r.t. metric name
        if performed_rounds < 1:
            return True, "Less than 1 round completed, keeping default config."

        if performed_rounds >= 2:
            if self.metric == 'time':
                delta_metrics = - last_round_time + self._cached_time
            elif self.metric == 'communication_time':
                delta_metrics = (- last_communication_time + self._cached_communication_time) / max(
                    self._cached_communication_time, 1e-6)
            else:
                delta_metrics = new_aggregated_metrics[model_type][self.metric][-1] - \
                                new_aggregated_metrics[model_type][self.metric][-2]
            reward = delta_metrics
            log(INFO, reward)
            self.update(reward)

        if performed_rounds >= 1:
            self._cached_time = last_round_time
            self._cached_communication_time = last_communication_time

        context = self._extract_context(args)
        self._init_if_needed(context.shape[0])

        scores = {}
        for arm in self.arms:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mean = float(theta.T @ context)
            uncertainty = self.alpha * np.sqrt(float(context.T @ A_inv @ context))
            scores[arm] = mean + uncertainty

        best = max(scores.values())
        best_arms = [a for a, s in scores.items() if s == best]
        if performed_rounds <= 1:
            chosen_arm = 1  # force one ON trial
        else:
            chosen_arm = np.random.choice(best_arms)

        # store decision for later update
        self._last_context = context
        self._last_arm = chosen_arm

        activate_pattern = chosen_arm == 1

        return activate_pattern, None, f"{self.pattern} {'activated ✅' if activate_pattern else 'de-activated ❌'}"

    # --------------------------------------------------
    # ONLINE UPDATE (called after the round)
    # --------------------------------------------------
    def update(self, reward: float):
        """
        Update the bandit with the observed reward.
        """
        if self._last_context is None or self._last_arm is None:
            return

        x = self._last_context
        arm = self._last_arm

        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

        # reset stored state
        self._last_context = None
        self._last_arm = None

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------
    def _init_if_needed(self, context_dim):
        if self.d is None:
            self.d = context_dim
            for arm in self.arms:
                self.A[arm] = np.identity(self.d)
                self.b[arm] = np.zeros((self.d, 1))

    def _extract_context(self, args):
        """
        Build the context vector from args.
        This mirrors the information already used by other strategies.
        """
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        metrics = self.metric.split(',')
        last_metrics = {m: new_aggregated_metrics[model_type][m][-1] for m in metrics if 'time' not in m}
        last_round_time = args['time']
        last_communication_time = args['communication_time']
        if self.metric == 'communication_time':
            last_time_metric = last_communication_time
        else:
            last_time_metric = last_round_time
        last_val_f1 = last_metrics.get('val_f1', 0.0)
        next_round = len(new_aggregated_metrics[model_type]['val_f1']) + 1

        # adjust keys if naming differs in your codebase
        return np.array([
            next_round,
            last_val_f1,
            last_time_metric
        ], dtype=float).reshape(-1, 1)


class ContextualBanditLocalActivationCriterion(ActivationCriterion):
    """
    Contextual bandit (LinUCB) activation strategy for HDH.
    One bandit per client, fully online.
    """

    def __init__(self, pattern, metric, strategy_name, clients_config, alpha=1.0):
        self.alpha = alpha
        self.arms = [0, 1]  # 0 = OFF, 1 = ON

        # one bandit per client
        self.bandits = {}  # client_id -> {A, b, d}

        # stored for update step
        self._last_context = {}
        self._last_arm = {}
        self._cached_time = None

        super().__init__(pattern, metric, strategy_name, clients_config)

    def __str__(self):
        return f'Metric: {self.metric}, online learning (client-level) with alpha {self.alpha}'

    # --------------------------------------------------
    # DECISION
    # --------------------------------------------------
    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']

        performed_rounds = len(new_aggregated_metrics[model_type]['val_f1'])
        if performed_rounds < 1:
            return True, "Less than 1 round completed, keeping default config."

        last_round_time = args['time']
        # reward from previous round (global signal)
        if performed_rounds >= 2:
            delta_f1 = (
                    new_aggregated_metrics[model_type]['val_f1'][-1]
                    - new_aggregated_metrics[model_type]['val_f1'][-2]
            )
            delta_time = last_round_time - self._cached_time
            lambda_cost = 0.0001  # example value
            reward = delta_f1 - lambda_cost * delta_time
            self._update_all(reward)
            log(INFO, reward)

        if performed_rounds >= 1:
            self._cached_time = last_round_time

        metrics = self.metric.split(',')
        last_metrics = {m: new_aggregated_metrics[model_type][m][-1] for m in metrics}
        last_val_f1 = last_metrics['val_f1']
        next_round = performed_rounds + 1

        enabled_clients = []

        for client_i in range(len(self.clients_config['client_details'])):
            last_jsd = last_metrics['jsd'][client_i]
            context = self._extract_context(next_round, last_jsd, last_val_f1)

            self._init_client_if_needed(client_i, context.shape[0])

            scores = {}
            bandit = self.bandits[client_i]

            for arm in self.arms:
                A_inv = np.linalg.inv(bandit['A'][arm])
                theta = A_inv @ bandit['b'][arm]
                mean = float(theta.T @ context)
                uncertainty = self.alpha * np.sqrt(float(context.T @ A_inv @ context))
                scores[arm] = mean + uncertainty

            chosen_arm = max(scores, key=scores.get)

            # store decision for update
            self._last_context[client_i] = context
            self._last_arm[client_i] = chosen_arm

            if chosen_arm == 1:
                enabled_clients.append(f"Client {client_i + 1}")

        if not enabled_clients:
            return False, None, f'{self.pattern} de-activated ❌'
        else:
            return True, {"enabled_clients": enabled_clients}, (
                f'{self.pattern} activated ✅ for clients: {enabled_clients}'
            )

    # --------------------------------------------------
    # ONLINE UPDATE
    # --------------------------------------------------
    def _update_all(self, reward: float):
        """
        Update all client-level bandits with the observed reward.
        """
        for client_i in list(self._last_context.keys()):
            x = self._last_context[client_i]
            arm = self._last_arm[client_i]

            bandit = self.bandits[client_i]
            bandit['A'][arm] += x @ x.T
            bandit['b'][arm] += reward * x

        self._last_context.clear()
        self._last_arm.clear()

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------
    def _init_client_if_needed(self, client_i, context_dim):
        if client_i not in self.bandits:
            self.bandits[client_i] = {
                'd': context_dim,
                'A': {
                    arm: np.identity(context_dim) for arm in self.arms
                },
                'b': {
                    arm: np.zeros((context_dim, 1)) for arm in self.arms
                }
            }

    def _extract_context(self, next_round, last_jsd, last_val_f1):
        """
        Context vector for HDH.
        Mirrors BO features without offline scaling.
        """
        return np.array(
            [
                next_round,
                last_jsd,
                last_val_f1
            ],
            dtype=float
        ).reshape(-1, 1)
