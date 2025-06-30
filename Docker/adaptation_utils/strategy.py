import math
from pickle import load


def get_patterns(json_config):
    return json_config['patterns']


def get_activation_criteria(json_config):
    # TODO should work with multiple metrics
    metric_name = json_config['activation_criteria']['metrics'][0]['metric']
    threshold_config = json_config['activation_criteria']['metrics'][0]['threshold']
    strategy_name = json_config['activation_criteria']['metrics'][0]['name']
    if threshold_config['calculation_method'] == 'predictor_based':
        model_path = threshold_config['predictor']['model_path']
        with open(model_path, "rb") as f:
            model = load(f)

        return [PredictorBasedActivationCriterion(metric_name, model, model_path, strategy_name)]
    else:
        return [FixedThresholdActivationCriterion(metric_name, float(threshold_config['value']), strategy_name)]


class ActivationCriterion:
    def __init__(self, metric, strategy_name):
        self.metric = metric
        self.strategy_name = strategy_name
        pass

    def __str__(self):
        return 'Abstract activation criterion (keeps default config).'

    def activate_pattern(self, args):
        return True


class FixedThresholdActivationCriterion(ActivationCriterion):
    def __init__(self, metric, value, strategy_name):
        self.value = value
        super().__init__(metric, strategy_name)

    def __str__(self):
        return f'Metric: {self.metric}, comparison with fixed value: {self.value}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']

        last_f1 = new_aggregated_metrics[model_type][self.metric][-1]
        if len(new_aggregated_metrics[model_type][self.metric]) < 2:
            second_last_f1 = new_aggregated_metrics[None][self.metric][0]
        else:
            second_last_f1 = new_aggregated_metrics[model_type][self.metric][-2]

        decreasing_f1 = last_f1 < second_last_f1
        # TODO to add case in which thresholds are parametric (not fixed)
        insufficient_increase = not decreasing_f1 and math.fabs(last_f1 - second_last_f1) / second_last_f1 < 0.1
        low_f1 = last_f1 < self.value

        if decreasing_f1 or low_f1:
            return False, "Accuracy too low or decreasing, Selector de-activated ❌"
        else:
            return True, "Accuracy ok: Selector activated ✅"


class PredictorBasedActivationCriterion(ActivationCriterion):
    def __init__(self, metric, model, model_name, strategy_name):
        self.model = model
        self.model_name = model_name
        super().__init__(metric, strategy_name)

    def __str__(self):
        return f'Metric: {self.metric}, predictor-based: {self.model_name}'

    def activate_pattern(self, args):
        model_type = args['model_type']
        new_aggregated_metrics = args['metrics']
        last_round_time = args['time']

        last_f1 = new_aggregated_metrics[model_type][self.metric][-1]

        # TODO should be parametric w.r.t. predictor input
        if self.strategy_name == 'val_f1':
            # TODO retrieve iid/non-iid from config file, now it is always set to false
            prediction_w_pattern = self.model.predict([[False, True, last_f1]])[0][1]
            prediction_wo_pattern = self.model.predict([[False, False, last_f1]])[0][1]
        else:
            prediction_w_pattern = self.model.predict([[False, True, last_f1 / last_round_time]])[0][1]
            prediction_wo_pattern = self.model.predict([[False, False, last_f1 / last_round_time]])[0][1]

        expl = "Predicted wo:{:.4f} vs w:{:.4f}, ".format(prediction_wo_pattern, prediction_w_pattern)

        if prediction_wo_pattern > prediction_w_pattern:
            return False, expl + 'pattern de-activated ❌'
        else:
            return True, expl + 'pattern activated ✅'
