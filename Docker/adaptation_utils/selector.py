class SelectionCriterion:
    def check(self, **args):
        return True


class LowerBoundSelectionCriterion(SelectionCriterion):
    def __init__(self, lower):
        self.lower = lower
        super().__init__()

    def check(self, res):
        return res >= self.lower


class UpperBoundSelectionCriterion(SelectionCriterion):
    def __init__(self, upper):
        self.upper = upper
        super().__init__()

    def check(self, res):
        return res <= self.upper


class ClientSelector:
    def __init__(self, json_params):
        if json_params['selection_criteria'] == 'Resource-Based':
            self.criterion = LowerBoundSelectionCriterion(json_params['selection_value'])
        # TODO: elif?

        if json_params['adaptive'].lower() == 'y':
            self.min_clients = json_params['min_clients']