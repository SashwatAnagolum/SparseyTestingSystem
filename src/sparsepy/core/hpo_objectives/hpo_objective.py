import torch
from sparsepy.core.metrics.metric_factory import MetricFactory
class HPOObjective:
    def __init__(self, hpo_config: dict, model: torch.nn.Module,):
        self.metrics = []
        self.weights = []
        self.model = model
        if hpo_config:
            self.process_hpo_config(hpo_config)


    def add_objective(self, name, weight):
        self.metrics.append(name)
        self.weights.append(weight)

    def process_hpo_config(self, hpo_config: dict):
        """
        Processes the HPO configuration and populates the objectives.

        :param hpo_config: A dictionary representing the parsed HPO configuration.
        """
        for objective in hpo_config.get("optimization_objective", []):
            name = objective.get("name")
            weight = objective.get("weight")
            params = objective.get("params")
            if name and weight is not None:  # Ensures both name and weight are present
                if isinstance(params, dict):
                    metric = MetricFactory.create_metric(metric_name=name, model=self.model, params=params)
                else:
                    metric = MetricFactory.create_metric(metric_name=name, model=self.model)
                self.add_objective(metric, weight)
        print(self.metrics)

