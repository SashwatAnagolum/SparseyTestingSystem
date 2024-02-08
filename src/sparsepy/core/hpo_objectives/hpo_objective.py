import torch
from sparsepy.core.metrics.metric_factory import MetricFactory

class HPOObjective:
    def __init__(self, hpo_config: dict, model: torch.nn.Module):
        self.metrics = []  # Stores metric instances
        self.metric_names = []  # Stores metric names
        self.weights = []
        self.model = model
        if hpo_config:
            self.process_hpo_config(hpo_config)

    def add_objective(self, metric, weight):
        self.metrics.append(metric)
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
                self.metric_names.append(name)  # Store metric names

    @staticmethod
    def combine_metrics(metric_data: list, operation: str) -> float:
        """
        Combines multiple metric results into a single scalar value using a specified operation.

        :param metric_data: A list of dictionaries containing metric results.
        :param operation: The operation to perform on the metrics ('sum', 'mean', or 'product').
        :return: A single scalar value representing the combined result.
        """

        def to_scalar(value):
            if isinstance(value, list):
                # Convert list to sum of elements
                return sum(to_scalar(item) for item in value)
            elif hasattr(value, 'tolist'):  # Check for numpy array
                # Convert numpy array to list first
                return to_scalar(value.tolist())
            else:
                # Assume it is already a scalar
                return value

        # Flattening and converting all metrics to scalar values
        all_scalars = [to_scalar(metric[key]) for metric in metric_data for key in metric]

        # Perform the specified operation
        if operation == 'sum':
            return sum(all_scalars)
        elif operation == 'mean':
            return sum(all_scalars) / len(all_scalars) if all_scalars else 0
        elif operation == 'product':
            result = 1
            for scalar in all_scalars:
                result *= scalar
            return result
        else:
            raise ValueError("Invalid operation. Choose 'sum', 'mean', or 'product'.")
