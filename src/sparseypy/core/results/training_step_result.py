from datetime import datetime

from sparseypy.core.results.result import Result


class TrainingStepResult(Result):
    """
    Training Step Result: class to store the results of a single training step.
    Attributes:
        resolution (str): The resolution of the training step.
        metrics (dict): The metrics for the training step.
    """
    def __init__(self, batch_size: int = 1):
        """
        Initializes the TrainingStepResult.

        Args:
            batch_size (int): the size of the batch of items that will be stored
                in this TrainingStepResult
        """
        super().__init__()
        self.metrics = {}
        self.batch_size = batch_size


    def add_metric(self, name: str, values: list):
        """
        Add a metric to the training step result.

        Args:
            name (str): The name of the metric.
            values (list): The values of the metric.
        """
        self.metrics[name] = values


    def get_metric(self, name: str) -> list:
        """
        Get a metric from the training step result.

        Args:
            name (str): The name of the metric.

        Returns:
            (list): The values of the metric.
        """
        return self.metrics.get(name, None)


    def get_metrics(self) -> dict:
        """
        Get the metrics from the training step result.

        Returns:
            (dict): The metrics from the training step result.
        """
        return self.metrics
