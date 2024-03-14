import abc

import torch
from sparsepy.access_objects.models.model import Model

class Metric:
    """
    Metric: a base class for metrics.
        Metrics are used to compute different measurements requested by the user
        to provide estimations of model progress and information
        required for Dr. Rinkus' experiments.
    """

    def __init__(self, model: torch.nn.Module, name: str):
        self.model = model
        self.name = name


    @abc.abstractmethod
    def compute(self, m: Model, last_batch: torch.Tensor, labels: torch.Tensor, training: bool = True):
        """
        Computes a metric.

        Args:
            m: the model currently being trained.

            last_batch: the inputs to the current batch being evaluated

            labels: the output from the current batch being evaluated

        Returns:
            the Metric's results as a dict.
        """

    def get_name(self):
        """
        Returns the name of this metric.
        """
        return self.name